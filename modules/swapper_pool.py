"""SwapperPool — foundations for the dual-session pipelined live swap.

Runs N HyperSwapSwapper instances (each with its OWN onnxruntime session and
per-instance lock) on different CoreML compute units, consuming swap jobs
from a shared queue with greedy dispatch: the fast ANE worker naturally takes
~2.5x more frames than the GPU worker.

Measured on M4 Pro (benchmarks/bench_dual_session.py, hyperswap_1b_256):
  single ANE serial      19.5 swaps/s
  single GPU serial       6.6 swaps/s (contended; 7.1/s isolated)
  ANE+GPU greedy pool    22.9 swaps/s  (+17.6%)

This module is intentionally UI-agnostic. Integration into
ui.py:_processing_thread_func is described step by step in TODO.md.

Usage sketch:
    pool = SwapperPool(model_path, use_gpu_session=True)
    pool.start()
    if pool.try_submit(frame, target_face, source_face) is None:
        pass  # pool full — drain collect_ready() and retry
    for idx, swapped_frame, bbox in pool.collect_ready():  # ordered
        ...display...
    pool.stop()
"""

from __future__ import annotations

import heapq
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import numpy as np

BASE_COREML_CFG = {
    "ModelFormat": "MLProgram",
    "SpecializationStrategy": "FastPrediction",
    "AllowLowPrecisionAccumulationOnGPU": 1,
    "EnableOnSubgraphs": 1,
}


@dataclass(order=True)
class _Result:
    idx: int
    frame: Any = field(compare=False)
    bbox: Optional[np.ndarray] = field(compare=False, default=None)


class SwapperPool:
    """Pool of HyperSwapSwapper instances on distinct compute units.

    Invariants the integration MUST respect (see TODO.md):
    - submit() calls carry monotonically increasing implicit indices; results
      are re-ordered internally, collect_ready() never yields out of order.
    - Detection/tracking stays OUTSIDE the pool (FaceTracker is stateful and
      strictly sequential — optical flow needs the previous frame).
    - Post-processing (sharpen/interpolation) stays AFTER collection
      (interpolation blends consecutive frames, so it needs ordered input).
    - max_in_flight bounds latency: with 2 workers, 3 means at most one job
      queued behind the two running ones (~1-2 frames of extra latency).
    """

    def __init__(self, model_path: str, use_gpu_session: bool = True,
                 max_in_flight: int = 3):
        self.model_path = model_path
        self.use_gpu_session = use_gpu_session
        self.max_in_flight = max_in_flight
        self._in_q: "queue.Queue" = queue.Queue()
        self._out_lock = threading.Lock()
        self._out_heap: list = []          # min-heap of _Result by idx
        self._next_submit = 0
        self._next_emit = 0
        self._in_flight = threading.Semaphore(max_in_flight)
        self._threads: list = []
        self._stop = threading.Event()
        self._swappers: list = []

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        from modules.processors.frame.face_swapper import HyperSwapSwapper
        configs = [("ANE", dict(BASE_COREML_CFG, MLComputeUnits="CPUAndNeuralEngine"))]
        if self.use_gpu_session:
            configs.append(("GPU", dict(BASE_COREML_CFG, MLComputeUnits="CPUAndGPU")))
        for name, cfg in configs:
            swapper = HyperSwapSwapper(
                self.model_path,
                providers=[("CoreMLExecutionProvider", cfg), "CPUExecutionProvider"],
            )
            self._swappers.append(swapper)
            t = threading.Thread(target=self._worker, args=(name, swapper),
                                 daemon=True, name=f"swap-{name}")
            self._threads.append(t)
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        """Stop workers and release sessions. Safe to call twice."""
        self._stop.set()
        for _ in self._threads:
            self._in_q.put(None)  # wake blocked workers
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads.clear()
        self._swappers.clear()  # drop sessions (CoreML frees on GC)

    # -- data path -----------------------------------------------------

    def try_submit(self, frame: np.ndarray, target_face: Any,
                   source_face: Any) -> Optional[int]:
        """Enqueue one swap job. NON-BLOCKING: returns None when
        max_in_flight jobs are pending — the caller must drain
        collect_ready() first and retry.

        Deliberately never blocks: a blocking submit deadlocks the pipeline
        when results complete out of order (fast ANE frames pile up in the
        reorder heap behind a slow GPU frame, the semaphore stays exhausted,
        and the blocked caller can never reach collect_ready() to drain).
        Found by the smoke test; do not "simplify" this back to blocking.
        """
        if not self._in_flight.acquire(blocking=False):
            return None
        idx = self._next_submit
        self._next_submit += 1
        self._in_q.put((idx, frame, target_face, source_face))
        return idx

    def collect_ready(self) -> Iterator[tuple]:
        """Yield (idx, frame, bbox) for every finished job that is next in
        sequence. Non-blocking; yields nothing if the next frame in order
        is still being processed."""
        while True:
            with self._out_lock:
                if not self._out_heap or self._out_heap[0].idx != self._next_emit:
                    return
                res = heapq.heappop(self._out_heap)
                self._next_emit += 1
            self._in_flight.release()
            yield res.idx, res.frame, res.bbox

    def pending(self) -> int:
        """Jobs submitted but not yet emitted (queued + running + reordering)."""
        return self._next_submit - self._next_emit

    # -- internals -----------------------------------------------------

    def _worker(self, name: str, swapper: Any) -> None:
        # swap_face(swapper=...) keeps every post-swap feature (mouth mask,
        # poisson blend, opacity) identical to the single-session path while
        # routing inference to this worker's dedicated session.
        from modules.processors.frame.face_swapper import swap_face
        while not self._stop.is_set():
            item = self._in_q.get()
            if item is None:
                break
            idx, frame, target_face, source_face = item
            try:
                out = swap_face(source_face, target_face, frame, swapper=swapper)
                if not isinstance(out, np.ndarray):
                    out = frame
                bbox = getattr(target_face, "bbox", None)
                bbox = bbox.astype(int) if bbox is not None else None
            except Exception as e:  # never kill the pipeline on one bad frame
                print(f"SwapperPool[{name}]: swap failed on frame {idx}: {e}")
                out, bbox = frame, None
            with self._out_lock:
                heapq.heappush(self._out_heap, _Result(idx, out, bbox))


__all__ = ["SwapperPool", "BASE_COREML_CFG"]
