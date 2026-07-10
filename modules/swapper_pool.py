"""SwapperPool — GIL-relieved pipelined live swap (Faza 3.2).

Runs N HyperSwapSwapper instances (each with its OWN onnxruntime session and
per-instance lock) on different CoreML compute units, consuming swap jobs from
a shared queue with greedy dispatch: the fast ANE worker naturally takes ~2.5x
more frames than the (optional) GPU worker.

Faza 3.2 change — GIL relief. Workers used to call the FULL swap_face()
(~10-15 ms of numpy/cv2 CPU work per frame under the GIL), which starved the
main processing thread and capped the ANE-only gain at +4%. Now the worker runs
ONLY ``swapper.infer(blob, latent)`` — the one call that releases the GIL inside
session.run — and pushes the raw model outputs. ALL per-frame CPU work moves to
the main thread:
  * prepare (align + normalize + latent)   -> before try_submit
  * finalize + mouth mask + opacity + post  -> after collect_ready
so inference of frame N overlaps detection/finalization of its neighbours.

This module is intentionally UI-agnostic. Integration lives in
ui.py:_processing_thread_func and benchmarks/bench_live.py.

Usage sketch:
    pool = SwapperPool(model_path, use_gpu_session=False)
    pool.start()
    blob, latent, aimg, M = pool.reference_swapper.prepare(frame, tface, sface)
    if pool.try_submit(frame, tface, blob, latent, aimg, M) is None:
        pass  # pool full — drain collect_ready() and retry
    for res in pool.collect_ready():             # ordered
        if res.passthrough:
            out = res.frame
        else:
            out = finalize_swap(res.swapper, res.target_face, res.frame,
                                res.pred_img, res.pred_mask, res.aimg, res.M)
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
    """One pipeline result, ordered by idx for the reorder min-heap.

    For real swap jobs the worker fills pred_img/pred_mask/aimg/M/swapper and
    the caller runs finalize_swap(). For passthrough / failed jobs the caller
    emits ``frame`` unchanged (passthrough=True).
    """
    idx: int
    frame: Any = field(compare=False, default=None)
    target_face: Any = field(compare=False, default=None)
    aimg: Any = field(compare=False, default=None)
    M: Any = field(compare=False, default=None)
    pred_img: Any = field(compare=False, default=None)
    pred_mask: Any = field(compare=False, default=None)
    bbox: Optional[np.ndarray] = field(compare=False, default=None)
    swapper: Any = field(compare=False, default=None)
    passthrough: bool = field(compare=False, default=False)
    _consumed_sem: bool = field(compare=False, default=True)


class SwapperPool:
    """Pool of HyperSwapSwapper instances on distinct compute units.

    Invariants the integration MUST respect:
    - try_submit() carries monotonically increasing implicit indices; results
      are re-ordered internally, collect_ready() never yields out of order.
    - Workers run ONLY inference (GIL relief). prepare() runs on the caller
      before submit; finalize + mouth mask + post run on the caller after
      collect. Never move CPU work back into the worker.
    - Detection/tracking stays OUTSIDE the pool (FaceTracker is stateful and
      strictly sequential — optical flow needs the previous frame).
    - Post-processing (sharpen/interpolation) stays AFTER collection.
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

    @property
    def reference_swapper(self) -> Any:
        """A ready HyperSwapSwapper for the caller's prepare()/finalize() calls.

        prepare() and finalize() touch no session state, so using any pooled
        instance from the main thread is safe even while its worker runs
        infer() under that instance's lock."""
        return self._swappers[0] if self._swappers else None

    # -- data path -----------------------------------------------------

    def try_submit(self, frame: np.ndarray, target_face: Any,
                   blob: np.ndarray, latent: np.ndarray,
                   aimg: np.ndarray, M: np.ndarray) -> Optional[int]:
        """Enqueue one inference job (tensors already prepared by the caller).
        NON-BLOCKING: returns None when max_in_flight jobs are pending — the
        caller must drain collect_ready() first and retry.

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
        bbox = getattr(target_face, "bbox", None)
        bbox = bbox.astype(int) if bbox is not None else None
        self._in_q.put((idx, frame, target_face, blob, latent, aimg, M, bbox))
        return idx

    def submit_passthrough(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None,
                          face: "Optional[Any]" = None) -> int:
        """Enqueue a no-op result for frames that need no swap (no face
        detected, enhancer-only, etc.). The frame passes through the reorder
        heap untouched, preserving emit order.

        Returns the assigned sequence index so the caller can correlate.
        The semaphore is *not* consumed — this is intentionally free.
        """
        idx = self._next_submit
        self._next_submit += 1
        with self._out_lock:
            heapq.heappush(self._out_heap, _Result(
                idx, frame=frame, target_face=face, bbox=bbox,
                passthrough=True, _consumed_sem=False))
        return idx

    def collect_ready(self) -> Iterator[_Result]:
        """Yield each finished _Result that is next in sequence. Non-blocking;
        yields nothing if the next frame in order is still being processed.

        Only releases the in_flight semaphore for real swap jobs (not
        passthrough results), so the semaphore stays accurate.
        """
        while True:
            with self._out_lock:
                if not self._out_heap or self._out_heap[0].idx != self._next_emit:
                    return
                res = heapq.heappop(self._out_heap)
                self._next_emit += 1
            if res._consumed_sem:
                self._in_flight.release()
            yield res

    def pending(self) -> int:
        """Jobs submitted but not yet emitted (queued + running + reordering)."""
        return self._next_submit - self._next_emit

    # -- internals -----------------------------------------------------

    def _worker(self, name: str, swapper: Any) -> None:
        # ONLY inference here. session.run releases the GIL, so the main thread
        # keeps running detection/finalization while this executes. Any numpy or
        # cv2 work added here would re-introduce the GIL starvation this refactor
        # exists to remove — keep it inference-only.
        while not self._stop.is_set():
            item = self._in_q.get()
            if item is None:
                break
            idx, frame, target_face, blob, latent, aimg, M, bbox = item
            try:
                pred_img, pred_mask = swapper.infer(blob, latent)
                res = _Result(idx, frame=frame, target_face=target_face,
                              aimg=aimg, M=M, pred_img=pred_img, pred_mask=pred_mask,
                              bbox=bbox, swapper=swapper, passthrough=False,
                              _consumed_sem=True)
            except Exception as e:  # never kill the pipeline on one bad frame
                print(f"SwapperPool[{name}]: infer failed on frame {idx}: {e}")
                res = _Result(idx, frame=frame, target_face=target_face,
                              bbox=None, passthrough=True, _consumed_sem=True)
            with self._out_lock:
                heapq.heappush(self._out_heap, res)


__all__ = ["SwapperPool", "BASE_COREML_CFG"]
