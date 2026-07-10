"""Validate the core assumption of the dual-session pipeline refactor:
that a hyperswap session on the Neural Engine and a second one on the GPU
can run inference CONCURRENTLY with near-zero contention.

Measures aggregate throughput (swaps/sec) of:
  A) single ANE session, serial          (baseline, ~16-18/s expected)
  B) ANE + GPU sessions, one thread each (target, ~1000/56 + 1000/140 = ~25/s)

If (B) is not clearly better than (A), the whole refactor in TODO.md is
NOT worth doing — stop and re-evaluate.

Usage: python benchmarks/bench_dual_session.py [--model hyperswap_1b_256.onnx]
"""
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np
import onnxruntime as ort

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

BASE_CFG = {
    "ModelFormat": "MLProgram",
    "SpecializationStrategy": "FastPrediction",
    "AllowLowPrecisionAccumulationOnGPU": 1,
    "EnableOnSubgraphs": 1,
}
ANE = dict(BASE_CFG, MLComputeUnits="CPUAndNeuralEngine")
GPU = dict(BASE_CFG, MLComputeUnits="CPUAndGPU")

WARMUP = 5


def make_session(path, cfg):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        path, sess_options=so,
        providers=[("CoreMLExecutionProvider", cfg), "CPUExecutionProvider"],
    )


def make_feed():
    rng = np.random.RandomState(42)
    blob = rng.rand(1, 3, 256, 256).astype(np.float32) * 2 - 1
    lat = rng.rand(1, 512).astype(np.float32)
    lat /= np.linalg.norm(lat)
    return {"target": blob, "source": lat}


def run_serial(sess, jobs):
    feed = make_feed()
    outs = [o.name for o in sess.get_outputs()]
    for _ in range(WARMUP):
        sess.run(outs, {k: v.copy() for k, v in feed.items()})
    t0 = time.perf_counter()
    for _ in range(jobs):
        sess.run(outs, {k: v.copy() for k, v in feed.items()})
    dt = time.perf_counter() - t0
    return jobs / dt


def run_pool(sessions, jobs):
    """Greedy dispatch: each worker owns one session and pulls from a shared
    queue — exactly the SwapperPool model from modules/swapper_pool.py."""
    q = queue.Queue()
    for i in range(jobs):
        q.put(i)
    counts = {}

    feed = make_feed()

    def worker(name, sess):
        outs = [o.name for o in sess.get_outputs()]
        for _ in range(WARMUP):
            sess.run(outs, {k: v.copy() for k, v in feed.items()})
        done = 0
        barrier.wait()
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
            sess.run(outs, {k: v.copy() for k, v in feed.items()})
            done += 1
        counts[name] = done

    barrier = threading.Barrier(len(sessions) + 1)
    threads = [threading.Thread(target=worker, args=(n, s), daemon=True)
               for n, s in sessions]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    dt = time.perf_counter() - t0
    return jobs / dt, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hyperswap_1b_256.onnx")
    ap.add_argument("--jobs", type=int, default=60)
    args = ap.parse_args()
    path = os.path.join(_REPO_ROOT, "models", args.model)

    print(f"model: {args.model}, jobs: {args.jobs}")
    sess_ane = make_session(path, ANE)
    fps_a = run_serial(sess_ane, args.jobs)
    print(f"A) single ANE serial:      {fps_a:6.2f} swaps/s")

    sess_gpu = make_session(path, GPU)
    fps_gpu = run_serial(sess_gpu, args.jobs // 2)
    print(f"   (single GPU serial:     {fps_gpu:6.2f} swaps/s)")

    fps_b, counts = run_pool([("ANE", sess_ane), ("GPU", sess_gpu)], args.jobs)
    print(f"B) ANE+GPU greedy pool:    {fps_b:6.2f} swaps/s   split: {counts}")

    gain = (fps_b / fps_a - 1) * 100
    print(f"\ngain vs single ANE: {gain:+.1f}%  "
          f"({'REFACTOR VIABLE' if gain > 25 else 'NOT WORTH IT — see TODO.md abort criteria'})")


if __name__ == "__main__":
    main()
