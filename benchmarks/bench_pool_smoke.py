"""Smoke test + throughput measure of modules/swapper_pool.py.

Exercises the full swap_face() path through the pool (ordering, non-blocking
submit, resilience) on real frames. Reference results (M4 Pro, gif-derived
clip, hyperswap_1b): ANE-only 14.8 FPS, ANE+GPU 17.3 FPS swap-stage.

Usage:
  python benchmarks/bench_pool_smoke.py --source /tmp/source_face.jpg \
      --target /tmp/target_clip.mp4
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules.globals

modules.globals.execution_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

import modules.processors.frame.face_swapper as fs  # noqa: E402
from modules.face_analyser import get_one_face, detect_one_face_fast  # noqa: E402
from modules.swapper_pool import SwapperPool  # noqa: E402

fs.update_status = lambda msg, scope="BENCH": print(f"[{scope}] {msg}")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(jobs, src, model_path, use_gpu, deadline_s=120):
    pool = SwapperPool(model_path, use_gpu_session=use_gpu)
    pool.start()
    pool.try_submit(jobs[0][0].copy(), jobs[0][1], src)
    deadline = time.time() + deadline_s
    while pool.pending() and time.time() < deadline:  # warmup drain
        list(pool.collect_ready())
        time.sleep(0.005)

    t0 = time.perf_counter()
    got, i = [], 0
    while len(got) < len(jobs) and time.time() < deadline:
        if i < len(jobs) and pool.try_submit(jobs[i][0].copy(), jobs[i][1], src) is not None:
            i += 1
        for idx, out, _bbox in pool.collect_ready():
            got.append(idx)
            assert out.shape == jobs[0][0].shape, "frame shape changed in pool"
        time.sleep(0.001)
    dt = time.perf_counter() - t0
    ok = len(got) == len(jobs) and got == sorted(got)
    label = "ANE+GPU" if use_gpu else "ANE-only"
    print(f"{label}: {len(got)}/{len(jobs)} frames in {dt:.2f}s = "
          f"{len(got)/dt:.1f} FPS  ordered={got == sorted(got)}")
    pool.stop()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--model", default="hyperswap_1b_256.onnx")
    args = ap.parse_args()

    src = get_one_face(cv2.imread(os.path.expanduser(args.source)))
    assert src is not None, "no face in source"
    cap = cv2.VideoCapture(os.path.expanduser(args.target))
    frames = []
    while len(frames) < args.frames:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(f, (960, 540)))
    cap.release()
    jobs = [(f, detect_one_face_fast(f)) for f in frames]
    jobs = [(f, t) for f, t in jobs if t is not None]
    print(f"jobs: {len(jobs)}")
    model_path = os.path.join(_REPO_ROOT, "models", args.model)

    ok1 = run(jobs, src, model_path, use_gpu=False)
    ok2 = run(jobs, src, model_path, use_gpu=True)
    print("POOL SMOKE:", "OK" if ok1 and ok2 else "FAILED")
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
