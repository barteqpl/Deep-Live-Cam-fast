"""Headless benchmark of the LIVE webcam hot-path (ui.py:_processing_thread_func).

Replays frames from a video file through the exact same per-frame pipeline the
live preview uses: frame.copy -> detect/track (DETECT_EVERY_N=3) -> swap_face
-> apply_post_processing. Frames are preloaded to RAM so video decode does not
pollute the measurement (in live mode decode happens in the capture thread).

Usage:
  python benchmarks/bench_live.py --source source_face.jpg \
      --target clip.mov --frames 120 \
      --save-frame /tmp/frame_baseline.png
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules.globals

# Match the live defaults regardless of switch_states.json
modules.globals.execution_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
modules.globals.many_faces = False
modules.globals.map_faces = False
modules.globals.mouth_mask = False
modules.globals.poisson_blend = False
modules.globals.opacity = 1.0
modules.globals.enable_interpolation = False
modules.globals.show_fps = False
modules.globals.live_mirror = False
modules.globals.frame_processors = ["face_swapper"]

from modules.face_analyser import get_one_face, detect_one_face_fast, FaceTracker  # noqa: E402
from modules.processors.frame import face_swapper as fs  # noqa: E402

# Headless: update_status tries to touch the Tk status label
fs.update_status = lambda msg, scope="BENCH": print(f"[{scope}] {msg}")

DETECT_EVERY_N = 3
WARMUP = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--detect-every", type=int, default=DETECT_EVERY_N)
    ap.add_argument("--save-frame", default=None, help="save processed frame #30 as PNG")
    ap.add_argument("--sharpness", type=float, default=None)
    args = ap.parse_args()

    if args.sharpness is not None:
        modules.globals.sharpness = args.sharpness

    src_img = cv2.imread(os.path.expanduser(args.source))
    assert src_img is not None, f"cannot read source {args.source}"
    source_face = get_one_face(src_img)
    assert source_face is not None, "no face in source image"

    # Preload target frames, resized to live-preview working resolution
    cap = cv2.VideoCapture(os.path.expanduser(args.target))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(f, (args.width, args.height), interpolation=cv2.INTER_AREA))
    cap.release()
    assert frames, "no frames read from target"
    print(f"preloaded {len(frames)} frames at {args.width}x{args.height}")

    # Instrument raw inference time of the swapper session
    swapper = fs.get_face_swapper()
    assert swapper is not None
    orig_run = swapper.session.run
    infer_t = [0.0, 0]

    def timed_run(*a, **kw):
        t0 = time.perf_counter()
        out = orig_run(*a, **kw)
        infer_t[0] += time.perf_counter() - t0
        infer_t[1] += 1
        return out

    swapper.session.run = timed_run

    tracker = FaceTracker()
    cached_target_face = None
    stages = {k: 0.0 for k in ("copy", "detect", "track", "swap", "post")}
    n_detect = 0
    total_t0 = None
    processed = 0
    saved = False

    idx = 0
    while processed < args.frames + WARMUP:
        frame = frames[idx % len(frames)]
        idx += 1
        processed += 1
        if processed == WARMUP + 1:
            total_t0 = time.perf_counter()
            for k in stages:
                stages[k] = 0.0
            infer_t[0] = 0.0
            infer_t[1] = 0
            n_detect = 0

        t0 = time.perf_counter()
        temp_frame = frame.copy()
        t1 = time.perf_counter()
        stages["copy"] += t1 - t0

        run_detection = (processed % args.detect_every == 1) or cached_target_face is None
        if run_detection:
            t0 = time.perf_counter()
            raw_face = detect_one_face_fast(temp_frame)  # mirrors ui.py live path
            raw_faces = [raw_face] if raw_face is not None else []
            tracked = tracker.update(temp_frame, raw_faces)
            cached_target_face = tracked[0] if tracked else None
            stages["detect"] += time.perf_counter() - t0
            n_detect += 1
        else:
            t0 = time.perf_counter()
            tracked = tracker.update(temp_frame)
            cached_target_face = tracked[0] if tracked else None
            stages["track"] += time.perf_counter() - t0

        swapped_bboxes = []
        if cached_target_face is not None:
            t0 = time.perf_counter()
            temp_frame = fs.swap_face(source_face, cached_target_face, temp_frame)
            stages["swap"] += time.perf_counter() - t0
            if getattr(cached_target_face, "bbox", None) is not None:
                swapped_bboxes.append(cached_target_face.bbox.astype(int))

        t0 = time.perf_counter()
        temp_frame = fs.apply_post_processing(temp_frame, swapped_bboxes)
        stages["post"] += time.perf_counter() - t0

        if args.save_frame and processed == WARMUP + 30 and not saved:
            cv2.imwrite(os.path.expanduser(args.save_frame), temp_frame)
            saved = True

    total = time.perf_counter() - total_t0
    n = args.frames
    print(f"\n=== bench_live results ({n} frames, warmup {WARMUP} excluded) ===")
    print(f"FPS: {n / total:.2f}   total {total*1000:.0f} ms   {total/n*1000:.2f} ms/frame")
    n_track = n - n_detect
    print(f"copy   : {stages['copy']/n*1000:7.2f} ms/frame")
    print(f"detect : {stages['detect']/max(n_detect,1)*1000:7.2f} ms/detect-frame  (x{n_detect})")
    print(f"track  : {stages['track']/max(n_track,1)*1000:7.2f} ms/track-frame   (x{n_track})")
    print(f"swap   : {stages['swap']/n*1000:7.2f} ms/frame  (infer {infer_t[0]/max(infer_t[1],1)*1000:.2f} ms x{infer_t[1]})")
    print(f"post   : {stages['post']/n*1000:7.2f} ms/frame")
    accounted = sum(stages.values())
    print(f"accounted: {accounted/n*1000:.2f} ms/frame of {total/n*1000:.2f}")


if __name__ == "__main__":
    main()
