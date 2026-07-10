"""Headless benchmark of the LIVE webcam hot-path (ui.py:_processing_thread_func).

Replays frames from a video file through the exact same per-frame pipeline the
live preview uses: frame.copy -> detect/track (DETECT_EVERY_N=3) -> swap_face
-> apply_post_processing. Frames are preloaded to RAM so video decode does not
pollute the measurement (in live mode decode happens in the capture thread).

Usage:
  python benchmarks/bench_live.py --source source_face.jpg \
      --target clip.mov --frames 120 \
      --save-frame /tmp/frame_baseline.png

  python benchmarks/bench_live.py --source source_face.jpg \
      --target clip.mov --frames 120 --model hyperswap-1b --pool
"""
import argparse
import os
import sys
import time

import cv2

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

from modules.face_analyser import get_one_face, detect_one_face_fast, ensure_landmarks, FaceTracker  # noqa: E402
from modules.processors.frame import face_swapper as fs  # noqa: E402

# Headless: update_status tries to touch the Tk status label
fs.update_status = lambda msg, scope="BENCH": print(f"[{scope}] {msg}")

DETECT_EVERY_N = 3
WARMUP = 6


def _run_serial(args, frames, source_face, enhancer_mod):
    """Serial pipeline — baseline."""
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
    stages = {k: 0.0 for k in ("copy", "detect", "track", "swap", "enh", "post")}
    n_detect = 0
    n = args.frames

    saved = False
    total_t0 = None
    for i in range(WARMUP + n):
        frame = frames[i % len(frames)]

        t0 = time.perf_counter()
        temp_frame = frame.copy()
        t1 = time.perf_counter()
        stages["copy"] += t1 - t0

        if i >= WARMUP and total_t0 is None:
            total_t0 = time.perf_counter()
            for k in stages:
                stages[k] = 0.0
            infer_t[0] = 0.0
            infer_t[1] = 0
            n_detect = 0

        run_detection = (i % args.detect_every == args.detect_every - 1) \
            or cached_target_face is None
        if run_detection:
            t0 = time.perf_counter()
            raw_face = detect_one_face_fast(temp_frame)
            raw_faces = [raw_face] if raw_face is not None else []
            if modules.globals.mouth_mask and raw_faces:
                ensure_landmarks(temp_frame, raw_faces)
            tracked = tracker.update(temp_frame, raw_faces)
            cached_target_face = tracked[0] if tracked else None
            stages["detect"] += time.perf_counter() - t0
            n_detect += 1 if i >= WARMUP else 0
        else:
            t0 = time.perf_counter()
            tracked = tracker.update(temp_frame)
            cached_target_face = tracked[0] if tracked else None
            stages["track"] += time.perf_counter() - t0

        swapped_bboxes = []
        if cached_target_face is not None:
            t0 = time.perf_counter()
            temp_frame = fs.swap_face(source_face, cached_target_face, temp_frame)
            stages["swap"] += time.perf_counter() - t0 if i >= WARMUP else 0.0
            if getattr(cached_target_face, "bbox", None) is not None:
                swapped_bboxes.append(cached_target_face.bbox.astype(int))

        if enhancer_mod is not None and cached_target_face is not None:
            t0 = time.perf_counter()
            temp_frame = enhancer_mod.process_frame(
                None, temp_frame, detected_faces=[cached_target_face]
            )
            stages["enh"] += time.perf_counter() - t0 if i >= WARMUP else 0.0

        t0 = time.perf_counter()
        temp_frame = fs.apply_post_processing(temp_frame, swapped_bboxes)
        stages["post"] += time.perf_counter() - t0 if i >= WARMUP else 0.0

        if args.save_frame and i == WARMUP + 30 and not saved:
            cv2.imwrite(os.path.expanduser(args.save_frame), temp_frame)
            saved = True

    total = time.perf_counter() - total_t0
    n = args.frames
    print(f"\n=== bench_live results (serial, {n} frames, warmup {WARMUP}) ===")
    print(f"FPS: {n / total:.2f}   total {total*1000:.0f} ms   "
          f"{total/n*1000:.2f} ms/frame")
    n_track = n - n_detect
    print(f"copy   : {stages['copy']/n*1000:7.2f} ms/frame")
    if n_detect:
        print(f"detect : {stages['detect']/max(n_detect,1)*1000:7.2f} ms/detect  x{n_detect}")
    if n_track:
        print(f"track  : {stages['track']/max(n_track,1)*1000:7.2f} ms/track   x{n_track}")
    print(f"swap   : {stages['swap']/n*1000:7.2f} ms/frame  "
          f"(infer {infer_t[0]/max(infer_t[1],1)*1000:.2f} ms x{infer_t[1]})")
    if stages["enh"]:
        print(f"enh    : {stages['enh']/n*1000:7.2f} ms/frame")
    print(f"post   : {stages['post']/n*1000:7.2f} ms/frame")


def _run_pool(args, frames, source_face, enhancer_mod):
    """Pipelined submit/collect with SwapperPool."""
    from modules.swapper_pool import SwapperPool
    modules.globals.dual_session = args.dual_session
    pool = SwapperPool(
        os.path.join(fs.models_dir, fs.get_model_name()),
        use_gpu_session=args.dual_session,
        max_in_flight=3,
    )
    pool.start()

    tracker = FaceTracker()
    cached_target_face = None
    stages = {k: 0.0 for k in ("detect", "track", "enh", "post")}
    n_detect = 0
    n = args.frames
    total_f = n + WARMUP
    saved = False

    # 1. Submit all frames with blocking retry + drain
    submitted = 0
    collected = 0
    total_t0 = None
    # detection stage timer (aggregated inside the submit loop)
    detect_timer = 0.0
    track_timer = 0.0
    n_detect_total = 0
    n_track_total = 0

    while submitted < total_f:
        # Drain first
        for _idx, out_f, bbox, face in pool.collect_ready():
            collected += 1
            if collected == WARMUP + 1 and total_t0 is None:
                total_t0 = time.perf_counter()
                for k in stages:
                    stages[k] = 0.0
                detect_timer = 0.0
                track_timer = 0.0
                n_detect_total = 0
                n_track_total = 0
            if collected > WARMUP:
                t0 = time.perf_counter()
                if enhancer_mod is not None and face is not None:
                    out_f = enhancer_mod.process_frame(None, out_f, detected_faces=[face])
                p_bboxes = [bbox] if bbox is not None else []
                out_f = fs.apply_post_processing(out_f, p_bboxes)
                stages["post"] += time.perf_counter() - t0
            if args.save_frame and collected == WARMUP + 30 and not saved:
                cv2.imwrite(os.path.expanduser(args.save_frame), out_f)
                saved = True

        # Submit (ALL frames, including warmup — warmup triggers ANE compilation)
        frame = frames[submitted % len(frames)]

        temp_frame = frame.copy()
        run_det = (submitted % args.detect_every == 0) or cached_target_face is None
        if run_det:
            t0 = time.perf_counter()
            raw_face = detect_one_face_fast(temp_frame)
            raw_faces = [raw_face] if raw_face is not None else []
            if modules.globals.mouth_mask and raw_faces:
                ensure_landmarks(temp_frame, raw_faces)
            tracked = tracker.update(temp_frame, raw_faces)
            cached_target_face = tracked[0] if tracked else None
            t1 = time.perf_counter()
            if submitted >= WARMUP:
                detect_timer += t1 - t0
                n_detect_total += 1
        else:
            t0 = time.perf_counter()
            tracked = tracker.update(temp_frame)
            cached_target_face = tracked[0] if tracked else None
            t1 = time.perf_counter()
            if submitted >= WARMUP:
                track_timer += t1 - t0
                n_track_total += 1

        ok = True
        if cached_target_face is not None:
            ok = pool.try_submit(temp_frame, cached_target_face, source_face)
        else:
            pool.submit_passthrough(temp_frame, face=None)
        if ok is None:
            time.sleep(0.01)
            continue  # retry submit on next iteration after drain
        submitted += 1

    # Drain remaining
    while collected < total_f:
        for _idx, out_f, bbox, face in pool.collect_ready():
            collected += 1
            if collected == WARMUP + 1 and total_t0 is None:
                total_t0 = time.perf_counter()
                for k in stages:
                    stages[k] = 0.0
            if collected > WARMUP:
                t0 = time.perf_counter()
                if enhancer_mod is not None and face is not None:
                    out_f = enhancer_mod.process_frame(None, out_f, detected_faces=[face])
                p_bboxes = [bbox] if bbox is not None else []
                out_f = fs.apply_post_processing(out_f, p_bboxes)
                stages["post"] += time.perf_counter() - t0
        if collected < total_f:
            time.sleep(0.01)

    total = time.perf_counter() - total_t0
    pool.stop()

    print(f"\n=== bench_live results (--pool, {n} frames, warmup {WARMUP}) ===")
    print(f"FPS: {n / total:.2f}   total {total*1000:.0f} ms   "
          f"{total/n*1000:.2f} ms/frame")
    if n_detect_total:
        print(f"detect : {detect_timer/max(n_detect_total,1)*1000:7.2f}"
              f" ms/detect  x{n_detect_total}")
    if n_track_total:
        print(f"track  : {track_timer/max(n_track_total,1)*1000:7.2f}"
              f" ms/track   x{n_track_total}")
    if stages["enh"]:
        print(f"enh    : {stages['enh']/n*1000:7.2f} ms/frame")
    print(f"post   : {stages['post']/n*1000:7.2f} ms/frame")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--detect-every", type=int, default=DETECT_EVERY_N)
    ap.add_argument("--save-frame", default=None,
                    help="save processed frame #30 as PNG")
    ap.add_argument("--sharpness", type=float, default=None)
    ap.add_argument("--model", default=None,
                    help="swapper model: inswapper|simswap|hififace|hyperswap|hyperswap-1b")
    ap.add_argument("--enhancer", default=None, choices=["gfpgan", "gpen256"],
                    help="also run the live face enhancer per frame")
    ap.add_argument("--mouth-mask", action="store_true",
                    help="enable mouth masking (mirrors the ui.py live path)")
    ap.add_argument("--pool", action="store_true",
                    help="use SwapperPool submit/collect (Faza 1 pipeline)")
    ap.add_argument("--dual-session", action="store_true",
                    help="enable second ONNX session on GPU alongside ANE")
    args = ap.parse_args()

    if args.sharpness is not None:
        modules.globals.sharpness = args.sharpness
    if args.model:
        modules.globals.swapper_model = args.model
    if args.mouth_mask:
        modules.globals.mouth_mask = True

    enhancer_mod = None
    if args.enhancer:
        modules.globals.live_enhancer_model = args.enhancer
        from modules.processors.frame import face_enhancer as enhancer_mod
        enhancer_mod.update_status = lambda msg, scope="BENCH": print(f"[{scope}] {msg}")

    src_img = cv2.imread(os.path.expanduser(args.source))
    assert src_img is not None, f"cannot read source {args.source}"
    source_face = get_one_face(src_img)
    assert source_face is not None, "no face in source image"

    cap = cv2.VideoCapture(os.path.expanduser(args.target))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(f, (args.width, args.height),
                                  interpolation=cv2.INTER_AREA))
    cap.release()
    assert frames, "no frames read from target"
    print(f"preloaded {len(frames)} frames at {args.width}x{args.height}", flush=True)

    print("[bench] starting pool...", flush=True) if args.pool else None
    if args.pool:
        _run_pool(args, frames, source_face, enhancer_mod)
    else:
        _run_serial(args, frames, source_face, enhancer_mod)


if __name__ == "__main__":
    main()
