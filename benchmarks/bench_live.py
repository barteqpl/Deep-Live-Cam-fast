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


def _pct(vals, p):
    """Linear-interpolated percentile of `vals` (list of ms). Empty -> 0.0."""
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _print_frame_times(intervals):
    """Pacing is the acceptance criterion, not just average FPS."""
    if not intervals:
        return
    p50, p95, mx = _pct(intervals, 50), _pct(intervals, 95), max(intervals)
    ratio = p95 / p50 if p50 > 0 else 0.0
    print(f"frame-time: p50 {p50:6.1f} ms  p95 {p95:6.1f} ms  "
          f"max {mx:6.1f} ms  (p95/p50 {ratio:.2f}x, n={len(intervals)})")


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
    intervals = []          # per-emitted-frame wall interval (ms), post-warmup
    last_emit = None
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

        if i >= WARMUP:
            now = time.perf_counter()
            if last_emit is not None:
                intervals.append((now - last_emit) * 1000.0)
            last_emit = now

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
    _print_frame_times(intervals)


def _run_pool(args, frames, source_face, enhancer_mod):
    """Pipelined submit/collect with SwapperPool (Faza 3.2, GIL relief).

    Mirrors ui.py: prepare() (align + latent) runs on THIS thread before submit,
    the worker runs infer() only, and finalize_swap() + post run here on collect.
    """
    from modules.swapper_pool import SwapperPool
    modules.globals.dual_session = args.dual_session
    pool = SwapperPool(
        os.path.join(fs.models_dir, fs.get_model_name()),
        use_gpu_session=args.dual_session,
        max_in_flight=3,
    )
    pool.start()
    ref = pool.reference_swapper

    tracker = FaceTracker()
    cached_target_face = None
    stages = {k: 0.0 for k in ("detect", "track", "prep", "final", "enh", "post")}
    n = args.frames
    total_f = n + WARMUP
    saved = False

    submitted = 0
    collected = 0
    total_t0 = None
    detect_timer = 0.0
    track_timer = 0.0
    n_detect_total = 0
    n_track_total = 0
    intervals = []          # per-emitted-frame wall interval (ms), post-warmup
    last_emit = None

    def _drain():
        nonlocal collected, total_t0, detect_timer, track_timer
        nonlocal n_detect_total, n_track_total, saved, last_emit
        for res in pool.collect_ready():
            collected += 1
            if collected == WARMUP + 1 and total_t0 is None:
                total_t0 = time.perf_counter()
                for k in stages:
                    stages[k] = 0.0
                detect_timer = 0.0
                track_timer = 0.0
                n_detect_total = 0
                n_track_total = 0
            out_f = res.frame
            if collected > WARMUP:
                t0 = time.perf_counter()
                if not res.passthrough and res.pred_img is not None:
                    out_f = fs.finalize_swap(
                        res.swapper, res.target_face, out_f,
                        res.pred_img, res.pred_mask, res.aimg, res.M)
                stages["final"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                if enhancer_mod is not None and res.target_face is not None:
                    out_f = enhancer_mod.process_frame(
                        None, out_f, detected_faces=[res.target_face])
                stages["enh"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                p_bboxes = [res.bbox] if res.bbox is not None else []
                out_f = fs.apply_post_processing(out_f, p_bboxes)
                stages["post"] += time.perf_counter() - t0
                now = time.perf_counter()
                if last_emit is not None:
                    intervals.append((now - last_emit) * 1000.0)
                last_emit = now
            # Save keyed on the SUBMIT index (res.idx), not the collected
            # counter, so it matches the serial path's frames[WARMUP+30] exactly
            # (serial saves at loop index i == WARMUP+30). Same source frame ->
            # a meaningful PSNR comparison.
            if args.save_frame and res.idx == WARMUP + 30 and not saved:
                cv2.imwrite(os.path.expanduser(args.save_frame), out_f)
                saved = True

    while submitted < total_f:
        # detect/track + prepare exactly ONCE per frame (ALL frames, including
        # warmup — warmup triggers ANE compilation). Retrying the *submit* only,
        # never the detection, mirrors the live thread's per-frame work.
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

        if cached_target_face is not None:
            t0 = time.perf_counter()
            blob, latent, aimg, M = ref.prepare(temp_frame, cached_target_face, source_face)
            if submitted >= WARMUP:
                stages["prep"] += time.perf_counter() - t0
            # Retry submit (draining between) until the pool accepts it. Draining
            # inside the retry releases the semaphore, so this cannot deadlock.
            while pool.try_submit(temp_frame, cached_target_face, blob, latent, aimg, M) is None:
                _drain()
                time.sleep(0.002)
        else:
            pool.submit_passthrough(temp_frame, face=None)
        submitted += 1
        _drain()

    # Drain remaining
    while collected < total_f:
        _drain()
        if collected < total_f:
            time.sleep(0.005)

    total = time.perf_counter() - total_t0
    pool.stop()

    tag = "ANE+GPU" if args.dual_session else "ANE-only"
    print(f"\n=== bench_live results (--pool {tag}, {n} frames, warmup {WARMUP}) ===")
    print(f"FPS: {n / total:.2f}   total {total*1000:.0f} ms   "
          f"{total/n*1000:.2f} ms/frame")
    if n_detect_total:
        print(f"detect : {detect_timer/max(n_detect_total,1)*1000:7.2f}"
              f" ms/detect  x{n_detect_total}")
    if n_track_total:
        print(f"track  : {track_timer/max(n_track_total,1)*1000:7.2f}"
              f" ms/track   x{n_track_total}")
    print(f"prep   : {stages['prep']/n*1000:7.2f} ms/frame")
    print(f"final  : {stages['final']/n*1000:7.2f} ms/frame")
    if stages["enh"]:
        print(f"enh    : {stages['enh']/n*1000:7.2f} ms/frame")
    print(f"post   : {stages['post']/n*1000:7.2f} ms/frame")
    _print_frame_times(intervals)


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
