# Benchmarks

Headless performance benchmarks for the live face-swap pipeline. No webcam or
GUI required — they replay a video file through the exact same per-frame code
path the live preview uses.

## bench_live.py — full live hot-path

Replays frames through `frame.copy -> detect/track -> swap_face ->
apply_post_processing` (the `ui.py:_processing_thread_func` pipeline) and
reports FPS plus a per-stage breakdown, including raw `session.run` inference
time of the swapper.

```bash
python benchmarks/bench_live.py \
    --source path/to/source_face.jpg \
    --target path/to/clip.mov \
    --frames 120 \
    --save-frame /tmp/frame.png   # optional: dump frame #30 for PSNR checks
```

Flags: `--width/--height` (default 960x540, the live working resolution),
`--detect-every` (default 3), `--sharpness`.

## bench_swap_configs.py — raw swapper inference matrix

Times `session.run` of the inswapper model across CoreML EP configurations
(compute units, precision, model variants) with a synthetic blob. Use it to
re-evaluate provider settings after an onnxruntime upgrade.

```bash
python benchmarks/bench_swap_configs.py          # all model variants
python benchmarks/bench_swap_configs.py fp16     # single variant
```

Watch the `GetCapability ... number of partitions` warning in the output:
more than 1 partition means CPU fallbacks inside the graph — usually the
dominant cost on Apple Silicon.

## bench_256_models.py — 256px swapper inference matrix

Times `session.run` of simswap / hififace / hyperswap across CoreML configs
(GPU vs Neural Engine). Add `--optimized` to route through
`optimize_for_coreml` first, or a model name to test one variant.

## Reference results (Apple M4 Pro, 960x540)

Per-model raw inference and full-pipeline FPS (`bench_live.py --model ...`):

| Model | Best compute units | Inference | Pipeline FPS |
|---|---|---|---|
| simswap_256 (default) | GPU (`CPUAndGPU`) | 21.5 ms | 35.2 |
| inswapper_128 fp16 | GPU (`CPUAndGPU`) | 37.5 ms | 22.7 |
| hififace_256 | GPU (`CPUAndGPU`) | 46.5 ms | — |
| hyperswap_256 | **ANE (`CPUAndNeuralEngine`)** | 46.6 ms (was 137.5 on GPU) | 17.2 |

HyperSwap is the notable case: it is ~3x faster on the Neural Engine, while
simswap (81 ms) and hififace (116 ms) get *slower* on ANE — compute-unit
routing is per-model, not global (see `get_face_swapper`).

## HyperSwap 1a vs 1b + enhancer results (M4 Pro, 960x540, gif-derived clip)

Measured with `bench_live.py --frames 60` on a small-face test clip (numbers
are comparable within this table, not with the table above):

| Config | Pipeline FPS | Swap infer | Enhancer |
|---|---|---|---|
| hyperswap (1a) | 13.0 | 63.7 ms | — |
| hyperswap + mouth mask | 12.8 | 61.8 ms | — |
| hyperswap-1b | 14.4 | 55.8 ms | — |
| simswap + GFPGAN-512 | 6.6 | 23.0 ms | 119.2 ms |
| simswap + GPEN-256 | 12.4 | 22.3 ms | 48.9 ms |
| hyperswap-1b + GPEN-256 | 8.2 | 52.4 ms | 56.8 ms |

Findings:

- **hyperswap_1b_256** (same architecture, newer FaceFusion checkpoint) runs
  ~10% faster than 1a on ANE and is exposed as the `hyperswap-1b` model choice.
- **The face enhancer was the live bottleneck**: GFPGAN-512 costs ~108-119
  ms/frame on GPU (and 716 ms on ANE — never route it there). The live path
  now defaults to GPEN-BFR-256 (~50 ms, `--live-enhancer` to override); file
  processing keeps GFPGAN-512 for quality.
- Dead ends (measured, don't retry): `optimize_for_coreml` on hyperswap is a
  no-op; hyperswap weights are already fp16 internally (onnxconverter refuses
  a second conversion); hyperswap on GPU is ~140 ms (2x slower than ANE).
- Mouth masking costs ~1.4% FPS (landmark model +3.7 ms on detection frames
  only, every `DETECT_EVERY_N`), and only when the switch is on.

## Pipelined pool results (`--pool`, branch feat/dual-session-pipeline)

Same clip and harness as the table above (`bench_live.py --frames 120`,
interleaved serial/pool to hold thermal state constant). Faza 3.2 (GIL relief):
pool workers now run ONLY `session.run` (GIL released); all per-frame CPU work
(prepare + finalize + mouth mask + post) moved to the main thread, so ANE-only
inference overlaps detection/finalization with **uniform** pacing.

| Config | Pipeline FPS | frame-time p95/p50 |
|---|---|---|
| hyperswap-1b serial (baseline) | ~14.8 | 1.30x |
| hyperswap-1b `--pool` (ANE only) | **~16.4 (+10%)** | **1.08x** |
| + `--mouth-mask` | 16.5 (cost ~0%) | 1.06x |
| simswap serial (regression check) | 34.6 (unchanged) | 1.29x |

Pacing, not just throughput, is the acceptance criterion: the ANE-only pool
emits uniform ~52-56 ms frames (p95/p50 1.08x) vs the serial path's 1.30x, and
beats serial in every interleaved run. **`--pool` is ON by default** for the
hyperswap single-face live fast path (excludes map_faces / many_faces /
poisson_blend). Quality vs serial: PSNR 53 dB on the same frame index.

`--dual-session` still adds a second ONNX session on the GPU (opt-in). It raises
average FPS but has bursty frame pacing (a ~150 ms GPU frame stalls emission of
2-3 ready ANE frames), so it stays opt-in — see TODO.md 2.3.

Pre-3.2 numbers, for reference: pool-through-full-`swap_face()` ANE-only was
14.75 (+4%, GIL-bound); dual 18.3-18.9 (+29% avg but jumpy).

## Historical results (inswapper_128 optimization arc)

| Configuration | Inference | Pipeline FPS |
|---|---|---|
| fp32, onnxruntime 1.19.2 (16 CoreML partitions) | 68.3 ms | 12.9 |
| fp32, onnxruntime 1.27.0 (1 partition) | 45.6 ms | — |
| fp16, ORT 1.27, MLProgram CPUAndGPU + lowprec | 37.5 ms | 20.5 |
| + ROI paste-back + copy elimination | — | 22.3 |
| + detection-only fast path | — | 22.7 |

Quality gates used when changing precision/blending: `cv2.PSNR` between
output frames — fp16 vs fp32 measured 60.1 dB (max pixel diff 1), ROI
paste-back vs full-frame 81 dB. Anything ≥ 45 dB on the face region is
visually indistinguishable.
