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
