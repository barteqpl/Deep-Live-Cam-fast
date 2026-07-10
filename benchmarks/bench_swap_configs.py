"""Experiment: time raw inswapper inference under different CoreML configs/models.

Isolates session.run cost with a realistic blob+latent, 40 runs after 8 warmup.
Prints CoreML partition counts (log_severity_level=2 captures fallback info).
"""
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

MODELS = {
    "fp32": os.path.join(_REPO_ROOT, "models/inswapper_128.onnx"),
    "fp16": os.path.join(_REPO_ROOT, "models/inswapper_128_fp16.onnx"),
    "simplified": os.path.join(_REPO_ROOT, "models/inswapper_128_simplified.onnx"),
}

CONFIGS = {
    "CPUAndGPU": {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU",
                  "SpecializationStrategy": "FastPrediction",
                  "AllowLowPrecisionAccumulationOnGPU": 0, "EnableOnSubgraphs": 1},
    "ALL(ANE)": {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL",
                 "SpecializationStrategy": "FastPrediction",
                 "AllowLowPrecisionAccumulationOnGPU": 1, "EnableOnSubgraphs": 1},
    "GPU-lowprec": {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU",
                    "SpecializationStrategy": "FastPrediction",
                    "AllowLowPrecisionAccumulationOnGPU": 1, "EnableOnSubgraphs": 1},
    "CPUOnly": None,  # pure CPU EP reference
}

WARMUP, RUNS = 8, 40


def bench(model_path, cfg_name, cfg):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if cfg is None:
        providers = ["CPUExecutionProvider"]
    else:
        providers = [("CoreMLExecutionProvider", cfg), "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    except Exception as e:
        return f"{cfg_name:12s} LOAD FAIL: {e}"

    inputs = sess.get_inputs()
    blob = np.random.rand(1, 3, 128, 128).astype(np.float32)
    latent = np.random.rand(1, 512).astype(np.float32)
    latent /= np.linalg.norm(latent)
    feed = {inputs[0].name: blob, inputs[1].name: latent}
    outs = [o.name for o in sess.get_outputs()]

    try:
        for _ in range(WARMUP):
            sess.run(outs, {inputs[0].name: blob, inputs[1].name: latent.copy()})
        t0 = time.perf_counter()
        for _ in range(RUNS):
            sess.run(outs, {inputs[0].name: blob, inputs[1].name: latent.copy()})
        dt = (time.perf_counter() - t0) / RUNS * 1000
        return f"{cfg_name:12s} {dt:8.2f} ms/run"
    except Exception as e:
        return f"{cfg_name:12s} RUN FAIL: {type(e).__name__}: {str(e)[:120]}"


if __name__ == "__main__":
    only_model = sys.argv[1] if len(sys.argv) > 1 else None
    for mname, mpath in MODELS.items():
        if only_model and mname != only_model:
            continue
        if not os.path.exists(mpath):
            print(f"--- {mname}: MISSING {mpath}")
            continue
        print(f"--- {mname} ({mpath}, {os.path.getsize(mpath)//(1024*1024)} MB)")
        for cname, cfg in CONFIGS.items():
            print("   ", bench(mpath, cname, cfg))
