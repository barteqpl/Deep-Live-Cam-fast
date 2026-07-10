"""Experiment: raw inference time of the 256px swappers (simswap / hififace /
hyperswap) across CoreML configs, incl. optimize_for_coreml variants.

Prints partition counts from the GetCapability warning (run with stderr visible).
"""
import os
import sys
import time

import numpy as np
import onnxruntime as ort

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

MODELS = {
    "simswap": ("simswap_256.onnx", {"in": ["input", "latent"]}),
    "hififace": ("hififace_unofficial_256.onnx", {"in": ["input", "latent"]}),
    "hyperswap": ("hyperswap_1a_256.onnx", {"in": ["target", "source"]}),
    "hyperswap-1b": ("hyperswap_1b_256.onnx", {"in": ["target", "source"]}),
}

CFG = {
    "MLProgram-GPU-lowprec": {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU",
                              "AllowLowPrecisionAccumulationOnGPU": 1,
                              "SpecializationStrategy": "FastPrediction", "EnableOnSubgraphs": 1},
    "MLProgram-ALL": {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL",
                      "AllowLowPrecisionAccumulationOnGPU": 1,
                      "SpecializationStrategy": "FastPrediction", "EnableOnSubgraphs": 1},
    "CPUOnly": None,
}

WARMUP, RUNS = 8, 30


def bench(path, cfg):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"] if cfg is None else [("CoreMLExecutionProvider", cfg), "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(path, sess_options=so, providers=providers)
    except Exception as e:
        return f"LOAD FAIL: {str(e)[:100]}"
    ins = sess.get_inputs()
    feed = {}
    for i in ins:
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        feed[i.name] = np.random.rand(*shape).astype(np.float32)
    outs = [o.name for o in sess.get_outputs()]
    try:
        for _ in range(WARMUP):
            sess.run(outs, {k: v.copy() for k, v in feed.items()})
        t0 = time.perf_counter()
        for _ in range(RUNS):
            sess.run(outs, {k: v.copy() for k, v in feed.items()})
        return f"{(time.perf_counter() - t0) / RUNS * 1000:8.2f} ms/run"
    except Exception as e:
        return f"RUN FAIL: {str(e)[:100]}"


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    use_opt = "--optimized" in sys.argv
    for name, (fname, _) in MODELS.items():
        if only and only != name and not only.startswith("--"):
            continue
        path = os.path.join(_REPO_ROOT, "models", fname)
        if not os.path.exists(path):
            print(f"--- {name}: MISSING")
            continue
        if use_opt:
            from modules.onnx_optimize import optimize_for_coreml
            path = optimize_for_coreml(path, input_shape=None)
        print(f"--- {name} ({os.path.basename(path)}, {os.path.getsize(path)//(1024*1024)} MB)")
        for cname, cfg in CFG.items():
            print(f"    {cname:22s}", bench(path, cfg))
