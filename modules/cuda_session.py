"""Keep ONNX models on the GPU when cuDNN cannot run a convolution.

With some cuDNN builds (seen with cuDNN 9.27 on a Tesla T4, loaded into the
process by ``import torch``), the default exhaustive conv-algorithm search
finds no engine for some Conv layers ("No valid engine configs").
onnxruntime then silently moves the whole session to the CPU at its first
run (face recognizer ~200 ms, SimSwap ~610 ms per run). Here every CUDA
session runs one dummy inference at load time with that fallback disabled;
when it fails, the session is rebuilt with a cheaper algorithm search, which
keeps it on the GPU (recognizer 17 ms, SimSwap 64 ms with DEFAULT).

Sessions without the CUDA provider (CoreML, DirectML, CPU) are untouched.
"""
import os
from typing import Any, List

import numpy as np
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import EPFail

_ALGOS = ("HEURISTIC", "DEFAULT")
CUDA = "CUDAExecutionProvider"


def _wants_cuda(providers: List[Any]) -> bool:
    return any((p[0] if isinstance(p, tuple) else p) == CUDA for p in providers)


def _with_algo(providers: List[Any], algo: str) -> List[Any]:
    out = []
    for p in providers:
        name, opts = p if isinstance(p, tuple) else (p, {})
        out.append((name, dict(opts, cudnn_conv_algo_search=algo)) if name == CUDA else p)
    return out


def _runs_on_cuda(session: Any) -> bool:
    """One zero-filled inference with onnxruntime's CPU fallback disabled."""
    if session.get_providers()[0] != CUDA:
        return False
    feed = {}
    for i in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else (1 if k == 0 else 640)
                 for k, d in enumerate(i.shape)]
        feed[i.name] = np.zeros(shape, np.float16 if "float16" in i.type else np.float32)
    session.disable_fallback()
    try:
        session.run(None, feed)
    except EPFail:
        return False
    except Exception:
        return True  # not a CUDA failure (e.g. unusual input shapes): leave it alone
    finally:
        session.enable_fallback()
    return True


def _rebuild(path: str, providers: List[Any], **kwargs: Any) -> Any:
    for algo in _ALGOS:
        session = ort.InferenceSession(path, providers=_with_algo(providers, algo), **kwargs)
        if _runs_on_cuda(session):
            print(f"[DLC.CUDA] {os.path.basename(path)}: cuDNN found no engine with the "
                  f"exhaustive conv search, using {algo}.")
            return session
    return None


def cuda_safe_session(path: str, providers: List[Any], **kwargs: Any) -> Any:
    """ort.InferenceSession that stays on CUDA if any conv-algorithm search works."""
    session = ort.InferenceSession(path, providers=providers, **kwargs)
    if not _wants_cuda(providers) or _runs_on_cuda(session):
        return session
    return _rebuild(path, providers, **kwargs) or session


def ensure_cuda(model: Any, providers: List[Any]) -> None:
    """Same check for an insightface model object (``.session``, ``.model_file``)."""
    path = getattr(model, "model_file", None)
    session = getattr(model, "session", None)
    if path is None or session is None or not _wants_cuda(providers) or _runs_on_cuda(session):
        return
    rebuilt = _rebuild(path, providers)
    if rebuilt is not None:
        model.session = rebuilt
