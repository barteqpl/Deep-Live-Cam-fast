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


def _with_algo(providers: List[Any], algo: str) -> List[Any]:
    out = []
    for p in providers:
        name, opts = p if isinstance(p, tuple) else (p, {})
        out.append((name, dict(opts, cudnn_conv_algo_search=algo)) if name == CUDA else p)
    return out


def _conv_ok(session: Any) -> bool:
    """False only when a zero-filled run fails inside an execution provider."""
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
        pass  # not an EP failure (unusual inputs): leave the session alone
    finally:
        session.enable_fallback()
    return True


def _fix(session: Any, path: str, providers: List[Any], **kwargs: Any) -> Any:
    if CUDA not in session.get_providers() or _conv_ok(session):
        return session
    for algo in _ALGOS:
        try:
            retry = ort.InferenceSession(path, providers=_with_algo(providers, algo), **kwargs)
        except Exception:
            continue
        if _conv_ok(retry):
            print(f"[DLC.CUDA] {os.path.basename(path)}: cuDNN found no engine with the "
                  f"exhaustive conv search, using {algo}.")
            return retry
        del retry
    return session  # onnxruntime's own CPU fallback, as before


def cuda_safe_session(path: str, providers: List[Any], **kwargs: Any) -> Any:
    """ort.InferenceSession that stays on CUDA if any conv-algorithm search works."""
    return _fix(ort.InferenceSession(path, providers=providers, **kwargs), path, providers, **kwargs)


def ensure_cuda(model: Any, providers: List[Any]) -> None:
    """Same for an insightface model object (``.session``, ``.model_file``)."""
    if getattr(model, "model_file", None) and getattr(model, "session", None) is not None:
        model.session = _fix(model.session, model.model_file, providers)
