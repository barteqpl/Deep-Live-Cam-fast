"""Filter benign CoreML/ANE compiler diagnostics out of stderr.

On Apple Silicon the CoreML execution provider prints low-level compiler
messages straight to the process's stderr (file descriptor 2) whenever a model
partition can't be placed on the Neural Engine and falls back to GPU/CPU:

    E5RT encountered an STL exception. msg = ... unbounded dimension ...
    MILCompilerForANE error: failed to compile ANE model using ANEF. ...

These are non-fatal — inference still runs correctly on the fallback device —
but they flood the console during model load, first inference, and interpreter
teardown, and read like a crash. This installs a one-time fd-level filter that
drops only those known-benign lines and forwards everything else unchanged, so
real errors still reach the terminal.
"""
import os
import sys
import threading

_INSTALLED = False

_BENIGN_MARKERS = (
    b"E5RT encountered an STL exception",
    b"MILCompilerForANE error",
    b"ANECCompile() FAILED",
    b"has unbounded dimension which is not supported",
    b"Failed to PropagateInputTensorShapes",
    b"MILCompilerForANE",
)


def _looks_benign(line: bytes) -> bool:
    return any(marker in line for marker in _BENIGN_MARKERS)


def install() -> None:
    """Redirect the real stderr through a filtering pump. Idempotent, and a
    no-op off macOS (nothing emits these messages there)."""
    global _INSTALLED
    if _INSTALLED or sys.platform != "darwin":
        return
    _INSTALLED = True

    # Duplicate the current stderr so we can keep writing the kept lines to it.
    real_stderr_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    # Point fd 2 (what CoreML writes to) at the pipe's write end.
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def pump() -> None:
        buf = b""
        with os.fdopen(read_fd, "rb", buffering=0) as reader:
            while True:
                chunk = reader.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not _looks_benign(line):
                        os.write(real_stderr_fd, line + b"\n")
            if buf and not _looks_benign(buf):
                os.write(real_stderr_fd, buf)

    threading.Thread(target=pump, name="coreml-stderr-filter", daemon=True).start()
