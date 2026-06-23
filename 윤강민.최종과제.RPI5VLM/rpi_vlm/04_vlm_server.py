#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location


ROOT = Path(__file__).resolve().parent
_spec = spec_from_file_location("vlm_inference", ROOT / "04_vlm_inference.py")
_mod = module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["vlm_inference"] = _mod
_spec.loader.exec_module(_mod)


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch a resident llama-cpp-python SmolVLM server.")
    ap.add_argument("--model", type=Path, default=_mod.DEFAULT_GGUF)
    ap.add_argument("--mmproj", type=Path, default=_mod.DEFAULT_MMPROJ)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--threads-batch", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "llama_cpp.server",
        "--model",
        str(args.model),
        "--clip_model_path",
        str(args.mmproj),
        "--chat_format",
        "llava-1-6",
        "--n_ctx",
        str(args.ctx),
        "--n_threads",
        str(args.threads),
        "--n_threads_batch",
        str(args.threads_batch),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--verbose",
        "False",
    ]
    print("starting:", " ".join(cmd), flush=True)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
