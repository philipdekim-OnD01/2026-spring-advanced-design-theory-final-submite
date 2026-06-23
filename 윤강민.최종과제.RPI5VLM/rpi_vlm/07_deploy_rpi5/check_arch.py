#!/usr/bin/env python3
from __future__ import annotations

import platform
import subprocess


def cpuinfo() -> str:
    try:
        return open("/proc/cpuinfo", "r", encoding="utf-8").read()
    except Exception:
        return ""


def check_import(name: str):
    try:
        mod = __import__(name)
        return True, getattr(mod, "__version__", "?")
    except Exception as exc:
        return False, repr(exc)


def main() -> None:
    print("platform:", platform.platform())
    print("machine:", platform.machine())
    info = cpuinfo().lower()
    for feat in ["neon", "asimd", "dotprod", "fp16"]:
        print(f"cpu feature {feat}:", feat in info)
    for mod in ["cv2", "numpy", "onnxruntime", "llama_cpp", "psutil"]:
        ok, detail = check_import(mod)
        print(f"import {mod}:", "OK" if ok else "FAIL", detail)
    try:
        print(subprocess.check_output(["uname", "-a"], text=True).strip())
    except Exception:
        pass


if __name__ == "__main__":
    main()

