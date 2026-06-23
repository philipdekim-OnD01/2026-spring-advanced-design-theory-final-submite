#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
YOLO_ONNX = MODELS / "yolov8n.onnx"
SMOLVLM_DIR = MODELS / "SmolVLM-256M-Instruct"
SMOLVLM_GGUF = MODELS / "smolvlm-256m-instruct-Q4_K_M.gguf"
SMOLVLM_GGUF_REPO = "ggml-org/SmolVLM-256M-Instruct-GGUF"
SMOLVLM_Q8 = MODELS / "SmolVLM-256M-Instruct-Q8_0.gguf"
SMOLVLM_Q8_MMPROJ = MODELS / "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"


def require_import(name: str):
    try:
        return __import__(name)
    except Exception as exc:
        raise SystemExit(f"Missing dependency {name!r}: {exc}. Install 00_setup/requirements.txt first.")


def export_yolo() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    if YOLO_ONNX.exists():
        print(f"YOLO ONNX already exists: {YOLO_ONNX}")
        return
    ultralytics = require_import("ultralytics")
    model = ultralytics.YOLO("yolov8n.pt")
    exported = Path(model.export(format="onnx", imgsz=640, opset=12, simplify=True))
    shutil.move(str(exported), YOLO_ONNX)
    print(f"Wrote {YOLO_ONNX}")


def download_smolvlm() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    hub = require_import("huggingface_hub")
    if SMOLVLM_DIR.exists() and any(SMOLVLM_DIR.iterdir()):
        print(f"SmolVLM HF snapshot already exists: {SMOLVLM_DIR}")
        return
    hub.snapshot_download(
        repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        local_dir=str(SMOLVLM_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"Wrote {SMOLVLM_DIR}")


def download_smolvlm_gguf() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    hub = require_import("huggingface_hub")
    files = [
        ("SmolVLM-256M-Instruct-Q8_0.gguf", SMOLVLM_Q8),
        ("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf", SMOLVLM_Q8_MMPROJ),
    ]
    for filename, out_path in files:
        if out_path.exists():
            print(f"SmolVLM GGUF already exists: {out_path}")
            continue
        downloaded = hub.hf_hub_download(repo_id=SMOLVLM_GGUF_REPO, filename=filename)
        shutil.copy2(downloaded, out_path)
        print(f"Wrote {out_path}")


def convert_smolvlm_to_gguf(llama_cpp: Path | None, quant: str) -> None:
    if SMOLVLM_GGUF.exists():
        print(f"GGUF already exists: {SMOLVLM_GGUF}")
        return
    if llama_cpp is None:
        raise SystemExit(
            "GGUF conversion needs --llama-cpp /path/to/llama.cpp. "
            "SmolVLM multimodal GGUF support depends on the installed llama.cpp version."
        )
    convert = llama_cpp / "convert_hf_to_gguf.py"
    quantize = llama_cpp / "build/bin/llama-quantize"
    if not convert.exists():
        raise SystemExit(f"Missing converter: {convert}")
    if not quantize.exists():
        raise SystemExit(f"Missing quantizer: {quantize}. Build llama.cpp first.")
    f16 = MODELS / "smolvlm-256m-instruct-f16.gguf"
    subprocess.run([sys.executable, str(convert), str(SMOLVLM_DIR), "--outtype", "f16", "--outfile", str(f16)], check=True)
    subprocess.run([str(quantize), str(f16), str(SMOLVLM_GGUF), quant], check=True)
    print(f"Wrote {SMOLVLM_GGUF}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-yolo", action="store_true")
    ap.add_argument("--download-smolvlm", action="store_true")
    ap.add_argument("--download-smolvlm-gguf", action="store_true")
    ap.add_argument("--convert-smolvlm", action="store_true")
    ap.add_argument("--llama-cpp", type=Path)
    ap.add_argument("--quant", default="Q4_K_M")
    args = ap.parse_args()
    if not any([args.export_yolo, args.download_smolvlm, args.download_smolvlm_gguf, args.convert_smolvlm]):
        args.export_yolo = True
        args.download_smolvlm_gguf = True
    if args.export_yolo:
        export_yolo()
    if args.download_smolvlm:
        download_smolvlm()
    if args.download_smolvlm_gguf:
        download_smolvlm_gguf()
    if args.convert_smolvlm:
        convert_smolvlm_to_gguf(args.llama_cpp, args.quant)


if __name__ == "__main__":
    main()
