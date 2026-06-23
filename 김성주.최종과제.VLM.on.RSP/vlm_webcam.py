#!/usr/bin/env python3
"""
Capture webcam frames and classify them with a llama.cpp multimodal model.

Examples:
  python3 scripts/vlm_webcam.py --hf-repo ggml-org/SmolVLM-500M-Instruct-GGUF --mode rps
  python3 scripts/vlm_webcam.py --model /path/model.gguf --mmproj /path/mmproj.gguf --mode objects
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from vlm_common import (
    CAPTURE_DIR,
    DEFAULT_LLAMA_CLI,
    OBJECT_PROMPT,
    RPS_PROMPT_JSON,
    build_mtmd_command,
    capture_with_ffmpeg,
    extract_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a webcam frame with a llama.cpp VLM for RPS or object recognition."
    )
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--llama-cli", default=str(DEFAULT_LLAMA_CLI), help="Path to llama-mtmd-cli.")
    model_group.add_argument("--hf-repo", default=os.environ.get("VLM_HF_REPO"), help="Hugging Face GGUF repo. Uses llama.cpp auto-download.")
    model_group.add_argument("--hf-file", default=os.environ.get("VLM_HF_FILE"), help="Optional model file inside --hf-repo.")
    model_group.add_argument("--model", default=os.environ.get("VLM_MODEL"), help="Local language model GGUF.")
    model_group.add_argument("--mmproj", default=os.environ.get("VLM_MMPROJ"), help="Local multimodal projector GGUF.")
    model_group.add_argument("--chat-template", default=os.environ.get("VLM_CHAT_TEMPLATE"), help="Optional llama.cpp chat template, e.g. vicuna or smolvlm.")

    camera_group = parser.add_argument_group("camera")
    camera_group.add_argument("--camera", default="/dev/video0", help="Camera device for ffmpeg fallback.")
    camera_group.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    camera_group.add_argument("--width", type=int, default=640)
    camera_group.add_argument("--height", type=int, default=480)
    camera_group.add_argument("--interval", type=float, default=2.0, help="Seconds between recognitions.")
    camera_group.add_argument("--once", action="store_true", help="Run once and exit.")
    camera_group.add_argument("--save-dir", default=str(CAPTURE_DIR), help="Where captured frames are stored.")
    camera_group.add_argument("--no-opencv", action="store_true", help="Force ffmpeg snapshot capture.")

    infer_group = parser.add_argument_group("inference")
    infer_group.add_argument("--mode", choices=["rps", "objects"], default="rps")
    infer_group.add_argument("--prompt", help="Override the built-in prompt.")
    infer_group.add_argument("--threads", type=int, default=4)
    infer_group.add_argument("--ctx-size", type=int, default=2048)
    infer_group.add_argument("--predict", type=int, default=96)
    infer_group.add_argument("--temp", type=float, default=0.1)
    infer_group.add_argument("--raw", action="store_true", help="Print raw model output too.")
    return parser.parse_args()


def ensure_model_args(args: argparse.Namespace) -> None:
    if args.hf_repo:
        return
    if args.model and args.mmproj:
        return
    raise SystemExit(
        "Model is not configured. Use --hf-repo, or use both --model and --mmproj.\n"
        "Example: python3 scripts/vlm_webcam.py --hf-repo ggml-org/SmolVLM-500M-Instruct-GGUF --mode rps"
    )


def prompt_for_mode(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    return RPS_PROMPT_JSON if args.mode == "rps" else OBJECT_PROMPT


def opencv_available() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


def capture_with_opencv(args: argparse.Namespace, image_path: Path) -> None:
    import cv2

    camera = cv2.VideoCapture(args.camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open OpenCV camera index {args.camera_index}")

    ok = False
    frame = None
    for _ in range(8):
        ok, frame = camera.read()
        if ok:
            break
        time.sleep(0.05)
    camera.release()

    if not ok or frame is None:
        raise RuntimeError("OpenCV camera read failed")
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Could not write captured frame: {image_path}")


def capture_frame(args: argparse.Namespace, image_path: Path) -> None:
    if not args.no_opencv and opencv_available():
        capture_with_opencv(args, image_path)
        return
    capture_with_ffmpeg(args.camera, args.width, args.height, image_path)


def run_vlm(args: argparse.Namespace, image_path: Path, prompt: str) -> str:
    cmd = build_mtmd_command(
        args.llama_cli,
        image_path,
        prompt,
        temp=args.temp,
        predict=args.predict,
        ctx_size=args.ctx_size,
        threads=args.threads,
        hf_repo=args.hf_repo,
        hf_file=args.hf_file,
        model=args.model,
        mmproj=args.mmproj,
        chat_template=args.chat_template,
    )
    result = subprocess.run(cmd, text=True, capture_output=True)
    combined = "\n".join(part for part in [result.stdout, result.stderr] if part)
    if result.returncode != 0:
        raise RuntimeError(f"llama.cpp failed with exit code {result.returncode}\n{combined}")
    return combined.strip()


def print_result(raw_text: str, raw: bool) -> None:
    parsed = extract_json(raw_text)
    if parsed is not None:
        print(json.dumps(parsed, ensure_ascii=False), flush=True)
    else:
        print(raw_text.strip(), flush=True)
    if raw:
        print("\n--- raw output ---", flush=True)
        print(raw_text, flush=True)


def main() -> int:
    args = parse_args()
    ensure_model_args(args)

    llama_cli = Path(args.llama_cli)
    if not llama_cli.exists():
        raise SystemExit(f"llama.cpp CLI not found: {llama_cli}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompt_for_mode(args)

    print(f"mode={args.mode} camera={args.camera} save_dir={save_dir}", file=sys.stderr)
    print("Press Ctrl+C to stop.", file=sys.stderr)

    try:
        while True:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_path = save_dir / f"frame-{timestamp}.jpg"
            capture_frame(args, image_path)
            raw_text = run_vlm(args, image_path, prompt)
            print_result(raw_text, args.raw)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
