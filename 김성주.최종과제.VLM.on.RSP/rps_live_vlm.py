#!/usr/bin/env python3
"""
Live webcam preview with periodic VLM rock-paper-scissors recognition.

Run from SSH and show the window on the Raspberry Pi monitor:
  python3 scripts/rps_live_vlm.py --server-url http://127.0.0.1:8082 --display :0
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2

from vlm_common import (
    CAPTURE_DIR,
    DEFAULT_LLAMA_CLI,
    RPS_PROMPT_WORD as RPS_PROMPT,
    build_mtmd_command,
    clean_label,
    prepare_display,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live webcam RPS recognition with VLM output overlay.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--infer-width", type=int, default=160, help="Resize cropped frame to this width before VLM inference.")
    parser.add_argument("--jpeg-quality", type=int, default=65)
    parser.add_argument("--no-crop", action="store_true", help="Send the whole frame to the VLM.")
    parser.add_argument("--crop-scale", type=float, default=0.55, help="Crop box size as a fraction of the shorter frame side.")
    parser.add_argument("--crop-offset-x", type=float, default=0.0, help="Move crop center horizontally, -1.0 to 1.0.")
    parser.add_argument("--crop-offset-y", type=float, default=0.0, help="Move crop center vertically, -1.0 to 1.0.")
    parser.add_argument("--display", default=None, help="Attached monitor display, usually :0.")
    parser.add_argument("--llama-cli", default=str(DEFAULT_LLAMA_CLI))
    parser.add_argument("--server-url", default=None, help="Use an already-running llama-server, e.g. http://127.0.0.1:8080.")
    parser.add_argument("--hf-repo", default="ggml-org/SmolVLM-500M-Instruct-GGUF:Q8_0")
    parser.add_argument("--hf-file", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--mmproj", default=None)
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds between VLM recognitions.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=256)
    parser.add_argument("--predict", type=int, default=24)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--window-title", default="RPS VLM Live")
    return parser.parse_args()


def run_server_inference(args: argparse.Namespace, image_path: Path) -> tuple[str, str]:
    with image_path.open("rb") as file:
        image_b64 = base64.b64encode(file.read()).decode("ascii")

    payload = {
        "model": args.hf_repo or "local",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": RPS_PROMPT},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
                ],
            }
        ],
        "temperature": args.temp,
        "max_tokens": args.predict,
    }
    data = json.dumps(payload).encode("utf-8")
    url = args.server_url.rstrip("/") + "/v1/chat/completions"
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return "error", str(exc)

    raw = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    return clean_label(raw), raw


def infer_label(args: argparse.Namespace, image_path: Path) -> tuple[str, str]:
    if args.server_url:
        return run_server_inference(args, image_path)

    cmd = build_mtmd_command(
        args.llama_cli,
        image_path,
        RPS_PROMPT,
        temp=args.temp,
        predict=args.predict,
        ctx_size=args.ctx_size,
        threads=args.threads,
        hf_repo=args.hf_repo,
        hf_file=args.hf_file,
        model=args.model,
        mmproj=args.mmproj,
    )
    result = subprocess.run(cmd, text=True, capture_output=True)
    raw = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
    if result.returncode != 0:
        return "error", raw[-300:]
    return clean_label(raw), raw


def inference_worker(args: argparse.Namespace, jobs: queue.Queue[Path], results: queue.Queue[tuple[str, str, float]]) -> None:
    while True:
        image_path = jobs.get()
        if image_path is None:
            return
        started = time.time()
        label, raw = infer_label(args, image_path)
        results.put((label, raw, time.time() - started))


def crop_box(frame, scale: float, offset_x: float, offset_y: float) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    size = int(min(width, height) * max(0.05, min(scale, 1.0)))
    max_shift_x = (width - size) // 2
    max_shift_y = (height - size) // 2
    center_x = width // 2 + int(max_shift_x * max(-1.0, min(offset_x, 1.0)))
    center_y = height // 2 + int(max_shift_y * max(-1.0, min(offset_y, 1.0)))
    x1 = max(0, min(width - size, center_x - size // 2))
    y1 = max(0, min(height - size, center_y - size // 2))
    return x1, y1, x1 + size, y1 + size


def draw_overlay(frame, label: str, status: str, latency: float | None, roi: tuple[int, int, int, int] | None) -> None:
    height, width = frame.shape[:2]
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(frame, "VLM crop", (x1 + 8, max(y1 + 24, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

    cv2.rectangle(frame, (0, 0), (width, 92), (0, 0, 0), -1)
    cv2.putText(frame, f"VLM: {label.upper()}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 255, 0), 3)
    detail = status if latency is None else f"{status} | last inference: {latency:.1f}s"
    cv2.putText(frame, detail, (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
    cv2.putText(frame, "Press q to quit", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def write_inference_frame(
    frame,
    image_path: Path,
    infer_width: int,
    jpeg_quality: int,
    roi: tuple[int, int, int, int] | None,
) -> None:
    output = frame if roi is None else frame[roi[1] : roi[3], roi[0] : roi[2]]
    if infer_width > 0 and output.shape[1] > infer_width:
        scale = infer_width / output.shape[1]
        output = cv2.resize(output, (infer_width, int(output.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(image_path), output, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])


def main() -> int:
    args = parse_args()
    prepare_display(args.display)
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    if not args.server_url and not Path(args.llama_cli).exists():
        print(f"llama CLI not found: {args.llama_cli}")
        return 1

    camera = cv2.VideoCapture(args.camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        print(f"Could not open camera index {args.camera_index}")
        return 1

    jobs: queue.Queue[Path] = queue.Queue(maxsize=1)
    results: queue.Queue[tuple[str, str, float]] = queue.Queue()
    worker = threading.Thread(target=inference_worker, args=(args, jobs, results), daemon=True)
    worker.start()

    cv2.namedWindow(args.window_title, cv2.WINDOW_NORMAL)
    last_request = 0.0
    label = "waiting"
    status = "warming up"
    latency = None

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                status = "camera read failed"
                time.sleep(0.1)
                continue

            now = time.time()
            roi = None if args.no_crop else crop_box(frame, args.crop_scale, args.crop_offset_x, args.crop_offset_y)
            if now - last_request >= args.interval and jobs.empty():
                image_path = CAPTURE_DIR / "live-vlm-frame.jpg"
                write_inference_frame(frame, image_path, args.infer_width, args.jpeg_quality, roi)
                jobs.put(image_path)
                last_request = now
                status = "VLM thinking..."

            while not results.empty():
                label, raw, latency = results.get()
                status = raw[:80].replace("\n", " ") if raw else "no text output"
                print(f"{time.strftime('%H:%M:%S')} label={label} latency={latency:.1f}s raw={raw!r}", flush=True)

            draw_overlay(frame, label, status, latency, roi)
            cv2.imshow(args.window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
