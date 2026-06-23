#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import cv2

from importlib.util import spec_from_file_location, module_from_spec

ROOT = Path(__file__).resolve().parent


def load_module(filename: str, name: str):
    spec = spec_from_file_location(name, ROOT / filename)
    mod = module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cap_mod = load_module("01_camera_capture.py", "camera_capture")
yolo_mod = load_module("02_yolo_detector.py", "yolo_detector")
vlm_mod = load_module("04_vlm_inference.py", "vlm_inference")
ui_mod = load_module("03_click_select_ui.py", "click_ui")

Detection = yolo_mod.Detection


def crop_with_padding(frame, xyxy, pad_frac: float = 0.15):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    pad = int(max(bw, bh) * pad_frac)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
    return frame[y1:y2, x1:x2].copy()


def resize_max_side(frame, max_side: int):
    if max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    scale = max_side / float(max(h, w))
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def mock_detections(frame) -> list[Detection]:
    h, w = frame.shape[:2]
    return [Detection((w // 4, h // 4, 3 * w // 4, 3 * h // 4), 0, "object", 0.99)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--yolo", type=Path, default=yolo_mod.DEFAULT_MODEL)
    ap.add_argument("--gguf", type=Path, default=vlm_mod.DEFAULT_GGUF)
    ap.add_argument("--mmproj", type=Path, default=vlm_mod.DEFAULT_MMPROJ)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--vlm-max-tokens", type=int, default=32)
    ap.add_argument("--vlm-prompt", default="What object is shown? Answer with one noun.")
    ap.add_argument("--vlm-server-url", default="", help="OpenAI-compatible llama server chat completions URL.")
    ap.add_argument("--crop-max-side", type=int, default=512)
    ap.add_argument(
        "--inprocess-vlm",
        action="store_true",
        help="Use llama-cpp-python in the worker thread instead of spawning 04_vlm_inference.py per request.",
    )
    ap.add_argument("--mock-yolo", action="store_true")
    ap.add_argument("--mock-vlm", action="store_true")
    ap.add_argument("--no-window", action="store_true", help="Run one iteration and save output.")
    ap.add_argument(
        "--auto-vlm-first-box",
        action="store_true",
        help="In no-window/image smoke tests, submit the smallest first detection to VLM and wait for the result.",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "pipeline_smoke.jpg")
    args = ap.parse_args()

    kind, source_obj = cap_mod.open_source(args.source)
    detector = None if args.mock_yolo else yolo_mod.YoloOnnxDetector(args.yolo)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    worker_state = {"vlm": None}
    pending: dict[int, concurrent.futures.Future] = {}
    labels: dict[int, str] = {}
    last_frame = {"frame": None, "detections": []}

    def submit_vlm(idx: int):
        frame = last_frame["frame"]
        dets = last_frame["detections"]
        if frame is None or idx is None or idx >= len(dets):
            return
        if idx in pending and not pending[idx].done():
            print(f"box {idx}: already thinking; ignoring duplicate click")
            return
        crop = resize_max_side(crop_with_padding(frame, dets[idx].xyxy), args.crop_max_side)
        crop_path = ROOT / "models" / f"clicked_box_{idx}.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_path), crop)
        labels[idx] = "thinking..."

        def work():
            if args.mock_vlm:
                time.sleep(1.0)
                return f"mock VLM result for box {idx}"
            if args.vlm_server_url:
                client = vlm_mod.SmolVlmHttpClient(args.vlm_server_url)
                return client.describe(crop_path, args.vlm_prompt, max_tokens=args.vlm_max_tokens)
            if args.inprocess_vlm:
                if worker_state["vlm"] is None:
                    worker_state["vlm"] = vlm_mod.SmolVlmGguf(args.gguf, args.mmproj, threads=args.threads)
                return worker_state["vlm"].describe(
                    crop_path,
                    args.vlm_prompt,
                    max_tokens=args.vlm_max_tokens,
                )
            cmd = [
                sys.executable,
                str(ROOT / "04_vlm_inference.py"),
                "--model",
                str(args.gguf),
                "--mmproj",
                str(args.mmproj),
                "--image",
                str(crop_path),
                "--prompt",
                args.vlm_prompt,
                "--max-tokens",
                str(args.vlm_max_tokens),
                "--threads",
                str(args.threads),
            ]
            proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            content = [
                line for line in lines
                if not line.startswith("vlm_ms=") and not line.startswith("load_plus_infer_ms=")
            ]
            return content[0] if content else proc.stdout.strip()

        pending[idx] = executor.submit(work)
        print(f"submitted VLM for box {idx}: {crop_path}")

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        idx = ui_mod.pick_box(x, y, last_frame["detections"])
        if idx is not None:
            submit_vlm(idx)

    if not args.no_window:
        cv2.namedWindow("rpi_vlm_pipeline")
        cv2.setMouseCallback("rpi_vlm_pipeline", on_mouse)

    try:
        while True:
            if kind == "image":
                frame = source_obj.copy()
            else:
                ok, frame = source_obj.read()
                if not ok:
                    break
            detections = mock_detections(frame) if args.mock_yolo else detector.detect(frame)
            last_frame.update(frame=frame.copy(), detections=detections)
            if args.no_window and args.auto_vlm_first_box and detections and not pending and not labels:
                submit_vlm(0)
                while pending:
                    for idx, fut in list(pending.items()):
                        if fut.done():
                            try:
                                labels[idx] = fut.result()
                            except Exception as exc:
                                labels[idx] = f"VLM error: {exc}"
                            print(f"VLM result for box {idx}: {labels[idx]}")
                            del pending[idx]
                    time.sleep(0.05)
            for idx, fut in list(pending.items()):
                if fut.done():
                    try:
                        labels[idx] = fut.result()
                    except Exception as exc:
                        labels[idx] = f"VLM error: {exc}"
                    print(f"VLM result for box {idx}: {labels[idx]}")
                    del pending[idx]
            rendered = yolo_mod.draw_detections(frame, detections, labels)
            if args.no_window:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.out), rendered)
                print(f"saved {args.out} detections={len(detections)}")
                break
            cv2.imshow("rpi_vlm_pipeline", rendered)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if kind == "image":
                time.sleep(0.03)
    finally:
        if kind == "video":
            source_obj.release()
        executor.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
