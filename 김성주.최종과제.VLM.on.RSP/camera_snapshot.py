#!/usr/bin/env python3
"""Capture one webcam frame to verify the camera before running the VLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_common import CAPTURE_DIR, capture_with_ffmpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one frame from a V4L2 webcam.")
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", default=str(CAPTURE_DIR / "test.jpg"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    capture_with_ffmpeg(args.camera, args.width, args.height, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
