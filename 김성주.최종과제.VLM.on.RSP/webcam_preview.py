#!/usr/bin/env python3
"""Show the webcam feed on the Raspberry Pi monitor."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from vlm_common import prepare_display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a live webcam preview window.")
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--window-title", default="Webcam Preview")
    parser.add_argument(
        "--display",
        default=None,
        help="Display target for an attached monitor, e.g. :0. Auto-detected when possible.",
    )
    return parser.parse_args()


def warn_if_headless() -> None:
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    print(
        "warning: no DISPLAY or WAYLAND_DISPLAY is set. "
        "Run this from the Raspberry Pi desktop terminal if no preview window appears.",
        file=sys.stderr,
    )


def main() -> int:
    args = parse_args()
    ffplay = shutil.which("ffplay")
    if ffplay is None:
        print("error: ffplay is not installed. Install it with: sudo apt install ffmpeg", file=sys.stderr)
        return 1

    prepare_display(args.display)
    warn_if_headless()

    cmd = [
        ffplay,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-framedrop",
        "-f",
        "v4l2",
        "-video_size",
        f"{args.width}x{args.height}",
        "-framerate",
        str(args.fps),
        "-i",
        args.camera,
        "-window_title",
        args.window_title,
    ]
    if args.fullscreen:
        cmd.append("-fs")

    print("Press q in the preview window to quit.", file=sys.stderr)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
