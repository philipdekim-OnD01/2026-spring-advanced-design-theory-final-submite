#!/usr/bin/env python3
"""Shared helpers for the webcam VLM scripts.

All scripts in this folder live next to one another, so the project root is
simply the directory that contains this file.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LLAMA_CLI = PROJECT_ROOT / "llama.cpp" / "build" / "bin" / "llama-mtmd-cli"
CAPTURE_DIR = PROJECT_ROOT / "captures"

RPS_LABELS = ("rock", "paper", "scissors", "unknown")

# Single-word answer prompt, used by the live overlay where short output is faster.
RPS_PROMPT_WORD = """Look at the visible hand gesture in this camera image.
Choose the best class from these four classes:
- rock: closed fist
- paper: open flat hand
- scissors: two extended fingers
- unknown: no clear hand gesture
Final answer must be exactly one lowercase word.
"""

# Structured JSON prompt, used by the batch script that prints parsed results.
RPS_PROMPT_JSON = """Look at the hand gesture in this image.
Classify it as exactly one of: rock, paper, scissors, unknown.
Return only compact JSON like:
{"label":"rock","confidence":0.0,"reason":"short visual reason"}
"""

OBJECT_PROMPT = """Identify the main visible objects in this image.
Return only compact JSON like:
{"objects":[{"name":"object","confidence":0.0}],"scene":"short scene description"}
"""


def prepare_display(display: str | None) -> None:
    """Point OpenCV/ffplay at an attached monitor when run over SSH."""
    if display:
        os.environ["DISPLAY"] = display
    elif not os.environ.get("DISPLAY") and os.path.exists("/tmp/.X11-unix/X0"):
        os.environ["DISPLAY"] = ":0"

    if os.environ.get("DISPLAY") and not os.environ.get("XAUTHORITY"):
        xauthority = os.path.expanduser("~/.Xauthority")
        if os.path.exists(xauthority):
            os.environ["XAUTHORITY"] = xauthority


def capture_with_ffmpeg(camera: str, width: int, height: int, image_path: Path) -> None:
    """Grab a single frame from a V4L2 device with ffmpeg."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "v4l2",
        "-video_size",
        f"{width}x{height}",
        "-i",
        camera,
        "-frames:v",
        "1",
        str(image_path),
    ]
    subprocess.run(cmd, check=True)


def build_mtmd_command(
    llama_cli: str,
    image_path: Path,
    prompt: str,
    *,
    temp: float,
    predict: int,
    ctx_size: int,
    threads: int,
    hf_repo: str | None = None,
    hf_file: str | None = None,
    model: str | None = None,
    mmproj: str | None = None,
    chat_template: str | None = None,
) -> list[str]:
    """Build a llama-mtmd-cli command line.

    A local model+projector pair takes priority; otherwise the Hugging Face
    repo is used so llama.cpp can auto-download it.
    """
    cmd = [
        llama_cli,
        "--image",
        str(image_path),
        "-p",
        prompt,
        "--temp",
        str(temp),
        "-n",
        str(predict),
        "-c",
        str(ctx_size),
        "-t",
        str(threads),
        "--verbosity",
        "1",
    ]

    if model and mmproj:
        cmd.extend(["-m", model, "--mmproj", mmproj])
    elif hf_repo:
        cmd.extend(["--hf-repo", hf_repo])
        if hf_file:
            cmd.extend(["--hf-file", hf_file])

    if chat_template:
        cmd.extend(["--chat-template", chat_template])

    return cmd


def extract_json(text: str) -> dict | None:
    """Return the first JSON object embedded in ``text``, or None."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def clean_label(text: str) -> str:
    """Reduce noisy model output to one of RPS_LABELS."""
    parsed = extract_json(text)
    if parsed is not None:
        label = str(parsed.get("label", "")).lower()
        if label in RPS_LABELS:
            return label

    lowered = text.strip().lower()
    labels = re.findall(r"\b(rock|paper|scissors|unknown)\b", lowered)
    unique_labels = list(dict.fromkeys(labels))
    if len(unique_labels) == 1:
        return unique_labels[0]

    final_match = re.search(
        r"(?:final answer|answer|label)\s*[:=]\s*(rock|paper|scissors|unknown)\b", lowered
    )
    if final_match:
        return final_match.group(1)

    return "unknown"
