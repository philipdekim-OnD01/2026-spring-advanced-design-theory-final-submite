#!/usr/bin/env python3
"""Create a wide GPU-vs-DLA summary CSV from row-wise benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


MODEL_RE = re.compile(r"rps_scaled_w(?P<int>\d+)_(?P<frac>\d+)_int8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-csv", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args()


def width_from_model(model_name: str) -> float:
    match = MODEL_RE.fullmatch(model_name)
    if not match:
        raise ValueError(f"Unexpected model name: {model_name}")
    return float(f"{match.group('int')}.{match.group('frac')}")


def load_param_counts(artifact_root: Path) -> dict[float, int]:
    params: dict[float, int] = {}
    for info_path in artifact_root.glob("sweep_w*/model_info.json"):
        info = json.loads(info_path.read_text())
        params[float(info["width_mult"])] = int(info["param_count"])
    return params


def main() -> None:
    args = parse_args()
    param_counts = load_param_counts(args.artifact_root)
    latest: dict[tuple[float, str], dict[str, str]] = {}

    with args.benchmark_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["backend"] != "tensorrt" or row["precision"] != "int8":
                continue
            width = width_from_model(row["model"])
            latest[(width, row["device"])] = row

    fieldnames = [
        "width_mult",
        "param_count",
        "gpu_accuracy",
        "dla_accuracy",
        "gpu_latency_ms_mean",
        "dla_latency_ms_mean",
        "dla_minus_gpu_ms",
        "dla_over_gpu_latency_ratio",
        "faster_device",
        "gpu_p95_ms",
        "dla_p95_ms",
        "num_samples",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for width in sorted({key[0] for key in latest}):
            gpu = latest.get((width, "gpu"))
            dla = latest.get((width, "dla0"))
            if gpu is None or dla is None:
                continue
            gpu_latency = float(gpu["latency_ms_mean"])
            dla_latency = float(dla["latency_ms_mean"])
            writer.writerow({
                "width_mult": width,
                "param_count": param_counts.get(width, ""),
                "gpu_accuracy": gpu["accuracy"],
                "dla_accuracy": dla["accuracy"],
                "gpu_latency_ms_mean": gpu_latency,
                "dla_latency_ms_mean": dla_latency,
                "dla_minus_gpu_ms": dla_latency - gpu_latency,
                "dla_over_gpu_latency_ratio": dla_latency / gpu_latency if gpu_latency else "",
                "faster_device": "dla0" if dla_latency < gpu_latency else "gpu",
                "gpu_p95_ms": gpu["latency_ms_p95"],
                "dla_p95_ms": dla["latency_ms_p95"],
                "num_samples": gpu["num_samples"],
            })
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
