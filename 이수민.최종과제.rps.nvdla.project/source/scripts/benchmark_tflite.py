#!/usr/bin/env python3
"""Benchmark a TFLite model with the fixed RPS test manifest."""

from __future__ import annotations

import argparse
import csv
import socket
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite import Interpreter
    except ImportError as exc:
        raise SystemExit(
            "No TFLite runtime found. Install tflite-runtime or tensorflow in this environment."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--img-size", default=64, type=int)
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--model-name", default="rps_dla_cnn_int8")
    parser.add_argument("--precision", default="int8")
    parser.add_argument("--device", default="jetson_cpu")
    return parser.parse_args()


def load_manifest(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                rows.append((row["path"], int(row["label"])))
    if not rows:
        raise ValueError(f"No test rows in manifest: {path}")
    return rows


def load_image(path: Path, img_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        return (np.asarray(image, dtype=np.float32) / 255.0)[None, ...]


def quantize(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    dtype = detail["dtype"]
    if scale == 0:
        return values.astype(dtype)
    q = np.round(values / scale + zero_point)
    info = np.iinfo(dtype)
    return np.clip(q, info.min, info.max).astype(dtype)


def dequantize(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    if scale == 0:
        return values.astype(np.float32)
    return (values.astype(np.float32) - zero_point) * scale


def append_row(args: argparse.Namespace, accuracy: float, latencies: np.ndarray, n: int) -> None:
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "host", "model", "backend", "precision", "device",
        "batch_size", "num_samples", "accuracy", "latency_ms_mean",
        "latency_ms_p50", "latency_ms_p95", "latency_ms_min",
        "latency_ms_max", "throughput_fps", "artifact", "notes",
    ]
    exists = args.output_csv.exists()
    with args.output_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        mean = float(np.mean(latencies))
        writer.writerow({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": socket.gethostname(),
            "model": args.model_name,
            "backend": "tflite",
            "precision": args.precision,
            "device": args.device,
            "batch_size": 1,
            "num_samples": n,
            "accuracy": accuracy,
            "latency_ms_mean": mean,
            "latency_ms_p50": float(np.percentile(latencies, 50)),
            "latency_ms_p95": float(np.percentile(latencies, 95)),
            "latency_ms_min": float(np.min(latencies)),
            "latency_ms_max": float(np.max(latencies)),
            "throughput_fps": 1000.0 / mean if mean else 0.0,
            "artifact": str(args.model),
            "notes": "End-to-end TFLite interpreter latency including tensor set/get",
        })


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    interpreter = Interpreter(model_path=str(args.model), num_threads=4)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    warmup_image = load_image(args.dataset_dir / rows[0][0], args.img_size)
    if np.issubdtype(input_detail["dtype"], np.integer):
        warmup_image = quantize(warmup_image, input_detail)
    for _ in range(args.warmup):
        interpreter.set_tensor(input_detail["index"], warmup_image)
        interpreter.invoke()

    predictions: list[int] = []
    labels: list[int] = []
    latencies: list[float] = []
    for rel_path, label in rows:
        sample = load_image(args.dataset_dir / rel_path, args.img_size)
        if np.issubdtype(input_detail["dtype"], np.integer):
            sample = quantize(sample, input_detail)
        start = time.perf_counter_ns()
        interpreter.set_tensor(input_detail["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        end = time.perf_counter_ns()
        if np.issubdtype(output_detail["dtype"], np.integer):
            output = dequantize(output, output_detail)
        predictions.append(int(np.argmax(output.reshape(output.shape[0], -1), axis=-1)[0]))
        labels.append(label)
        latencies.append((end - start) / 1_000_000.0)

    pred = np.asarray(predictions)
    y = np.asarray(labels)
    latency = np.asarray(latencies, dtype=np.float64)
    accuracy = float(np.mean(pred == y))
    append_row(args, accuracy, latency, len(y))
    print({
        "accuracy": accuracy,
        "latency_ms_mean": float(np.mean(latency)),
        "latency_ms_p95": float(np.percentile(latency, 95)),
        "num_samples": len(y),
    })


if __name__ == "__main__":
    main()
