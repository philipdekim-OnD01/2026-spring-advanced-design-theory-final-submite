#!/usr/bin/env python3
"""Benchmark TensorRT engines with native INT8 bindings.

The input is quantized on the host before timing. CHW4/DLA_HWC4 bindings are
packed as four-channel pixels, and DLA_LINEAR output padding is handled using
the runtime-reported strides.
"""

from __future__ import annotations

import argparse
import csv
import socket
import struct
import time
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image


LOGGER = trt.Logger(trt.Logger.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--calib-cache", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--img-size", default=64, type=int)
    parser.add_argument("--warmup", default=100, type=int)
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", required=True)
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


def load_calibration_scales(path: Path) -> dict[str, float]:
    scales: dict[str, float] = {}
    for line in path.read_text().splitlines()[1:]:
        if ": " not in line:
            continue
        name, encoded = line.rsplit(": ", 1)
        scales[name] = struct.unpack(">f", bytes.fromhex(encoded))[0]
    return scales


def quantize_image(path: Path, img_size: int, scale: float) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        values = np.asarray(image, dtype=np.float32) / 255.0
    quantized = np.clip(np.rint(values / scale), -128, 127).astype(np.int8)
    packed = np.zeros((img_size, img_size, 4), dtype=np.int8)
    packed[:, :, :3] = quantized
    return np.ascontiguousarray(packed.ravel())


class NativeInt8Inference:
    def __init__(self, engine_path: Path) -> None:
        runtime = trt.Runtime(LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings: dict[str, dict] = {}
        self.input_name: str | None = None
        self.output_name: str | None = None
        self._allocate()

    def _physical_elements(self, name: str) -> int:
        shape = tuple(self.context.get_tensor_shape(name))
        strides = tuple(self.context.get_tensor_strides(name))
        components = int(self.engine.get_tensor_components_per_element(name))
        components = max(1, components)
        if any(dim < 0 for dim in shape) or any(stride < 0 for stride in strides):
            raise ValueError(f"Dynamic or invalid tensor layout for {name}: {shape}, {strides}")
        return int(shape[0] * strides[0] * components)

    def _allocate(self) -> None:
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if dtype != np.int8:
                raise ValueError(f"Native benchmark requires INT8 bindings: {name} is {dtype}")
            size = self._physical_elements(name)
            host = cuda.pagelocked_empty(size, dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(device))
            self.bindings[name] = {
                "host": host,
                "device": device,
                "logical_shape": tuple(self.context.get_tensor_shape(name)),
                "format": str(self.engine.get_tensor_format(name)),
                "format_desc": self.engine.get_tensor_format_desc(name),
            }
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Engine must have one input and one output")

    def describe(self) -> dict[str, dict]:
        return {
            name: {
                "logical_shape": binding["logical_shape"],
                "physical_elements": int(binding["host"].size),
                "format": binding["format"],
                "format_desc": binding["format_desc"],
            }
            for name, binding in self.bindings.items()
        }

    def infer(self, packed_input: np.ndarray) -> np.ndarray:
        input_binding = self.bindings[self.input_name]
        output_binding = self.bindings[self.output_name]
        if packed_input.size != input_binding["host"].size:
            raise ValueError(
                f"Packed input has {packed_input.size} elements, expected {input_binding['host'].size}"
            )
        np.copyto(input_binding["host"], packed_input)
        cuda.memcpy_htod_async(input_binding["device"], input_binding["host"], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(output_binding["host"], output_binding["device"], self.stream)
        self.stream.synchronize()
        logical_size = int(np.prod(output_binding["logical_shape"]))
        return np.asarray(output_binding["host"][:logical_size]).copy()


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
            "backend": "tensorrt_native_io",
            "precision": "int8",
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
            "artifact": str(args.engine),
            "notes": "Native INT8 I/O; latency includes H2D/D2H and sync; host quantization excluded",
        })


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    scales = load_calibration_scales(args.calib_cache)
    runner = NativeInt8Inference(args.engine)
    input_scale = scales.get(runner.input_name)
    if input_scale is None:
        raise KeyError(f"No calibration scale for input {runner.input_name}")

    packed_samples = [
        quantize_image(args.dataset_dir / rel_path, args.img_size, input_scale)
        for rel_path, _ in rows
    ]
    for _ in range(args.warmup):
        runner.infer(packed_samples[0])

    predictions: list[int] = []
    labels: list[int] = []
    latencies: list[float] = []
    for _ in range(args.repeat):
        for packed, (_, label) in zip(packed_samples, rows):
            start = time.perf_counter_ns()
            output = runner.infer(packed)
            end = time.perf_counter_ns()
            predictions.append(int(np.argmax(output)))
            labels.append(label)
            latencies.append((end - start) / 1_000_000.0)

    pred = np.asarray(predictions)
    expected = np.asarray(labels)
    latency = np.asarray(latencies, dtype=np.float64)
    accuracy = float(np.mean(pred == expected))
    append_row(args, accuracy, latency, len(expected))
    print({
        "accuracy": accuracy,
        "latency_ms_mean": float(np.mean(latency)),
        "latency_ms_p95": float(np.percentile(latency, 95)),
        "num_samples": len(expected),
        "bindings": runner.describe(),
    })


if __name__ == "__main__":
    main()
