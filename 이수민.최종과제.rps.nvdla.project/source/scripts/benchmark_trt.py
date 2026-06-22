#!/usr/bin/env python3
"""Benchmark TensorRT engine accuracy and latency on the fixed RPS test split."""

from __future__ import annotations

import argparse
import csv
import socket
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
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--img-size", default=64, type=int)
    parser.add_argument("--warmup", default=50, type=int)
    parser.add_argument("--repeat", default=1, type=int)
    parser.add_argument("--model-name", default="rps_dla_cnn_int8")
    parser.add_argument("--backend", default="tensorrt")
    parser.add_argument("--precision", default="int8")
    parser.add_argument("--device", default="gpu")
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


def infer_layout(shape: tuple[int, ...]) -> str:
    if len(shape) == 4 and shape[1] == 3:
        return "nchw"
    if len(shape) == 4 and shape[3] == 3:
        return "nhwc"
    return "nhwc"


def static_input_shape(shape: tuple[int, ...], img_size: int) -> tuple[int, int, int, int]:
    layout = infer_layout(shape)
    if layout == "nchw":
        return (1, 3, img_size, img_size)
    return (1, img_size, img_size, 3)


def load_image(path: Path, img_size: int, layout: str) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        sample = (np.asarray(image, dtype=np.float32) / 255.0)[None, ...]
    if layout == "nchw":
        sample = np.transpose(sample, (0, 3, 1, 2))
    return np.ascontiguousarray(sample)


class TRTInference:
    def __init__(self, engine_path: Path, img_size: int) -> None:
        runtime = trt.Runtime(LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.img_size = img_size
        self.bindings: dict[str, dict] = {}
        self.binding_addrs: list[int] = []
        self.input_name: str | None = None
        self.output_name: str | None = None
        self.input_layout = "nhwc"
        self._allocate()

    def _allocate(self) -> None:
        if hasattr(self.engine, "num_io_tensors"):
            self._allocate_v3()
        else:
            self._allocate_v2()
        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Engine must have one input and one output")

    def _allocate_v3(self) -> None:
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))
            if mode == trt.TensorIOMode.INPUT:
                if any(dim < 0 for dim in shape):
                    shape = static_input_shape(shape, self.img_size)
                    self.context.set_input_shape(name, shape)
                self.input_layout = infer_layout(shape)
                self.input_name = name
            else:
                self.output_name = name
                shape = tuple(self.context.get_tensor_shape(name))
            size = int(np.prod(shape))
            host = cuda.pagelocked_empty(size, dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(device))
            self.bindings[name] = {
                "host": host,
                "device": device,
                "shape": shape,
                "dtype": dtype,
                "mode": mode,
            }

    def _allocate_v2(self) -> None:
        self.binding_addrs = [0] * self.engine.num_bindings
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            is_input = self.engine.binding_is_input(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.engine.get_binding_shape(i))
            if is_input:
                if any(dim < 0 for dim in shape):
                    shape = static_input_shape(shape, self.img_size)
                    self.context.set_binding_shape(i, shape)
                self.input_layout = infer_layout(shape)
                self.input_name = name
            else:
                self.output_name = name
                shape = tuple(self.context.get_binding_shape(i))
            size = int(np.prod(shape))
            host = cuda.pagelocked_empty(size, dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.binding_addrs[i] = int(device)
            self.bindings[name] = {
                "host": host,
                "device": device,
                "shape": shape,
                "dtype": dtype,
                "mode": "input" if is_input else "output",
                "index": i,
            }

    def infer(self, sample: np.ndarray) -> np.ndarray:
        input_binding = self.bindings[self.input_name]
        output_binding = self.bindings[self.output_name]
        np.copyto(input_binding["host"], sample.astype(input_binding["dtype"]).ravel())
        cuda.memcpy_htod_async(input_binding["device"], input_binding["host"], self.stream)
        if hasattr(self.context, "execute_async_v3"):
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=self.binding_addrs, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(output_binding["host"], output_binding["device"], self.stream)
        self.stream.synchronize()
        return np.asarray(output_binding["host"]).reshape(output_binding["shape"]).copy()


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
            "backend": args.backend,
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
            "artifact": str(args.engine),
            "notes": "End-to-end TensorRT latency including H2D/D2H copies",
        })


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    runner = TRTInference(args.engine, args.img_size)

    warmup = load_image(args.dataset_dir / rows[0][0], args.img_size, runner.input_layout)
    for _ in range(args.warmup):
        runner.infer(warmup)

    predictions: list[int] = []
    labels: list[int] = []
    latencies: list[float] = []
    for _ in range(args.repeat):
        for rel_path, label in rows:
            sample = load_image(args.dataset_dir / rel_path, args.img_size, runner.input_layout)
            start = time.perf_counter_ns()
            output = runner.infer(sample)
            end = time.perf_counter_ns()
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
