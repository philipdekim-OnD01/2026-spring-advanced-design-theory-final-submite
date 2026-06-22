#!/usr/bin/env python3
"""Build TensorRT GPU or DLA engines from ONNX.

For INT8 engines this script uses TensorRT's entropy calibrator with the RPS
training split from the manifest. DLA builds can either allow GPU fallback or
enforce strict DLA execution with native INT8 bindings.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import tensorrt as trt
from PIL import Image


LOGGER = trt.Logger(trt.Logger.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--precision", choices=["fp16", "int8"], default="int8")
    parser.add_argument("--device", choices=["gpu", "dla"], default="gpu")
    parser.add_argument("--dla-core", default=0, type=int)
    parser.add_argument("--img-size", default=64, type=int)
    parser.add_argument("--calib-batch-size", default=16, type=int)
    parser.add_argument("--calib-samples", default=512, type=int)
    parser.add_argument("--workspace-mib", default=1024, type=int)
    parser.add_argument("--native-int8-io", action="store_true")
    parser.add_argument("--strict-dla", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path, split: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                rows.append((row["path"], int(row["label"])))
    if not rows:
        raise ValueError(f"No {split} rows in manifest: {path}")
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


def load_batch(
    dataset_dir: Path,
    rows: list[tuple[str, int]],
    img_size: int,
    layout: str,
) -> np.ndarray:
    batch = np.empty((len(rows), img_size, img_size, 3), dtype=np.float32)
    for i, (rel_path, _) in enumerate(rows):
        with Image.open(dataset_dir / rel_path) as image:
            image = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            batch[i] = np.asarray(image, dtype=np.float32) / 255.0
    if layout == "nchw":
        batch = np.transpose(batch, (0, 3, 1, 2))
    return np.ascontiguousarray(batch)


class RPSCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        dataset_dir: Path,
        manifest: Path,
        img_size: int,
        batch_size: int,
        max_samples: int,
        cache_file: Path,
        layout: str,
    ) -> None:
        super().__init__()
        rows = load_manifest(manifest, "train")[:max_samples]
        self.dataset_dir = dataset_dir
        self.rows = rows
        self.img_size = img_size
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.layout = layout
        self.index = 0
        self.device_input = cuda.mem_alloc(batch_size * img_size * img_size * 3 * np.dtype(np.float32).itemsize)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        if self.index >= len(self.rows):
            return None
        batch_rows = self.rows[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        if len(batch_rows) < self.batch_size:
            return None
        batch = load_batch(self.dataset_dir, batch_rows, self.img_size, self.layout)
        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self) -> bytes | None:
        if self.cache_file.exists():
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(cache)


def main() -> None:
    args = parse_args()
    if args.native_int8_io and args.precision != "int8":
        raise SystemExit("--native-int8-io requires --precision int8")
    if args.strict_dla and args.device != "dla":
        raise SystemExit("--strict-dla requires --device dla")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib << 20)

    input_tensor = network.get_input(0)
    input_shape = tuple(input_tensor.shape)
    input_layout = infer_layout(input_shape)
    static_shape = static_input_shape(input_shape, args.img_size)
    if any(dim < 0 for dim in input_shape):
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, static_shape, static_shape, static_shape)
        config.add_optimization_profile(profile)

    if args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    if args.precision == "int8":
        if args.dataset_dir is None or args.manifest is None:
            raise SystemExit("--dataset-dir and --manifest are required for INT8 calibration")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = RPSCalibrator(
            args.dataset_dir,
            args.manifest,
            args.img_size,
            args.calib_batch_size,
            args.calib_samples,
            args.engine.with_suffix(".calib.cache"),
            input_layout,
        )

    if args.native_int8_io:
        output_tensor = network.get_output(0)
        input_tensor.dtype = trt.int8
        output_tensor.dtype = trt.int8
        if args.device == "dla":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.DLA_HWC4)
            output_tensor.allowed_formats = 1 << int(trt.TensorFormat.DLA_LINEAR)
        else:
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)
            output_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
            config.set_flag(trt.BuilderFlag.DIRECT_IO)

    if args.device == "dla":
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = args.dla_core
        if not args.strict_dla:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        if args.precision == "int8":
            config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise SystemExit("TensorRT engine build failed")
    args.engine.write_bytes(serialized_engine)
    print(f"wrote {args.engine}")


if __name__ == "__main__":
    main()
