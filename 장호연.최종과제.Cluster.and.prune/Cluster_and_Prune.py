"""
CPU-friendly rock/paper/scissors model compression demo.

What it does:
1. Downloads a small public rock-paper-scissors image dataset.
2. Trains a MobileNetV2 transfer-learning baseline.
3. Creates compressed variants with:
   - magnitude pruning
   - weight clustering
   - pruning + clustering
4. Compares accuracy, model size, and CPU inference latency.
5. Saves compression-aware weight packages that actually exploit sparsity/codebooks.
6. Exports Arial-styled result tables and blue/gray visualizations.
7. Optionally compares hardware-aware strategies that can reduce real CPU/TFLite work.

Install dependencies:
    python -m pip install "tensorflow>=2.15,<2.17" "numpy<2" pillow matplotlib

Run:
    python C:\PythonProject\Cluster_and_Prune.py

Fast smoke test:
    python C:\PythonProject\Cluster_and_Prune.py --epochs 1 --fine-tune-epochs 1 --max-train 450 --max-val 150
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import numpy as np
    import tensorflow as tf
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with:\n"
        '  python -m pip install "tensorflow>=2.15,<2.17" "numpy<2" pillow'
    ) from exc


DATA_URLS = {
    "train": "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip",
    "val": "https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip",
}


@dataclass
class Result:
    name: str
    accuracy: float
    loss: float
    packed_mb: float
    zipped_weights_mb: float
    raw_mb: float
    latency_p50_ms: float
    latency_mean_ms: float
    latency_p95_ms: float
    batch_latency_p50_ms: float
    throughput_images_per_sec: float
    nonzero_ratio: float


@dataclass
class HardwareResult:
    name: str
    strategy: str
    accuracy: float
    size_mb: float
    size_reduction_percent: float
    latency_p50_ms: float
    latency_mean_ms: float
    latency_p95_ms: float
    batch_latency_p50_ms: float
    throughput_images_per_sec: float
    params_million: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rock/Paper/Scissors clustering and pruning demo.")
    parser.add_argument("--workdir", default=str(Path(__file__).with_suffix("")))
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fine-tune-epochs", type=int, default=2)
    parser.add_argument("--max-train", type=int, default=1200, help="Limit train images for CPU speed. Use 0 for all.")
    parser.add_argument("--max-val", type=int, default=360, help="Limit validation images for CPU speed. Use 0 for all.")
    parser.add_argument("--prune-sparsity", type=float, default=0.50, help="Fraction of eligible Conv2D weights set to zero.")
    parser.add_argument("--prune-depthwise", action="store_true", help="Also prune DepthwiseConv2D layers. Smaller, but usually hurts accuracy more.")
    parser.add_argument("--prune-dense", action="store_true", help="Also prune Dense layers. Off by default because the classifier is tiny and accuracy-sensitive.")
    parser.add_argument("--recovery-unfreeze-last", type=int, default=24, help="Unfreeze this many final backbone layers while recovering the pruned model.")
    parser.add_argument("--recovery-learning-rate", type=float, default=0.00001)
    parser.add_argument("--clusters", type=int, default=256, help="Number of centroids per layer for weight clustering. 256 still packs into uint8 codes.")
    parser.add_argument("--cluster-depthwise", action="store_true", help="Also cluster DepthwiseConv2D layers. Smaller, but usually hurts accuracy more.")
    parser.add_argument("--cluster-dense", action="store_true", help="Also cluster Dense layers. Off by default because the classifier is tiny and accuracy-sensitive.")
    parser.add_argument("--cluster-fine-tune-epochs", type=int, default=1, help="Short recovery fine-tuning after clustering, then weights are re-clustered.")
    parser.add_argument("--cluster-recovery-unfreeze-last", type=int, default=16, help="Unfreeze this many final backbone layers while recovering the clustered model.")
    parser.add_argument("--profile-rounds", type=int, default=50, help="Timed inference runs per model for latency profiling.")
    parser.add_argument("--profile-warmup", type=int, default=8, help="Warmup inference runs before timing.")
    parser.add_argument("--skip-hardware-aware", action="store_true", help="Skip structured-pruning/TFLite deployment comparison.")
    parser.add_argument("--skip-tflite", action="store_true", help="Skip TFLite conversion while keeping the structured slimming hardware-aware comparison.")
    parser.add_argument("--hardware-alpha", type=float, default=0.50, help="MobileNetV2 width multiplier for hardware-friendly structured channel slimming.")
    parser.add_argument("--hardware-epochs", type=int, default=3, help="Training epochs for the hardware-friendly slim model.")
    parser.add_argument("--hardware-fine-tune-epochs", type=int, default=1, help="Recovery fine-tuning epochs for the slim model.")
    parser.add_argument("--hardware-unfreeze-last", type=int, default=16, help="Unfreeze this many final slim backbone layers during recovery fine-tuning.")
    parser.add_argument("--tflite-representative-batches", type=int, default=20, help="Calibration batches for full INT8 TFLite conversion.")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def configure_cpu(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(max(2, os.cpu_count() or 2))


def download_and_extract(workdir: Path) -> tuple[Path, Path]:
    data_dir = workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split, url in DATA_URLS.items():
        zip_path = tf.keras.utils.get_file(
            fname=f"{split}.zip",
            origin=url,
            cache_dir=str(data_dir),
            cache_subdir="downloads",
            extract=False,
        )
        out_dir = data_dir / split
        marker = out_dir / ".extracted"
        if not marker.exists():
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(out_dir)
            marker.write_text("ok", encoding="utf-8")
        paths[split] = out_dir

    train_root = next(paths["train"].glob("rps"))
    val_root = next(paths["val"].glob("rps-test-set"))
    return train_root, val_root


def make_dataset(
    root: Path,
    img_size: int,
    batch_size: int,
    seed: int,
    max_images: int,
    shuffle: bool,
) -> tf.data.Dataset:
    ds = tf.keras.utils.image_dataset_from_directory(
        root,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    if max_images > 0:
        max_batches = max(1, (max_images + batch_size - 1) // batch_size)
        ds = ds.take(max_batches)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(
    img_size: int,
    num_classes: int,
    alpha: float = 1.0,
    name: str = "rps_mobilenetv2_base",
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=name)
    compile_model(model, learning_rate=0.001)
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 0.0005) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def clone_with_weights(model: tf.keras.Model, name: str) -> tf.keras.Model:
    cloned = tf.keras.models.clone_model(model)
    cloned.set_weights(copy.deepcopy(model.get_weights()))
    cloned._name = name
    compile_model(cloned)
    return cloned


def iter_model_layers(model: tf.keras.Model):
    seen: set[int] = set()

    def visit(layer):
        for child in getattr(layer, "layers", []):
            if id(child) in seen:
                continue
            seen.add(id(child))
            yield child
            yield from visit(child)

    yield from visit(model)


def is_compressible_layer(layer: tf.keras.layers.Layer) -> bool:
    return isinstance(
        layer,
        (
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D,
            tf.keras.layers.Dense,
        ),
    )


def is_prunable_layer(
    layer: tf.keras.layers.Layer,
    prune_depthwise: bool,
    prune_dense: bool,
) -> bool:
    if isinstance(layer, tf.keras.layers.Conv2D):
        return True
    if prune_depthwise and isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        return True
    if prune_dense and isinstance(layer, tf.keras.layers.Dense):
        return True
    return False


def is_clusterable_layer(
    layer: tf.keras.layers.Layer,
    cluster_depthwise: bool,
    cluster_dense: bool,
) -> bool:
    if isinstance(layer, tf.keras.layers.Conv2D):
        return True
    if cluster_depthwise and isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        return True
    if cluster_dense and isinstance(layer, tf.keras.layers.Dense):
        return True
    return False


def prune_model(
    model: tf.keras.Model,
    sparsity: float,
    prune_depthwise: bool = False,
    prune_dense: bool = False,
) -> tuple[tf.keras.Model, list[tuple[tf.keras.layers.Layer, np.ndarray]]]:
    pruned = clone_with_weights(model, "rps_mobilenetv2_pruned")
    masks: list[tuple[tf.keras.layers.Layer, np.ndarray]] = []
    for layer in iter_model_layers(pruned):
        if not is_prunable_layer(layer, prune_depthwise, prune_dense):
            continue
        weights = layer.get_weights()
        if not weights:
            continue
        kernel = weights[0]
        threshold = np.quantile(np.abs(kernel), sparsity)
        mask = (np.abs(kernel) > threshold).astype(np.float32)
        kernel = kernel * mask
        weights[0] = kernel.astype(np.float32)
        layer.set_weights(weights)
        masks.append((layer, mask))
    return pruned, masks


class PruningMaskCallback(tf.keras.callbacks.Callback):
    def __init__(self, masks: list[tuple[tf.keras.layers.Layer, np.ndarray]]):
        super().__init__()
        self.masks = masks

    def _apply_masks(self) -> None:
        for layer, mask in self.masks:
            weights = layer.get_weights()
            if not weights:
                continue
            weights[0] = (weights[0] * mask).astype(np.float32)
            layer.set_weights(weights)

    def on_train_batch_end(self, batch, logs=None) -> None:
        self._apply_masks()

    def on_train_end(self, logs=None) -> None:
        self._apply_masks()


def nearest_centroid_labels(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    centroids = np.sort(centroids.astype(np.float32))
    if len(centroids) == 1:
        return np.zeros_like(values, dtype=np.int64)
    right = np.searchsorted(centroids, values, side="left")
    right = np.clip(right, 1, len(centroids) - 1)
    left = right - 1
    choose_right = np.abs(values - centroids[right]) < np.abs(values - centroids[left])
    return np.where(choose_right, right, left)


def one_dimensional_kmeans(
    values: np.ndarray,
    k: int,
    iterations: int = 8,
    preserve_zeros: bool = False,
) -> np.ndarray:
    zero_mask = values == 0 if preserve_zeros else None
    flat = values.reshape(-1)
    if preserve_zeros:
        flat = flat[flat != 0]
    if flat.size == 0:
        return values

    unique = np.unique(flat)
    if unique.size <= k:
        return values

    percentiles = np.linspace(0, 100, k)
    centroids = np.percentile(flat, percentiles).astype(np.float32)

    for _ in range(iterations):
        centroids = np.sort(centroids.astype(np.float32))
        labels = nearest_centroid_labels(flat, centroids)
        sums = np.bincount(labels, weights=flat, minlength=k)
        counts = np.bincount(labels, minlength=k)
        used = counts > 0
        centroids[used] = (sums[used] / counts[used]).astype(np.float32)

    centroids = np.sort(centroids.astype(np.float32))
    labels = nearest_centroid_labels(flat, centroids)
    clustered_values = values.copy().reshape(-1)
    if preserve_zeros:
        nonzero_positions = np.flatnonzero(~zero_mask.reshape(-1))
        clustered_values[nonzero_positions] = centroids[labels]
    else:
        clustered_values = centroids[labels]
    return clustered_values.reshape(values.shape).astype(np.float32)


def cluster_model(
    model: tf.keras.Model,
    clusters: int,
    preserve_zeros: bool = False,
    cluster_depthwise: bool = False,
    cluster_dense: bool = False,
) -> tuple[tf.keras.Model, int]:
    clustered = clone_with_weights(model, "rps_mobilenetv2_clustered")
    clustered_count = 0
    for layer in iter_model_layers(clustered):
        if not is_clusterable_layer(layer, cluster_depthwise, cluster_dense):
            continue
        weights = layer.get_weights()
        if not weights:
            continue
        weights[0] = one_dimensional_kmeans(
            weights[0].astype(np.float32),
            clusters,
            preserve_zeros=preserve_zeros,
        )
        layer.set_weights(weights)
        clustered_count += 1
    return clustered, clustered_count


def set_recovery_trainable(model: tf.keras.Model, unfreeze_last: int) -> None:
    for layer in model.layers:
        layer.trainable = False

    nested_models = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    for nested in nested_models:
        nested.trainable = True
        for layer in nested.layers:
            layer.trainable = False
        if unfreeze_last > 0:
            for layer in nested.layers[-unfreeze_last:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.trainable = True


def fine_tune(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    learning_rate: float = 0.0005,
    pruning_masks: list[tuple[tf.keras.layers.Layer, np.ndarray]] | None = None,
) -> None:
    if epochs <= 0:
        return
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=1,
            mode="max",
            restore_best_weights=True,
        )
    ]
    if pruning_masks:
        callbacks.append(PruningMaskCallback(pruning_masks))
    compile_model(model, learning_rate=learning_rate)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)


def smallest_unsigned_dtype(max_value: int) -> np.dtype:
    if max_value <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def add_dense_array(arrays: dict[str, np.ndarray], metadata: list[dict], key: str, weight_name: str, values: np.ndarray) -> None:
    arrays[f"{key}_values"] = values.astype(np.float32, copy=False)
    metadata.append(
        {
            "key": key,
            "kind": "dense",
            "name": weight_name,
            "shape": list(values.shape),
            "dtype": "float32",
        }
    )


def add_sparse_array(
    arrays: dict[str, np.ndarray],
    metadata: list[dict],
    key: str,
    weight_name: str,
    values: np.ndarray,
) -> None:
    flat = values.reshape(-1).astype(np.float32, copy=False)
    mask = flat != 0
    arrays[f"{key}_mask"] = np.packbits(mask.astype(np.uint8))
    arrays[f"{key}_values"] = flat[mask]
    metadata.append(
        {
            "key": key,
            "kind": "sparse",
            "name": weight_name,
            "shape": list(values.shape),
            "dtype": "float32",
            "num_values": int(flat.size),
            "num_nonzero": int(mask.sum()),
        }
    )


def add_codebook_array(
    arrays: dict[str, np.ndarray],
    metadata: list[dict],
    key: str,
    weight_name: str,
    values: np.ndarray,
) -> None:
    flat = values.reshape(-1).astype(np.float32, copy=False)
    codebook, inverse = np.unique(flat, return_inverse=True)
    code_dtype = smallest_unsigned_dtype(len(codebook) - 1)
    arrays[f"{key}_codebook"] = codebook.astype(np.float32, copy=False)
    arrays[f"{key}_codes"] = inverse.astype(code_dtype, copy=False)
    metadata.append(
        {
            "key": key,
            "kind": "codebook",
            "name": weight_name,
            "shape": list(values.shape),
            "dtype": "float32",
            "code_dtype": np.dtype(code_dtype).name,
            "num_codes": int(len(codebook)),
        }
    )


def add_sparse_codebook_array(
    arrays: dict[str, np.ndarray],
    metadata: list[dict],
    key: str,
    weight_name: str,
    values: np.ndarray,
) -> None:
    flat = values.reshape(-1).astype(np.float32, copy=False)
    mask = flat != 0
    nonzero_values = flat[mask]
    codebook, inverse = np.unique(nonzero_values, return_inverse=True)
    code_dtype = smallest_unsigned_dtype(len(codebook) - 1)
    arrays[f"{key}_mask"] = np.packbits(mask.astype(np.uint8))
    arrays[f"{key}_codebook"] = codebook.astype(np.float32, copy=False)
    arrays[f"{key}_codes"] = inverse.astype(code_dtype, copy=False)
    metadata.append(
        {
            "key": key,
            "kind": "sparse_codebook",
            "name": weight_name,
            "shape": list(values.shape),
            "dtype": "float32",
            "code_dtype": np.dtype(code_dtype).name,
            "num_values": int(flat.size),
            "num_nonzero": int(mask.sum()),
            "num_codes": int(len(codebook)),
        }
    )


def save_compression_aware_package(model: tf.keras.Model, out_dir: Path, name: str) -> Path:
    package_path = out_dir / f"{name}.compressed_weights.npz"
    arrays: dict[str, np.ndarray] = {}
    metadata: list[dict] = []

    for idx, weight in enumerate(model.weights):
        values = weight.numpy()
        key = f"w{idx:03d}"
        weight_name = getattr(weight, "path", None) or weight.name

        if values.dtype.kind != "f" or values.size < 128:
            add_dense_array(arrays, metadata, key, weight_name, values)
            continue

        values = values.astype(np.float32, copy=False)
        flat = values.reshape(-1)
        zero_ratio = float(np.mean(flat == 0))
        unique_count = len(np.unique(flat))

        if zero_ratio >= 0.15 and unique_count <= 257:
            add_sparse_codebook_array(arrays, metadata, key, weight_name, values)
        elif zero_ratio >= 0.15:
            add_sparse_array(arrays, metadata, key, weight_name, values)
        elif unique_count <= 257:
            add_codebook_array(arrays, metadata, key, weight_name, values)
        else:
            add_dense_array(arrays, metadata, key, weight_name, values)

    arrays["metadata_json"] = np.array(json.dumps(metadata), dtype=np.str_)
    np.savez_compressed(package_path, **arrays)
    return package_path


def save_model_files(model: tf.keras.Model, out_dir: Path, name: str) -> tuple[float, float, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    keras_path = out_dir / f"{name}.keras"
    zipped_path = out_dir / f"{name}.weights.zip"
    packed_path = save_compression_aware_package(model, out_dir, name)

    model.save(keras_path, include_optimizer=False)
    with tempfile.TemporaryDirectory() as temp_dir:
        weights_path = Path(temp_dir) / f"{name}.weights.h5"
        model.save_weights(weights_path)
        with zipfile.ZipFile(zipped_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.write(weights_path, arcname=weights_path.name)

    raw_mb = keras_path.stat().st_size / (1024 * 1024)
    zipped_weights_mb = zipped_path.stat().st_size / (1024 * 1024)
    packed_mb = packed_path.stat().st_size / (1024 * 1024)
    return packed_mb, zipped_weights_mb, raw_mb


def timed_forward_passes(
    model: tf.keras.Model,
    images: tf.Tensor,
    warmup: int,
    rounds: int,
) -> list[float]:
    for _ in range(warmup):
        model(images, training=False).numpy()

    timings = []
    for _ in range(rounds):
        start = time.perf_counter()
        model(images, training=False).numpy()
        timings.append((time.perf_counter() - start) * 1000)
    return timings


def profile_inference(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    warmup: int,
    rounds: int,
) -> dict[str, float]:
    images = next(iter(val_ds))[0]
    single_image = images[:1]

    single_timings = timed_forward_passes(model, single_image, warmup, rounds)
    batch_timings = timed_forward_passes(model, images, max(2, warmup // 2), max(5, rounds // 2))
    batch_p50_ms = float(np.percentile(batch_timings, 50))
    batch_size = int(images.shape[0])

    return {
        "latency_p50_ms": float(np.percentile(single_timings, 50)),
        "latency_mean_ms": float(statistics.mean(single_timings)),
        "latency_p95_ms": float(np.percentile(single_timings, 95)),
        "batch_latency_p50_ms": batch_p50_ms,
        "throughput_images_per_sec": float(batch_size / (batch_p50_ms / 1000)) if batch_p50_ms > 0 else 0.0,
    }


def quantize_tflite_input(images: np.ndarray, input_detail: dict) -> np.ndarray:
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return images.astype(np.float32)

    scale, zero_point = input_detail.get("quantization", (0.0, 0))
    if not scale:
        return images.astype(dtype)

    quantized = np.round(images / scale + zero_point)
    limits = np.iinfo(dtype)
    return np.clip(quantized, limits.min, limits.max).astype(dtype)


def dequantize_tflite_output(output: np.ndarray, output_detail: dict) -> np.ndarray:
    if output.dtype == np.float32:
        return output
    scale, zero_point = output_detail.get("quantization", (0.0, 0))
    if not scale:
        return output.astype(np.float32)
    return (output.astype(np.float32) - zero_point) * scale


def prepare_tflite_interpreter(interpreter, images: np.ndarray) -> tuple[dict, dict]:
    input_detail = interpreter.get_input_details()[0]
    expected_shape = list(images.shape)
    if list(input_detail["shape"]) != expected_shape:
        interpreter.resize_tensor_input(input_detail["index"], expected_shape, strict=False)
        interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    return input_detail, output_detail


def invoke_tflite(interpreter, images: np.ndarray) -> np.ndarray:
    input_detail, output_detail = prepare_tflite_interpreter(interpreter, images)
    interpreter.set_tensor(input_detail["index"], quantize_tflite_input(images, input_detail))
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])
    return dequantize_tflite_output(output, output_detail)


def representative_dataset(train_ds: tf.data.Dataset, max_batches: int):
    for images, _ in train_ds.take(max_batches):
        for image in images:
            yield [image[None, ...].numpy().astype(np.float32)]


def collect_representative_images(train_ds: tf.data.Dataset, max_batches: int) -> np.ndarray:
    images_for_calibration = []
    for images, _ in train_ds.take(max_batches):
        for image in images:
            images_for_calibration.append(image.numpy().astype(np.float32))
    if not images_for_calibration:
        raise ValueError("No calibration images were available for TFLite conversion.")
    return np.stack(images_for_calibration, axis=0)


def run_tflite_conversion_subprocess(
    model_path: Path,
    calibration_path: Path,
    output_path: Path,
    mode: str,
    timeout_seconds: int = 600,
) -> Path | None:
    converter_script = r'''
import os
import pathlib
import shutil
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

model_path = pathlib.Path(sys.argv[1])
calibration_path = pathlib.Path(sys.argv[2])
output_path = pathlib.Path(sys.argv[3])
mode = sys.argv[4]

model = tf.keras.models.load_model(model_path, compile=False)
saved_model_dir = output_path.parent / (output_path.stem + "_saved_model")
if saved_model_dir.exists():
    shutil.rmtree(saved_model_dir)

try:
    if hasattr(model, "export"):
        model.export(str(saved_model_dir))
    else:
        tf.saved_model.save(model, str(saved_model_dir))
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
except Exception:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

if mode == "int8":
    calibration = np.load(calibration_path)

    def representative_dataset():
        for image in calibration:
            yield [image[None, ...].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

output_path.write_bytes(converter.convert())
'''
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            converter_script,
            str(model_path),
            str(calibration_path),
            str(output_path),
            mode,
        ],
        cwd=str(output_path.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode == 0 and output_path.exists():
        return output_path

    print(f"TFLite {mode} conversion failed for {output_path.stem}.")
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    if stderr:
        print(stderr[-1800:])
    if stdout:
        print(stdout[-800:])
    return None


def convert_to_full_int8_tflite(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    out_dir: Path,
    name: str,
    representative_batches: int,
) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_model_path = out_dir / f"{name}.conversion_source.keras"
    calibration_path = out_dir / f"{name}.calibration.npy"
    tflite_path = out_dir / f"{name}.int8.tflite"

    model.save(source_model_path, include_optimizer=False)
    np.save(calibration_path, collect_representative_images(train_ds, representative_batches))

    try:
        converted_path = run_tflite_conversion_subprocess(
            source_model_path,
            calibration_path,
            tflite_path,
            mode="int8",
        )
        if converted_path:
            return converted_path
    except subprocess.TimeoutExpired:
        print(f"Full INT8 conversion timed out for {name}.")
    except Exception as exc:
        print(f"Full INT8 conversion setup failed for {name}: {exc}")

    fallback_path = out_dir / f"{name}.dynamic_range.tflite"
    try:
        converted_path = run_tflite_conversion_subprocess(
            source_model_path,
            calibration_path,
            fallback_path,
            mode="dynamic",
        )
        if converted_path:
            print(f"Saved dynamic-range TFLite fallback: {fallback_path}")
            return converted_path
    except subprocess.TimeoutExpired:
        print(f"TFLite fallback conversion timed out for {name}.")
    except Exception as exc:
        print(f"TFLite fallback conversion setup failed for {name}: {exc}")

    return None


def evaluate_tflite_accuracy(interpreter, val_ds: tf.data.Dataset) -> float:
    correct = 0
    total = 0
    for images, labels in val_ds:
        predictions = invoke_tflite(interpreter, images.numpy().astype(np.float32))
        true_ids = np.argmax(labels.numpy(), axis=1)
        pred_ids = np.argmax(predictions, axis=1)
        correct += int(np.sum(true_ids == pred_ids))
        total += int(true_ids.size)
    return correct / total if total else 0.0


def timed_tflite_passes(interpreter, images: np.ndarray, warmup: int, rounds: int) -> list[float]:
    for _ in range(warmup):
        invoke_tflite(interpreter, images)

    timings = []
    for _ in range(rounds):
        start = time.perf_counter()
        invoke_tflite(interpreter, images)
        timings.append((time.perf_counter() - start) * 1000)
    return timings


def profile_tflite_interpreter(
    interpreter,
    val_ds: tf.data.Dataset,
    warmup: int,
    rounds: int,
) -> dict[str, float]:
    images = next(iter(val_ds))[0].numpy().astype(np.float32)
    single_image = images[:1]

    single_timings = timed_tflite_passes(interpreter, single_image, warmup, rounds)
    batch_timings = timed_tflite_passes(interpreter, images, max(2, warmup // 2), max(5, rounds // 2))
    batch_p50_ms = float(np.percentile(batch_timings, 50))
    batch_size = int(images.shape[0])

    return {
        "latency_p50_ms": float(np.percentile(single_timings, 50)),
        "latency_mean_ms": float(statistics.mean(single_timings)),
        "latency_p95_ms": float(np.percentile(single_timings, 95)),
        "batch_latency_p50_ms": batch_p50_ms,
        "throughput_images_per_sec": float(batch_size / (batch_p50_ms / 1000)) if batch_p50_ms > 0 else 0.0,
    }


def load_tflite_interpreter(tflite_path: Path):
    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_path),
        num_threads=max(1, min(4, os.cpu_count() or 1)),
    )
    interpreter.allocate_tensors()
    return interpreter


def nonzero_ratio(model: tf.keras.Model) -> float:
    nonzero = 0
    total = 0
    for layer in iter_model_layers(model):
        if not is_compressible_layer(layer):
            continue
        weights = layer.get_weights()
        if not weights:
            continue
        kernel = weights[0]
        nonzero += int(np.count_nonzero(kernel))
        total += int(kernel.size)
    return nonzero / total if total else 1.0


def evaluate_variant(
    model: tf.keras.Model,
    name: str,
    val_ds: tf.data.Dataset,
    out_dir: Path,
    profile_warmup: int,
    profile_rounds: int,
) -> Result:
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    packed_mb, zipped_weights_mb, raw_mb = save_model_files(model, out_dir, name)
    profile = profile_inference(model, val_ds, profile_warmup, profile_rounds)
    return Result(
        name=name,
        accuracy=float(accuracy),
        loss=float(loss),
        packed_mb=packed_mb,
        zipped_weights_mb=zipped_weights_mb,
        raw_mb=raw_mb,
        latency_p50_ms=profile["latency_p50_ms"],
        latency_mean_ms=profile["latency_mean_ms"],
        latency_p95_ms=profile["latency_p95_ms"],
        batch_latency_p50_ms=profile["batch_latency_p50_ms"],
        throughput_images_per_sec=profile["throughput_images_per_sec"],
        nonzero_ratio=nonzero_ratio(model),
    )


def hardware_result_from_keras(
    result: Result,
    strategy: str,
    base_size_mb: float,
    params_million: float,
    size_mb: float | None = None,
) -> HardwareResult:
    measured_size_mb = result.raw_mb if size_mb is None else size_mb
    return HardwareResult(
        name=result.name,
        strategy=strategy,
        accuracy=result.accuracy,
        size_mb=measured_size_mb,
        size_reduction_percent=100 * (1 - measured_size_mb / base_size_mb) if base_size_mb else 0.0,
        latency_p50_ms=result.latency_p50_ms,
        latency_mean_ms=result.latency_mean_ms,
        latency_p95_ms=result.latency_p95_ms,
        batch_latency_p50_ms=result.batch_latency_p50_ms,
        throughput_images_per_sec=result.throughput_images_per_sec,
        params_million=params_million,
    )


def evaluate_slim_keras_variant(
    model: tf.keras.Model,
    name: str,
    strategy: str,
    val_ds: tf.data.Dataset,
    out_dir: Path,
    base_size_mb: float,
    profile_warmup: int,
    profile_rounds: int,
) -> tuple[HardwareResult, Result]:
    result = evaluate_variant(model, name, val_ds, out_dir, profile_warmup, profile_rounds)
    hardware_result = hardware_result_from_keras(
        result,
        strategy=strategy,
        base_size_mb=base_size_mb,
        params_million=model.count_params() / 1_000_000,
    )
    return hardware_result, result


def evaluate_tflite_variant(
    tflite_path: Path,
    name: str,
    strategy: str,
    source_params_million: float,
    val_ds: tf.data.Dataset,
    base_size_mb: float,
    profile_warmup: int,
    profile_rounds: int,
) -> HardwareResult:
    accuracy_interpreter = load_tflite_interpreter(tflite_path)
    accuracy = evaluate_tflite_accuracy(accuracy_interpreter, val_ds)

    profile_interpreter = load_tflite_interpreter(tflite_path)
    profile = profile_tflite_interpreter(profile_interpreter, val_ds, profile_warmup, profile_rounds)
    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    return HardwareResult(
        name=name,
        strategy=strategy,
        accuracy=float(accuracy),
        size_mb=size_mb,
        size_reduction_percent=100 * (1 - size_mb / base_size_mb) if base_size_mb else 0.0,
        latency_p50_ms=profile["latency_p50_ms"],
        latency_mean_ms=profile["latency_mean_ms"],
        latency_p95_ms=profile["latency_p95_ms"],
        batch_latency_p50_ms=profile["batch_latency_p50_ms"],
        throughput_images_per_sec=profile["throughput_images_per_sec"],
        params_million=source_params_million,
    )


def print_results(results: list[Result]) -> None:
    base_size = results[0].packed_mb
    base_acc = results[0].accuracy
    headers = [
        "Model",
        "Accuracy",
        "Acc Delta",
        "Packed Size",
        "Size Down",
        "Zip Weights",
        ".keras",
        "P50",
        "P95",
        "Img/s",
        "Nonzero",
    ]
    rows = []
    for result in results:
        size_down = 100 * (1 - result.packed_mb / base_size) if base_size else 0
        rows.append(
            [
                result.name,
                f"{result.accuracy * 100:.2f}%",
                f"{(result.accuracy - base_acc) * 100:+.2f}p",
                f"{result.packed_mb:.2f} MB",
                f"{size_down:.1f}%",
                f"{result.zipped_weights_mb:.2f} MB",
                f"{result.raw_mb:.2f} MB",
                f"{result.latency_p50_ms:.1f} ms",
                f"{result.latency_p95_ms:.1f} ms",
                f"{result.throughput_images_per_sec:.1f}",
                f"{result.nonzero_ratio * 100:.1f}%",
            ]
        )

    widths = [max(len(str(row[idx])) for row in [headers] + rows) for idx in range(len(headers))]
    print("\nComparison")
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers))))


def write_results_csv(results: list[Result], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "compression_results.csv"
    base_size = results[0].packed_mb
    base_acc = results[0].accuracy
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "accuracy_percent",
                "accuracy_delta_points",
                "loss",
                "packed_size_mb",
                "size_reduction_percent",
                "zipped_weights_size_mb",
                "raw_keras_size_mb",
                "single_latency_p50_ms",
                "single_latency_mean_ms",
                "single_latency_p95_ms",
                "batch_latency_p50_ms",
                "throughput_images_per_sec",
                "nonzero_percent",
            ]
        )
        for result in results:
            size_down = 100 * (1 - result.packed_mb / base_size) if base_size else 0
            writer.writerow(
                [
                    result.name,
                    round(result.accuracy * 100, 4),
                    round((result.accuracy - base_acc) * 100, 4),
                    round(result.loss, 6),
                    round(result.packed_mb, 4),
                    round(size_down, 4),
                    round(result.zipped_weights_mb, 4),
                    round(result.raw_mb, 4),
                    round(result.latency_p50_ms, 4),
                    round(result.latency_mean_ms, 4),
                    round(result.latency_p95_ms, 4),
                    round(result.batch_latency_p50_ms, 4),
                    round(result.throughput_images_per_sec, 4),
                    round(result.nonzero_ratio * 100, 4),
                ]
            )
    return csv_path


def print_hardware_results(results: list[HardwareResult]) -> None:
    headers = [
        "Model",
        "Strategy",
        "Accuracy",
        "Size",
        "Size Down",
        "P50",
        "P95",
        "Img/s",
        "Params",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result.name,
                result.strategy,
                f"{result.accuracy * 100:.2f}%",
                f"{result.size_mb:.2f} MB",
                f"{result.size_reduction_percent:.1f}%",
                f"{result.latency_p50_ms:.1f} ms",
                f"{result.latency_p95_ms:.1f} ms",
                f"{result.throughput_images_per_sec:.1f}",
                f"{result.params_million:.2f}M",
            ]
        )

    widths = [max(len(str(row[idx])) for row in [headers] + rows) for idx in range(len(headers))]
    print("\nHardware-aware comparison")
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers))))


def write_hardware_results_csv(results: list[HardwareResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hardware_aware_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "strategy",
                "accuracy_percent",
                "deployment_size_mb",
                "size_reduction_percent",
                "single_latency_p50_ms",
                "single_latency_mean_ms",
                "single_latency_p95_ms",
                "batch_latency_p50_ms",
                "throughput_images_per_sec",
                "params_million",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.name,
                    result.strategy,
                    round(result.accuracy * 100, 4),
                    round(result.size_mb, 4),
                    round(result.size_reduction_percent, 4),
                    round(result.latency_p50_ms, 4),
                    round(result.latency_mean_ms, 4),
                    round(result.latency_p95_ms, 4),
                    round(result.batch_latency_p50_ms, 4),
                    round(result.throughput_images_per_sec, 4),
                    round(result.params_million, 4),
                ]
            )
    return csv_path


def final_comparison_rows(results: list[HardwareResult]) -> tuple[list[str], list[list[str]]]:
    headers = ["Model (Strategy)", "Accuracy", "Model Size", "Latency"]
    rows = []
    for result in results:
        rows.append(
            [
                f"{result.name} ({result.strategy})",
                f"{result.accuracy * 100:.2f}%",
                f"{result.size_mb:.2f} MB",
                f"{result.latency_p50_ms:.1f} ms",
            ]
        )
    return headers, rows


def print_final_comparison_table(results: list[HardwareResult]) -> None:
    headers, rows = final_comparison_rows(results)
    widths = [max(len(str(row[idx])) for row in [headers] + rows) for idx in range(len(headers))]
    print("\nFinal comparison")
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers))))


def write_final_comparison_csv(results: list[HardwareResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "final_comparison_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model_strategy", "accuracy_percent", "model_size_mb", "latency_p50_ms"])
        for result in results:
            writer.writerow(
                [
                    f"{result.name} ({result.strategy})",
                    round(result.accuracy * 100, 4),
                    round(result.size_mb, 4),
                    round(result.latency_p50_ms, 4),
                ]
            )
    return csv_path


BLUE_SCALE = ["#0B3D91", "#1D5FA7", "#3F7FBF", "#7FA6D8"]
GRAY_SCALE = ["#222222", "#555555", "#888888", "#C7C7C7"]
GRID_COLOR = "#D8E2F0"


def configure_plot_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "axes.edgecolor": "#4A5568",
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "grid.color": GRID_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def style_axis(axis, use_grid: bool = False) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#4A5568")
    axis.spines["bottom"].set_color("#4A5568")
    if use_grid:
        axis.grid(axis="y", alpha=0.6, linewidth=0.8)


def plot_results_dashboard(results: list[Result], out_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('\nVisualization skipped. Install matplotlib with: python -m pip install matplotlib')
        return None
    configure_plot_style(plt)

    out_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = out_dir / "compression_dashboard.png"

    names = [result.name for result in results]
    accuracy = [result.accuracy * 100 for result in results]
    packed_size = [result.packed_mb for result in results]
    latency = [result.latency_p50_ms for result in results]
    nonzero = [result.nonzero_ratio * 100 for result in results]
    base_size = results[0].packed_mb
    size_reduction = [100 * (1 - result.packed_mb / base_size) if base_size else 0 for result in results]

    colors = BLUE_SCALE
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Rock/Paper/Scissors Model Compression Comparison", fontsize=16, fontweight="bold")

    axes[0, 0].bar(names, accuracy, color=colors)
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_ylim(max(0, min(accuracy) - 5), 100)
    for idx, value in enumerate(accuracy):
        axes[0, 0].text(idx, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)

    axes[0, 1].bar(names, packed_size, color=colors)
    axes[0, 1].set_title("Compression-Aware Package Size")
    axes[0, 1].set_ylabel("Size (MB)")
    for idx, value in enumerate(packed_size):
        axes[0, 1].text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    axes[1, 0].bar(names, size_reduction, color=colors)
    axes[1, 0].set_title("Size Reduction vs Base")
    axes[1, 0].set_ylabel("Reduction (%)")
    axes[1, 0].set_ylim(0, max(5, max(size_reduction) * 1.2))
    for idx, value in enumerate(size_reduction):
        axes[1, 0].text(idx, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)

    axes[1, 1].scatter(packed_size, accuracy, s=[max(80, 220 - value) for value in latency], c=colors)
    for idx, name in enumerate(names):
        axes[1, 1].annotate(name, (packed_size[idx], accuracy[idx]), xytext=(6, 6), textcoords="offset points")
    axes[1, 1].set_title("Accuracy vs Size")
    axes[1, 1].set_xlabel("Packed size (MB)")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].grid(alpha=0.5)

    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=20)
        style_axis(axis, use_grid=True)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(dashboard_path, dpi=180)
    plt.close(fig)

    sparsity_path = out_dir / "nonzero_and_latency.png"
    fig, axis_left = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    width = 0.36
    axis_left.bar(x - width / 2, nonzero, width, color=BLUE_SCALE[1], label="Nonzero weights (%)")
    axis_left.set_ylabel("Nonzero weights (%)")
    axis_left.set_ylim(0, 105)
    axis_left.set_xticks(x)
    axis_left.set_xticklabels(names, rotation=20)

    axis_right = axis_left.twinx()
    axis_right.bar(x + width / 2, latency, width, color=GRAY_SCALE[2], label="P50 latency (ms)")
    axis_right.set_ylabel("P50 latency (ms)")

    left_handles, left_labels = axis_left.get_legend_handles_labels()
    right_handles, right_labels = axis_right.get_legend_handles_labels()
    axis_left.legend(left_handles + right_handles, left_labels + right_labels, loc="upper right")
    axis_left.set_title("Sparsity and CPU Inference Latency")
    style_axis(axis_left, use_grid=True)
    axis_right.spines["top"].set_visible(False)
    axis_right.spines["right"].set_color("#4A5568")
    fig.tight_layout()
    fig.savefig(sparsity_path, dpi=180)
    plt.close(fig)

    return dashboard_path


def result_rows_for_display(results: list[Result]) -> tuple[list[str], list[list[str]]]:
    base_size = results[0].packed_mb
    base_acc = results[0].accuracy
    headers = [
        "Model",
        "Accuracy",
        "Acc Delta",
        "Packed Size",
        "Size Down",
        "P50",
        "P95",
        "Img/s",
        "Nonzero",
    ]
    rows = []
    for result in results:
        size_down = 100 * (1 - result.packed_mb / base_size) if base_size else 0
        rows.append(
            [
                result.name,
                f"{result.accuracy * 100:.2f}%",
                f"{(result.accuracy - base_acc) * 100:+.2f}p",
                f"{result.packed_mb:.2f} MB",
                f"{size_down:.1f}%",
                f"{result.latency_p50_ms:.1f} ms",
                f"{result.latency_p95_ms:.1f} ms",
                f"{result.throughput_images_per_sec:.1f}",
                f"{result.nonzero_ratio * 100:.1f}%",
            ]
        )
    return headers, rows


def plot_results_table(results: list[Result], out_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    configure_plot_style(plt)

    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "results_table.png"
    headers, rows = result_rows_for_display(results)

    fig, axis = plt.subplots(figsize=(14, 3.2))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#B9C7DA")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(BLUE_SCALE[0])
            cell.set_text_props(color="white", weight="bold", fontfamily="Arial")
        else:
            cell.set_facecolor("#F5F8FC" if row % 2 else "#E8EEF7")
            cell.set_text_props(color="#222222", fontfamily="Arial")
            if col == 0:
                cell.set_text_props(weight="bold", fontfamily="Arial")

    axis.set_title("Compression and Inference Results", fontsize=15, fontweight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(table_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return table_path


def plot_final_comparison_table(results: list[HardwareResult], out_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    configure_plot_style(plt)

    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "final_comparison_table.png"
    headers, rows = final_comparison_rows(results)

    fig_height = max(3.0, 1.0 + 0.45 * len(rows))
    fig, axis = plt.subplots(figsize=(15, fig_height))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.56, 0.14, 0.15, 0.15],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#B9C7DA")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(BLUE_SCALE[0])
            cell.set_text_props(color="white", weight="bold", fontfamily="Arial")
        else:
            cell.set_facecolor("#F5F8FC" if row % 2 else "#E8EEF7")
            cell.set_text_props(color="#222222", fontfamily="Arial")
            if col == 0:
                cell.set_text_props(weight="bold", fontfamily="Arial")
                cell.get_text().set_ha("left")

    axis.set_title("Final Strategy Comparison", fontsize=15, fontweight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(table_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return table_path


def plot_inference_profile(results: list[Result], out_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    configure_plot_style(plt)

    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = out_dir / "inference_profile.png"

    names = [result.name for result in results]
    p50 = [result.latency_p50_ms for result in results]
    p95 = [result.latency_p95_ms for result in results]
    mean_latency = [result.latency_mean_ms for result in results]
    throughput = [result.throughput_images_per_sec for result in results]
    x = np.arange(len(names))
    width = 0.28

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CPU Inference Speed Profile", fontsize=16, fontweight="bold")

    axes[0].bar(x - width, p50, width, color=BLUE_SCALE[0], label="P50")
    axes[0].bar(x, mean_latency, width, color=BLUE_SCALE[2], label="Mean")
    axes[0].bar(x + width, p95, width, color=GRAY_SCALE[2], label="P95")
    axes[0].set_title("Single Image Latency")
    axes[0].set_ylabel("Milliseconds")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20)
    axes[0].legend()
    style_axis(axes[0], use_grid=True)

    axes[1].bar(names, throughput, color=BLUE_SCALE)
    axes[1].set_title("Batch Throughput")
    axes[1].set_ylabel("Images per second")
    axes[1].tick_params(axis="x", rotation=20)
    for idx, value in enumerate(throughput):
        axes[1].text(idx, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    style_axis(axes[1], use_grid=True)

    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(profile_path, dpi=180)
    plt.close(fig)
    return profile_path


def plot_hardware_aware_comparison(results: list[HardwareResult], out_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    configure_plot_style(plt)

    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / "hardware_aware_comparison.png"

    names = [result.name for result in results]
    accuracy = [result.accuracy * 100 for result in results]
    size = [result.size_mb for result in results]
    latency = [result.latency_p50_ms for result in results]
    throughput = [result.throughput_images_per_sec for result in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Hardware-Aware Compression and Deployment Comparison", fontsize=16, fontweight="bold")

    axes[0, 0].bar(names, accuracy, color=BLUE_SCALE)
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_ylim(max(0, min(accuracy) - 5), 100)
    for idx, value in enumerate(accuracy):
        axes[0, 0].text(idx, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)

    axes[0, 1].bar(names, size, color=BLUE_SCALE)
    axes[0, 1].set_title("Deployment File Size")
    axes[0, 1].set_ylabel("Size (MB)")
    for idx, value in enumerate(size):
        axes[0, 1].text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    bar_colors = (GRAY_SCALE + BLUE_SCALE)[: len(names)]
    axes[1, 0].bar(names, latency, color=bar_colors)
    axes[1, 0].set_title("Single Image P50 Latency")
    axes[1, 0].set_ylabel("Milliseconds")
    for idx, value in enumerate(latency):
        axes[1, 0].text(idx, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    axes[1, 1].scatter(latency, accuracy, s=[max(80, item * 2.0) for item in throughput], c=BLUE_SCALE)
    for idx, name in enumerate(names):
        axes[1, 1].annotate(name, (latency[idx], accuracy[idx]), xytext=(6, 6), textcoords="offset points")
    axes[1, 1].set_title("Accuracy vs Latency")
    axes[1, 1].set_xlabel("P50 latency (ms)")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].grid(alpha=0.5)

    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=20)
        style_axis(axis, use_grid=True)

    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)
    return comparison_path


def confusion_matrix_for_model(model: tf.keras.Model, val_ds: tf.data.Dataset, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        true_ids = np.argmax(labels.numpy(), axis=1)
        pred_ids = np.argmax(predictions, axis=1)
        for true_id, pred_id in zip(true_ids, pred_ids):
            matrix[true_id, pred_id] += 1
    return matrix


def plot_confusion_matrices(
    models: list[tuple[str, tf.keras.Model]],
    val_ds: tf.data.Dataset,
    class_names: list[str],
    out_dir: Path,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    configure_plot_style(plt)

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / "confusion_matrices.png"
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flat

    for axis, (name, model) in zip(axes, models):
        matrix = confusion_matrix_for_model(model, val_ds, len(class_names))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(name)
        axis.set_xticks(range(len(class_names)))
        axis.set_yticks(range(len(class_names)))
        axis.set_xticklabels(class_names, rotation=30)
        axis.set_yticklabels(class_names)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axis.text(
                    col,
                    row,
                    str(matrix[row, col]),
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontfamily="Arial",
                    color="#222222",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        style_axis(axis)

    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(matrix_path, dpi=180)
    plt.close(fig)
    return matrix_path


def run_hardware_aware_experiment(
    args: argparse.Namespace,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    num_classes: int,
    base_model: tf.keras.Model,
    pruned_model: tf.keras.Model,
    clustered_model: tf.keras.Model,
    results: list[Result],
    models_dir: Path,
    visual_dir: Path,
) -> list[HardwareResult]:
    result_by_name = {result.name: result for result in results}
    base_result = result_by_name["base"]
    base_reference_size_mb = base_result.raw_mb

    hardware_results = [
        hardware_result_from_keras(
            base_result,
            strategy="Dense Keras baseline",
            base_size_mb=base_reference_size_mb,
            params_million=base_model.count_params() / 1_000_000,
            size_mb=base_result.raw_mb,
        ),
        hardware_result_from_keras(
            result_by_name["pruned"],
            strategy="Old unstructured pruning; storage-only sparsity",
            base_size_mb=base_reference_size_mb,
            params_million=pruned_model.count_params() / 1_000_000,
            size_mb=result_by_name["pruned"].packed_mb,
        ),
        hardware_result_from_keras(
            result_by_name["clustered"],
            strategy="Old codebook clustering; storage-only sharing",
            base_size_mb=base_reference_size_mb,
            params_million=clustered_model.count_params() / 1_000_000,
            size_mb=result_by_name["clustered"].packed_mb,
        ),
    ]

    slim_name = f"hw_slim_alpha_{str(args.hardware_alpha).replace('.', '_')}"
    print(f"\nTraining hardware-friendly slim MobileNetV2 alpha={args.hardware_alpha}...")
    slim_model = build_model(
        args.img_size,
        num_classes,
        alpha=args.hardware_alpha,
        name=f"rps_mobilenetv2_alpha_{args.hardware_alpha}",
    )
    fine_tune(slim_model, train_ds, val_ds, args.hardware_epochs)
    set_recovery_trainable(slim_model, args.hardware_unfreeze_last)
    fine_tune(
        slim_model,
        train_ds,
        val_ds,
        args.hardware_fine_tune_epochs,
        learning_rate=args.recovery_learning_rate,
    )
    slim_hw_result, _ = evaluate_slim_keras_variant(
        slim_model,
        slim_name,
        strategy="HW structured channel slimming",
        val_ds=val_ds,
        out_dir=models_dir,
        base_size_mb=base_reference_size_mb,
        profile_warmup=args.profile_warmup,
        profile_rounds=args.profile_rounds,
    )
    hardware_results.append(slim_hw_result)

    if args.skip_tflite:
        print("\nSkipping TFLite conversion because --skip-tflite was set.")
    else:
        print("\nConverting hardware-friendly TFLite INT8 models...")
        for source_name, source_model, source_strategy in [
            ("base_int8", base_model, "HW INT8 quantization"),
            (f"{slim_name}_int8", slim_model, "HW structured slimming + INT8"),
        ]:
            tflite_path = convert_to_full_int8_tflite(
                source_model,
                train_ds,
                models_dir,
                source_name,
                args.tflite_representative_batches,
            )
            if not tflite_path:
                continue
            hardware_results.append(
                evaluate_tflite_variant(
                    tflite_path,
                    name=tflite_path.stem,
                    strategy=source_strategy,
                    source_params_million=source_model.count_params() / 1_000_000,
                    val_ds=val_ds,
                    base_size_mb=base_reference_size_mb,
                    profile_warmup=args.profile_warmup,
                    profile_rounds=args.profile_rounds,
                )
            )

    print_hardware_results(hardware_results)
    print_final_comparison_table(hardware_results)
    hardware_csv_path = write_hardware_results_csv(hardware_results, visual_dir)
    hardware_plot_path = plot_hardware_aware_comparison(hardware_results, visual_dir)
    final_csv_path = write_final_comparison_csv(hardware_results, visual_dir)
    final_table_path = plot_final_comparison_table(hardware_results, visual_dir)
    print(f"Saved hardware-aware CSV: {hardware_csv_path}")
    if hardware_plot_path:
        print(f"Saved hardware-aware comparison: {hardware_plot_path}")
    print(f"Saved final comparison CSV: {final_csv_path}")
    if final_table_path:
        print(f"Saved final comparison table: {final_table_path}")
    return hardware_results


def main() -> None:
    args = parse_args()
    configure_cpu(args.seed)

    workdir = Path(args.workdir)
    models_dir = workdir / "models"
    train_root, val_root = download_and_extract(workdir)

    train_ds = make_dataset(train_root, args.img_size, args.batch_size, args.seed, args.max_train, True)
    val_ds = make_dataset(val_root, args.img_size, args.batch_size, args.seed, args.max_val, False)
    class_names = sorted(path.name for path in train_root.iterdir() if path.is_dir())

    print(f"Classes: {', '.join(class_names)}")
    print("Training baseline MobileNetV2 head...")
    base_model = build_model(args.img_size, len(class_names))
    fine_tune(base_model, train_ds, val_ds, args.epochs)

    if args.clusters < 64:
        print(
            "\nWarning: fewer than 64 clusters is very aggressive for a pretrained CNN. "
            "Expect a large accuracy drop."
        )

    print("\nCreating pruned model...")
    pruned_model, pruning_masks = prune_model(
        base_model,
        args.prune_sparsity,
        prune_depthwise=args.prune_depthwise,
        prune_dense=args.prune_dense,
    )
    print(f"Pruned layers: {len(pruning_masks)}")
    set_recovery_trainable(pruned_model, args.recovery_unfreeze_last)
    fine_tune(
        pruned_model,
        train_ds,
        val_ds,
        args.fine_tune_epochs,
        learning_rate=args.recovery_learning_rate,
        pruning_masks=pruning_masks,
    )

    print("\nCreating clustered model...")
    clustered_model, clustered_layers = cluster_model(
        base_model,
        args.clusters,
        cluster_depthwise=args.cluster_depthwise,
        cluster_dense=args.cluster_dense,
    )
    print(f"Clustered layers: {clustered_layers}")
    set_recovery_trainable(clustered_model, args.cluster_recovery_unfreeze_last)
    fine_tune(
        clustered_model,
        train_ds,
        val_ds,
        args.cluster_fine_tune_epochs,
        learning_rate=args.recovery_learning_rate,
    )
    clustered_model, _ = cluster_model(
        clustered_model,
        args.clusters,
        cluster_depthwise=args.cluster_depthwise,
        cluster_dense=args.cluster_dense,
    )

    print("\nCreating pruned + clustered model...")
    pruned_clustered_model, pruned_clustered_layers = cluster_model(
        pruned_model,
        args.clusters,
        preserve_zeros=True,
        cluster_depthwise=args.cluster_depthwise,
        cluster_dense=args.cluster_dense,
    )
    print(f"Pruned + clustered layers: {pruned_clustered_layers}")

    print("\nEvaluating and saving models...")
    results = [
        evaluate_variant(base_model, "base", val_ds, models_dir, args.profile_warmup, args.profile_rounds),
        evaluate_variant(pruned_model, "pruned", val_ds, models_dir, args.profile_warmup, args.profile_rounds),
        evaluate_variant(clustered_model, "clustered", val_ds, models_dir, args.profile_warmup, args.profile_rounds),
        evaluate_variant(
            pruned_clustered_model,
            "pruned_clustered",
            val_ds,
            models_dir,
            args.profile_warmup,
            args.profile_rounds,
        ),
    ]
    print_results(results)

    visual_dir = workdir / "visualizations"
    csv_path = write_results_csv(results, visual_dir)
    dashboard_path = plot_results_dashboard(results, visual_dir)
    table_path = plot_results_table(results, visual_dir)
    profile_path = plot_inference_profile(results, visual_dir)
    matrix_path = plot_confusion_matrices(
        [
            ("base", base_model),
            ("pruned", pruned_model),
            ("clustered", clustered_model),
            ("pruned_clustered", pruned_clustered_model),
        ],
        val_ds,
        class_names,
        visual_dir,
    )
    if not args.skip_hardware_aware:
        run_hardware_aware_experiment(
            args=args,
            train_ds=train_ds,
            val_ds=val_ds,
            num_classes=len(class_names),
            base_model=base_model,
            pruned_model=pruned_model,
            clustered_model=clustered_model,
            results=results,
            models_dir=models_dir,
            visual_dir=visual_dir,
        )

    print(f"\nSaved models: {models_dir}")
    print(f"Saved CSV: {csv_path}")
    if dashboard_path:
        print(f"Saved dashboard: {dashboard_path}")
    if table_path:
        print(f"Saved table image: {table_path}")
    if profile_path:
        print(f"Saved inference profile: {profile_path}")
    if matrix_path:
        print(f"Saved confusion matrices: {matrix_path}")


if __name__ == "__main__":
    main()
