#!/usr/bin/env python3
"""Train an RPS classifier and export INT8 deployment artifacts.

The model is intentionally DLA-friendly: Conv2D + BatchNorm + ReLU blocks,
average pooling, and a 1x1 Conv classifier. The RPS lab notebooks use
DenseNet121, but this compact model keeps the NVDLA comparison cleaner by
reducing unsupported-layer fallback noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


CLASS_NAMES = {
    0: "scissors",
    1: "rock",
    2: "paper",
}


@dataclass
class EvalResult:
    accuracy: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_min: float
    latency_ms_max: float
    num_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--img-size", default=64, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--val-size", default=0.1, type=float)
    parser.add_argument("--calib-samples", default=256, type=int)
    parser.add_argument("--model-name", default="rps_dla_cnn_int8")
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    np.random.seed(seed)
    tf.random.set_seed(seed)


def list_images(dataset_dir: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for label in sorted(CLASS_NAMES):
        class_dir = dataset_dir / str(label)
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        files = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not files:
            raise FileNotFoundError(f"No images found in {class_dir}")
        samples.extend((p, label) for p in files)
    return samples


def split_samples(
    samples: list[tuple[Path, int]],
    seed: int,
    test_size: float,
    val_size: float,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = np.random.default_rng(seed)
    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    test: list[tuple[Path, int]] = []
    for label in sorted(CLASS_NAMES):
        class_samples = [s for s in samples if s[1] == label]
        indices = rng.permutation(len(class_samples))
        n_test = max(1, int(round(len(indices) * test_size)))
        remaining = indices[n_test:]
        n_val = max(1, int(round(len(remaining) * val_size)))
        test.extend(class_samples[i] for i in indices[:n_test])
        val.extend(class_samples[i] for i in remaining[:n_val])
        train.extend(class_samples[i] for i in remaining[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def load_images(samples: list[tuple[Path, int]], img_size: int) -> tuple[np.ndarray, np.ndarray]:
    images = np.empty((len(samples), img_size, img_size, 3), dtype=np.float32)
    labels = np.empty((len(samples),), dtype=np.int64)
    for i, (path, label) in enumerate(samples):
        with Image.open(path) as image:
            image = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            images[i] = np.asarray(image, dtype=np.float32) / 255.0
        labels[i] = label
    return images, labels


def write_manifest(path: Path, dataset_dir: Path, splits: dict[str, list[tuple[Path, int]]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "path", "label", "class_name"])
        writer.writeheader()
        for split, samples in splits.items():
            for image_path, label in samples:
                writer.writerow({
                    "split": split,
                    "path": image_path.relative_to(dataset_dir).as_posix(),
                    "label": label,
                    "class_name": CLASS_NAMES[label],
                })


def make_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        augmenter = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.08, seed=seed + 1),
            tf.keras.layers.RandomTranslation(0.08, 0.08, seed=seed + 2),
            tf.keras.layers.RandomZoom(0.12, seed=seed + 3),
            tf.keras.layers.RandomContrast(0.15, seed=seed + 4),
        ])

        def augment(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            return augmenter(x, training=True), y

        ds = ds.shuffle(len(images), seed=seed, reshuffle_each_iteration=True).map(
            augment, num_parallel_calls=tf.data.AUTOTUNE
        )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def conv_block(x: tf.Tensor, filters: int, stride: int, name: str) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    return tf.keras.layers.ReLU(name=f"{name}_relu")(x)


def build_model(img_size: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="input")
    x = conv_block(inputs, 24, 2, "stem")
    x = conv_block(x, 32, 1, "block1a")
    x = conv_block(x, 32, 2, "block1b")
    x = conv_block(x, 48, 1, "block2a")
    x = conv_block(x, 48, 2, "block2b")
    x = conv_block(x, 64, 1, "block3a")
    x = conv_block(x, 96, 2, "block3b")
    x = conv_block(x, 128, 1, "block4a")
    x = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), name="avg_pool")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Conv2D(3, 1, activation=None, name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="rps_dla_friendly_cnn")


def representative_dataset(train_images: np.ndarray, limit: int):
    count = min(limit, len(train_images))
    for i in range(count):
        yield [train_images[i : i + 1].astype(np.float32)]


def export_tflite(model: tf.keras.Model, out_dir: Path, train_images: np.ndarray, calib_samples: int) -> dict[str, Path]:
    float_path = out_dir / "model_float32.tflite"
    int8_path = out_dir / "model_int8_full_integer.tflite"

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_path.write_bytes(converter.convert())

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(train_images, calib_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_path.write_bytes(converter.convert())

    return {"tflite_float32": float_path, "tflite_int8": int8_path}


def export_onnx(
    model: tf.keras.Model,
    out_path: Path,
    img_size: int,
    inputs_as_nchw: bool = False,
    batch_size: int | None = None,
) -> None:
    import tf2onnx

    spec = (tf.TensorSpec((batch_size, img_size, img_size, 3), tf.float32, name="input"),)
    kwargs = {"inputs_as_nchw": ["input"]} if inputs_as_nchw else {}
    tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=str(out_path),
        **kwargs,
    )


def quantize_for_tflite(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    dtype = detail["dtype"]
    if scale == 0:
        return values.astype(dtype)
    q = np.round(values / scale + zero_point)
    info = np.iinfo(dtype)
    return np.clip(q, info.min, info.max).astype(dtype)


def dequantize_from_tflite(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    if scale == 0:
        return values.astype(np.float32)
    return (values.astype(np.float32) - zero_point) * scale


def evaluate_tflite(model_path: Path, images: np.ndarray, labels: np.ndarray) -> EvalResult:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    predictions: list[int] = []
    latencies: list[float] = []
    for image in images:
        sample = image[None, ...].astype(np.float32)
        if np.issubdtype(input_detail["dtype"], np.integer):
            sample = quantize_for_tflite(sample, input_detail)
        start = time.perf_counter_ns()
        interpreter.set_tensor(input_detail["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        end = time.perf_counter_ns()
        if np.issubdtype(output_detail["dtype"], np.integer):
            output = dequantize_from_tflite(output, output_detail)
        predictions.append(int(np.argmax(output.reshape(output.shape[0], -1), axis=-1)[0]))
        latencies.append((end - start) / 1_000_000.0)
    pred = np.asarray(predictions, dtype=np.int64)
    latency = np.asarray(latencies, dtype=np.float64)
    return EvalResult(
        accuracy=float(np.mean(pred == labels)),
        latency_ms_mean=float(np.mean(latency)),
        latency_ms_p50=float(np.percentile(latency, 50)),
        latency_ms_p95=float(np.percentile(latency, 95)),
        latency_ms_min=float(np.min(latency)),
        latency_ms_max=float(np.max(latency)),
        num_samples=int(len(labels)),
    )


def append_metric_row(
    csv_path: Path,
    model_name: str,
    backend: str,
    precision: str,
    device: str,
    batch_size: int,
    result: EvalResult,
    artifact: Path,
    notes: str,
) -> None:
    fieldnames = [
        "timestamp", "host", "model", "backend", "precision", "device",
        "batch_size", "num_samples", "accuracy", "latency_ms_mean",
        "latency_ms_p50", "latency_ms_p95", "latency_ms_min",
        "latency_ms_max", "throughput_fps", "artifact", "notes",
    ]
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": socket.gethostname(),
            "model": model_name,
            "backend": backend,
            "precision": precision,
            "device": device,
            "batch_size": batch_size,
            "num_samples": result.num_samples,
            "accuracy": result.accuracy,
            "latency_ms_mean": result.latency_ms_mean,
            "latency_ms_p50": result.latency_ms_p50,
            "latency_ms_p95": result.latency_ms_p95,
            "latency_ms_min": result.latency_ms_min,
            "latency_ms_max": result.latency_ms_max,
            "throughput_fps": 1000.0 / result.latency_ms_mean if result.latency_ms_mean else 0.0,
            "artifact": str(artifact),
            "notes": notes,
        })


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples = list_images(args.dataset_dir)
    train_samples, val_samples, test_samples = split_samples(
        samples, args.seed, args.test_size, args.val_size
    )
    write_manifest(
        args.out_dir / "manifest.csv",
        args.dataset_dir,
        {"train": train_samples, "val": val_samples, "test": test_samples},
    )
    (args.out_dir / "labels.json").write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")

    train_x, train_y = load_images(train_samples, args.img_size)
    val_x, val_y = load_images(val_samples, args.img_size)
    test_x, test_y = load_images(test_samples, args.img_size)

    train_y_model = train_y.reshape((-1, 1, 1))
    val_y_model = val_y.reshape((-1, 1, 1))
    test_y_model = test_y.reshape((-1, 1, 1))

    train_ds = make_dataset(train_x, train_y_model, args.batch_size, True, args.seed)
    val_ds = make_dataset(val_x, val_y_model, args.batch_size, False, args.seed)

    model = build_model(args.img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    checkpoint_path = args.out_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_accuracy", save_best_only=True, mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    best_model = tf.keras.models.load_model(checkpoint_path)
    keras_loss, keras_acc = best_model.evaluate(test_x, test_y_model, batch_size=args.batch_size, verbose=0)
    saved_model_dir = args.out_dir / "saved_model"
    tf.saved_model.save(best_model, saved_model_dir)

    onnx_path = args.out_dir / "model_float32.onnx"
    export_onnx(best_model, onnx_path, args.img_size)
    tflite_paths = export_tflite(best_model, args.out_dir, train_x, args.calib_samples)
    int8_eval = evaluate_tflite(tflite_paths["tflite_int8"], test_x, test_y)
    float_eval = evaluate_tflite(tflite_paths["tflite_float32"], test_x, test_y)

    metrics = {
        "keras_test_loss": float(keras_loss),
        "keras_test_accuracy": float(keras_acc),
        "tflite_float32": asdict(float_eval),
        "tflite_int8": asdict(int8_eval),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_test": len(test_samples),
        "history": history.history,
        "env": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "platform": platform.platform(),
            "host": socket.gethostname(),
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2), encoding="utf-8")

    summary_csv = args.out_dir / "metrics_summary.csv"
    append_metric_row(
        summary_csv, args.model_name, "keras", "fp32", "macallan",
        args.batch_size, EvalResult(float(keras_acc), 0, 0, 0, 0, 0, len(test_y)),
        checkpoint_path, "Keras test accuracy; latency not measured",
    )
    append_metric_row(
        summary_csv, args.model_name, "tflite", "fp32", "macallan_cpu",
        1, float_eval, tflite_paths["tflite_float32"], "TFLite interpreter on Macallan CPU",
    )
    append_metric_row(
        summary_csv, args.model_name, "tflite", "int8", "macallan_cpu",
        1, int8_eval, tflite_paths["tflite_int8"], "Full integer INT8 PTQ TFLite",
    )
    print(json.dumps(json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
