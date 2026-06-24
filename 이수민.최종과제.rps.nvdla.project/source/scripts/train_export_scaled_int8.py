#!/usr/bin/env python3
"""Train width-scaled RPS CNNs and export INT8/DLA artifacts.

Training uses a Dense classifier head because it converges reliably on this
dataset. Export converts that Dense head to an equivalent 1x1 Conv head so the
TensorRT graph remains DLA-friendly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tensorflow as tf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_export_int8 import (  # noqa: E402
    CLASS_NAMES,
    EvalResult,
    append_metric_row,
    evaluate_tflite,
    export_onnx,
    export_tflite,
    json_safe,
    list_images,
    load_images,
    make_dataset,
    set_reproducibility,
    split_samples,
    write_manifest,
)


BASE_FILTERS = (24, 32, 32, 48, 48, 64, 96, 128)
BLOCKS = (
    ("stem", 2),
    ("block1a", 1),
    ("block1b", 2),
    ("block2a", 1),
    ("block2b", 2),
    ("block3a", 1),
    ("block3b", 2),
    ("block4a", 1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--img-size", default=64, type=int)
    parser.add_argument("--width-mult", default=1.0, type=float)
    parser.add_argument("--channel-alignment", default=16, type=int)
    parser.add_argument("--channel-rounding", choices=["nearest", "ceil"], default="nearest")
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--early-stop-patience", default=12, type=int)
    parser.add_argument("--early-stop-start-epoch", default=15, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--val-size", default=0.1, type=float)
    parser.add_argument("--calib-samples", default=512, type=int)
    parser.add_argument("--model-name", default="rps_scaled_int8")
    return parser.parse_args()


def scaled_filters(width_mult: float, alignment: int, rounding: str) -> list[int]:
    if alignment <= 0:
        raise ValueError("--channel-alignment must be positive")
    filters: list[int] = []
    for value in BASE_FILTERS:
        units = (value * width_mult) / alignment
        aligned_units = math.ceil(units) if rounding == "ceil" else round(units)
        scaled = max(alignment, int(aligned_units * alignment))
        filters.append(scaled)
    return filters


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


def build_body(
    img_size: int,
    width_mult: float,
    channel_alignment: int,
    channel_rounding: str,
) -> tuple[tf.keras.Input, tf.Tensor, list[int]]:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="input")
    filters = scaled_filters(width_mult, channel_alignment, channel_rounding)
    x: tf.Tensor = inputs
    for (name, stride), channels in zip(BLOCKS, filters):
        x = conv_block(x, channels, stride, name)
    pool = img_size // 16
    if pool < 1 or img_size % 16 != 0:
        raise ValueError("--img-size must be a positive multiple of 16")
    x = tf.keras.layers.AveragePooling2D(pool_size=(pool, pool), name="avg_pool")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    return inputs, x, filters


def build_dense_model(
    img_size: int,
    width_mult: float,
    channel_alignment: int,
    channel_rounding: str,
) -> tuple[tf.keras.Model, list[int]]:
    inputs, x, filters = build_body(
        img_size, width_mult, channel_alignment, channel_rounding
    )
    x = tf.keras.layers.Flatten(name="flatten")(x)
    outputs = tf.keras.layers.Dense(3, activation=None, name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="rps_scaled_dense_train")
    return model, filters


def build_conv_model(
    img_size: int,
    width_mult: float,
    channel_alignment: int,
    channel_rounding: str,
) -> tf.keras.Model:
    inputs, x, _ = build_body(
        img_size, width_mult, channel_alignment, channel_rounding
    )
    outputs = tf.keras.layers.Conv2D(3, 1, activation=None, name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="rps_scaled_conv_export")


def convert_dense_to_conv(
    dense_model: tf.keras.Model,
    img_size: int,
    width_mult: float,
    channel_alignment: int,
    channel_rounding: str,
) -> tf.keras.Model:
    conv_model = build_conv_model(
        img_size, width_mult, channel_alignment, channel_rounding
    )
    dense_by_name = {layer.name: layer for layer in dense_model.layers}
    for layer in conv_model.layers:
        if layer.name == "logits":
            dense = dense_by_name["predictions"]
            dense_kernel, dense_bias = dense.get_weights()
            conv_kernel = dense_kernel.reshape((1, 1, dense_kernel.shape[0], dense_kernel.shape[1]))
            layer.set_weights([conv_kernel, dense_bias])
            continue
        if layer.name in dense_by_name and layer.weights:
            layer.set_weights(dense_by_name[layer.name].get_weights())
    return conv_model


def evaluate_keras_logits(model: tf.keras.Model, images: np.ndarray, labels: np.ndarray) -> EvalResult:
    latencies: list[float] = []
    predictions: list[int] = []
    for image in images:
        sample = image[None, ...].astype(np.float32)
        start = time.perf_counter_ns()
        output = model(sample, training=False).numpy()
        end = time.perf_counter_ns()
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


def compare_dense_and_conv(dense_model: tf.keras.Model, conv_model: tf.keras.Model, images: np.ndarray) -> dict[str, float]:
    dense_logits = dense_model.predict(images, batch_size=64, verbose=0)
    conv_logits = conv_model.predict(images, batch_size=64, verbose=0)
    dense_flat = dense_logits.reshape((dense_logits.shape[0], -1))
    conv_flat = conv_logits.reshape((conv_logits.shape[0], -1))
    return {
        "max_abs_logit_diff": float(np.max(np.abs(dense_flat - conv_flat))),
        "mean_abs_logit_diff": float(np.mean(np.abs(dense_flat - conv_flat))),
        "prediction_match_rate": float(
            np.mean(np.argmax(dense_flat, axis=-1) == np.argmax(conv_flat, axis=-1))
        ),
    }


def write_model_info(
    path: Path,
    model: tf.keras.Model,
    filters: list[int],
    width_mult: float,
    img_size: int,
    channel_alignment: int,
    channel_rounding: str,
) -> None:
    info = {
        "width_mult": width_mult,
        "img_size": img_size,
        "filters": filters,
        "param_count": int(model.count_params()),
        "channel_alignment": channel_alignment,
        "channel_rounding": channel_rounding,
    }
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")


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

    train_ds = make_dataset(train_x, train_y, args.batch_size, True, args.seed)
    val_ds = make_dataset(val_x, val_y, args.batch_size, False, args.seed)

    dense_model, filters = build_dense_model(
        args.img_size,
        args.width_mult,
        args.channel_alignment,
        args.channel_rounding,
    )
    dense_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    checkpoint_path = args.out_dir / "best_dense_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_accuracy", save_best_only=True, mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=args.early_stop_patience,
            mode="max",
            restore_best_weights=True,
            start_from_epoch=args.early_stop_start_epoch,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]
    history = dense_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    best_dense = tf.keras.models.load_model(checkpoint_path)
    keras_loss, keras_acc = best_dense.evaluate(test_x, test_y, batch_size=args.batch_size, verbose=0)
    conv_model = convert_dense_to_conv(
        best_dense,
        args.img_size,
        args.width_mult,
        args.channel_alignment,
        args.channel_rounding,
    )
    conv_path = args.out_dir / "best_model.keras"
    conv_model.save(conv_path)
    tf.saved_model.save(conv_model, args.out_dir / "saved_model")
    write_model_info(
        args.out_dir / "model_info.json",
        conv_model,
        filters,
        args.width_mult,
        args.img_size,
        args.channel_alignment,
        args.channel_rounding,
    )

    export_onnx(conv_model, args.out_dir / "model_float32.onnx", args.img_size)
    export_onnx(conv_model, args.out_dir / "model_float32_nchw.onnx", args.img_size, inputs_as_nchw=True)
    export_onnx(
        conv_model,
        args.out_dir / "model_float32_nchw_b1.onnx",
        args.img_size,
        inputs_as_nchw=True,
        batch_size=1,
    )
    tflite_paths = export_tflite(conv_model, args.out_dir, train_x, args.calib_samples)

    conv_eval = evaluate_keras_logits(conv_model, test_x, test_y)
    int8_eval = evaluate_tflite(tflite_paths["tflite_int8"], test_x, test_y)
    float_eval = evaluate_tflite(tflite_paths["tflite_float32"], test_x, test_y)
    equivalence = compare_dense_and_conv(best_dense, conv_model, test_x)

    metrics = {
        "keras_dense_test_loss": float(keras_loss),
        "keras_dense_test_accuracy": float(keras_acc),
        "keras_conv_test": asdict(conv_eval),
        "tflite_float32": asdict(float_eval),
        "tflite_int8": asdict(int8_eval),
        "equivalence": equivalence,
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_test": len(test_samples),
        "width_mult": args.width_mult,
        "img_size": args.img_size,
        "filters": filters,
        "param_count": int(conv_model.count_params()),
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
        checkpoint_path, "Dense-head training model; latency not measured",
    )
    append_metric_row(
        summary_csv, args.model_name, "tflite", "fp32", "macallan_cpu",
        1, float_eval, tflite_paths["tflite_float32"], "Conv-head TFLite interpreter on Macallan CPU",
    )
    append_metric_row(
        summary_csv, args.model_name, "tflite", "int8", "macallan_cpu",
        1, int8_eval, tflite_paths["tflite_int8"], "Conv-head full integer INT8 PTQ TFLite",
    )
    print(json.dumps(json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
