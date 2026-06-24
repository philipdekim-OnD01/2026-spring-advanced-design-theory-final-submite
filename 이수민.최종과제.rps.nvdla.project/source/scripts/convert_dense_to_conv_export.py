#!/usr/bin/env python3
"""Convert a trained Dense classifier head to an equivalent 1x1 Conv head.

The source model is expected to end with:
AveragePooling2D -> Flatten -> Dropout -> Dense(3)

Because the pooled feature map is 1x1, Dense weights with shape
(channels, classes) can be reshaped to Conv2D weights with shape
(1, 1, channels, classes). This keeps model predictions nearly identical
while removing Flatten/MatMul from the TensorRT graph for DLA experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    build_model,
    evaluate_tflite,
    export_onnx,
    export_tflite,
    json_safe,
    list_images,
    load_images,
    set_reproducibility,
    split_samples,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--source-keras", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--img-size", default=64, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--val-size", default=0.1, type=float)
    parser.add_argument("--calib-samples", default=512, type=int)
    parser.add_argument("--model-name", default="rps_dla_cnn_int8_conv_head")
    return parser.parse_args()


def copy_body_and_convert_head(source: tf.keras.Model, img_size: int) -> tf.keras.Model:
    converted = build_model(img_size)
    source_by_name = {layer.name: layer for layer in source.layers}

    for layer in converted.layers:
        if layer.name == "logits":
            dense = source_by_name["predictions"]
            dense_kernel, dense_bias = dense.get_weights()
            conv_kernel = dense_kernel.reshape((1, 1, dense_kernel.shape[0], dense_kernel.shape[1]))
            layer.set_weights([conv_kernel, dense_bias])
            continue
        if layer.name in source_by_name and layer.weights:
            layer.set_weights(source_by_name[layer.name].get_weights())

    return converted


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


def compare_source_and_converted(
    source: tf.keras.Model,
    converted: tf.keras.Model,
    images: np.ndarray,
) -> dict[str, float]:
    source_logits = source.predict(images, batch_size=64, verbose=0)
    converted_logits = converted.predict(images, batch_size=64, verbose=0)
    source_flat = source_logits.reshape((source_logits.shape[0], -1))
    converted_flat = converted_logits.reshape((converted_logits.shape[0], -1))
    return {
        "max_abs_logit_diff": float(np.max(np.abs(source_flat - converted_flat))),
        "mean_abs_logit_diff": float(np.mean(np.abs(source_flat - converted_flat))),
        "prediction_match_rate": float(
            np.mean(np.argmax(source_flat, axis=-1) == np.argmax(converted_flat, axis=-1))
        ),
    }


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

    train_x, _ = load_images(train_samples, args.img_size)
    test_x, test_y = load_images(test_samples, args.img_size)

    source = tf.keras.models.load_model(args.source_keras)
    converted = copy_body_and_convert_head(source, args.img_size)

    converted_path = args.out_dir / "best_model.keras"
    converted.save(converted_path)
    saved_model_dir = args.out_dir / "saved_model"
    tf.saved_model.save(converted, saved_model_dir)

    onnx_path = args.out_dir / "model_float32.onnx"
    onnx_nchw_path = args.out_dir / "model_float32_nchw.onnx"
    onnx_nchw_b1_path = args.out_dir / "model_float32_nchw_b1.onnx"
    export_onnx(converted, onnx_path, args.img_size)
    export_onnx(converted, onnx_nchw_path, args.img_size, inputs_as_nchw=True)
    export_onnx(
        converted,
        onnx_nchw_b1_path,
        args.img_size,
        inputs_as_nchw=True,
        batch_size=1,
    )
    tflite_paths = export_tflite(converted, args.out_dir, train_x, args.calib_samples)

    keras_eval = evaluate_keras_logits(converted, test_x, test_y)
    int8_eval = evaluate_tflite(tflite_paths["tflite_int8"], test_x, test_y)
    float_eval = evaluate_tflite(tflite_paths["tflite_float32"], test_x, test_y)
    equivalence = compare_source_and_converted(source, converted, test_x)

    metrics = {
        "keras_test_accuracy": keras_eval.accuracy,
        "keras_test_latency": asdict(keras_eval),
        "tflite_float32": asdict(float_eval),
        "tflite_int8": asdict(int8_eval),
        "equivalence": equivalence,
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_test": len(test_samples),
        "source_keras": str(args.source_keras),
        "onnx": str(onnx_path),
        "onnx_nchw": str(onnx_nchw_path),
        "onnx_nchw_b1": str(onnx_nchw_b1_path),
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
        1, keras_eval, converted_path, "Dense head converted to equivalent 1x1 Conv head",
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
