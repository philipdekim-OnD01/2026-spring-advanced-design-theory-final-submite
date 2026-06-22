"""
Fast hardware-aware comparison from already-saved experiment artifacts.

This script does NOT retrain MobileNetV2. It reuses:
  - C:\PythonProject\Cluster_and_Prune\visualizations\compression_results.csv
  - C:\PythonProject\Cluster_and_Prune\models\*.keras
  - C:\PythonProject\Cluster_and_Prune\models\*.tflite

Outputs:
  - visualizations\final_comparison_table_fast.csv
  - visualizations\final_comparison_table_fast.png

Run:
  python C:\PythonProject\Hardware_Aware_Compare.py
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


@dataclass
class FinalRow:
    model_strategy: str
    accuracy_percent: float
    model_size_mb: float
    latency_p50_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast final comparison table without retraining.")
    parser.add_argument("--root", default=r"C:\PythonProject\Cluster_and_Prune")
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-val", type=int, default=360)
    parser.add_argument("--profile-rounds", type=int, default=40)
    parser.add_argument("--profile-warmup", type=int, default=8)
    parser.add_argument("--hardware-alpha-label", default="0_5")
    parser.add_argument("--skip-keras-profile", action="store_true")
    parser.add_argument("--skip-tflite-profile", action="store_true")
    return parser.parse_args()


def make_val_dataset(root: Path, img_size: int, batch_size: int, max_images: int) -> tf.data.Dataset:
    val_root = root / "data" / "val" / "rps-test-set"
    if not val_root.exists():
        raise FileNotFoundError(f"Validation dataset not found: {val_root}")

    ds = tf.keras.utils.image_dataset_from_directory(
        val_root,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )
    if max_images > 0:
        max_batches = max(1, (max_images + batch_size - 1) // batch_size)
        ds = ds.take(max_batches)
    return ds.prefetch(tf.data.AUTOTUNE)


def timed_keras_passes(model: tf.keras.Model, images: tf.Tensor, warmup: int, rounds: int) -> list[float]:
    for _ in range(warmup):
        model(images, training=False).numpy()

    timings = []
    for _ in range(rounds):
        start = time.perf_counter()
        model(images, training=False).numpy()
        timings.append((time.perf_counter() - start) * 1000)
    return timings


def evaluate_keras_model(
    model_path: Path,
    val_ds: tf.data.Dataset,
    warmup: int,
    rounds: int,
) -> tuple[float, float]:
    model = tf.keras.models.load_model(model_path, compile=False)

    correct = 0
    total = 0
    for images, labels in val_ds:
        predictions = model(images, training=False).numpy()
        correct += int(np.sum(np.argmax(predictions, axis=1) == np.argmax(labels.numpy(), axis=1)))
        total += int(labels.shape[0])

    images = next(iter(val_ds))[0][:1]
    timings = timed_keras_passes(model, images, warmup, rounds)
    return (correct / total * 100 if total else 0.0), float(np.percentile(timings, 50))


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


def load_tflite_interpreter(tflite_path: Path):
    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_path),
        num_threads=max(1, min(4, os.cpu_count() or 1)),
    )
    interpreter.allocate_tensors()
    return interpreter


def timed_tflite_passes(interpreter, images: np.ndarray, warmup: int, rounds: int) -> list[float]:
    for _ in range(warmup):
        invoke_tflite(interpreter, images)

    timings = []
    for _ in range(rounds):
        start = time.perf_counter()
        invoke_tflite(interpreter, images)
        timings.append((time.perf_counter() - start) * 1000)
    return timings


def evaluate_tflite_model(
    tflite_path: Path,
    val_ds: tf.data.Dataset,
    warmup: int,
    rounds: int,
) -> tuple[float, float]:
    accuracy_interpreter = load_tflite_interpreter(tflite_path)
    correct = 0
    total = 0
    for images, labels in val_ds:
        predictions = invoke_tflite(accuracy_interpreter, images.numpy().astype(np.float32))
        correct += int(np.sum(np.argmax(predictions, axis=1) == np.argmax(labels.numpy(), axis=1)))
        total += int(labels.shape[0])

    profile_interpreter = load_tflite_interpreter(tflite_path)
    images = next(iter(val_ds))[0][:1].numpy().astype(np.float32)
    timings = timed_tflite_passes(profile_interpreter, images, warmup, rounds)
    return (correct / total * 100 if total else 0.0), float(np.percentile(timings, 50))


def read_existing_compression_rows(csv_path: Path) -> list[FinalRow]:
    rows = []
    if not csv_path.exists():
        return rows

    strategy_names = {
        "base": "Dense Keras baseline",
        "pruned": "Old unstructured pruning",
        "clustered": "Old codebook clustering",
        "pruned_clustered": "Old pruning + clustering",
    }
    with csv_path.open(newline="", encoding="utf-8") as file:
        for item in csv.DictReader(file):
            model = item["model"]
            rows.append(
                FinalRow(
                    model_strategy=f"{model} ({strategy_names.get(model, 'Existing strategy')})",
                    accuracy_percent=float(item["accuracy_percent"]),
                    model_size_mb=float(item["packed_size_mb"]),
                    latency_p50_ms=float(item["single_latency_p50_ms"]),
                )
            )
    return rows


def write_final_csv(rows: list[FinalRow], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "final_comparison_table_fast.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model_strategy", "accuracy_percent", "model_size_mb", "latency_p50_ms"])
        for row in rows:
            writer.writerow(
                [
                    row.model_strategy,
                    round(row.accuracy_percent, 4),
                    round(row.model_size_mb, 4),
                    round(row.latency_p50_ms, 4),
                ]
            )
    return csv_path


def plot_final_table(rows: list[FinalRow], out_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "#222222",
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "final_comparison_table_fast.png"
    headers = ["Model (Strategy)", "Accuracy", "Model Size", "Latency"]
    cell_text = [
        [
            row.model_strategy,
            f"{row.accuracy_percent:.2f}%",
            f"{row.model_size_mb:.2f} MB",
            f"{row.latency_p50_ms:.1f} ms",
        ]
        for row in rows
    ]

    fig_height = max(3.0, 1.0 + 0.42 * len(rows))
    fig, axis = plt.subplots(figsize=(15, fig_height))
    axis.axis("off")
    table = axis.table(
        cellText=cell_text,
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
            cell.set_facecolor("#0B3D91")
            cell.set_text_props(color="white", weight="bold", fontfamily="Arial")
        else:
            cell.set_facecolor("#F5F8FC" if row % 2 else "#E8EEF7")
            cell.set_text_props(color="#222222", fontfamily="Arial")
            if col == 0:
                cell.set_text_props(weight="bold", fontfamily="Arial")
                cell.get_text().set_ha("left")

    axis.set_title("Final Strategy Comparison", fontsize=15, fontweight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return png_path


def short_label(model_strategy: str) -> str:
    if model_strategy.startswith("base ("):
        return "Baseline"
    if model_strategy.startswith("pruned ("):
        return "Pruning"
    if model_strategy.startswith("clustered ("):
        return "Clustering"
    if model_strategy.startswith("pruned_clustered ("):
        return "Prune+Cluster"
    if model_strategy.startswith("hw_slim"):
        return "Slim"
    if model_strategy.startswith("base_int8"):
        return "INT8"
    if "slim" in model_strategy and "int8" in model_strategy:
        return "Slim+INT8"
    return model_strategy.split(" (", 1)[0]


def plot_baseline_comparison(rows: list[FinalRow], out_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not rows:
        return None

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4A5568",
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "grid.color": "#D8E2F0",
        }
    )

    baseline = next((row for row in rows if row.model_strategy.startswith("base (")), rows[0])
    labels = [short_label(row.model_strategy) for row in rows]
    accuracy_delta = [row.accuracy_percent - baseline.accuracy_percent for row in rows]
    size_reduction = [100 * (1 - row.model_size_mb / baseline.model_size_mb) for row in rows]
    speedup = [baseline.latency_p50_ms / row.latency_p50_ms if row.latency_p50_ms > 0 else 0 for row in rows]
    latency = [row.latency_p50_ms for row in rows]

    colors = ["#0B3D91", "#1D5FA7", "#3F7FBF", "#7FA6D8", "#222222", "#555555", "#888888", "#C7C7C7"]
    bar_colors = (colors * ((len(rows) // len(colors)) + 1))[: len(rows)]

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "hardware_baseline_comparison_fast.png"
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Comparison vs Baseline: Slim and INT8 Strategies", fontsize=16, fontweight="bold")

    axes[0, 0].bar(labels, accuracy_delta, color=bar_colors)
    axes[0, 0].axhline(0, color="#222222", linewidth=0.9)
    axes[0, 0].set_title("Accuracy Change vs Baseline")
    axes[0, 0].set_ylabel("Percentage points")
    for idx, value in enumerate(accuracy_delta):
        va = "bottom" if value >= 0 else "top"
        axes[0, 0].text(idx, value, f"{value:+.1f}", ha="center", va=va, fontsize=8)

    axes[0, 1].bar(labels, size_reduction, color=bar_colors)
    axes[0, 1].axhline(0, color="#222222", linewidth=0.9)
    axes[0, 1].set_title("Model Size Reduction vs Baseline")
    axes[0, 1].set_ylabel("Reduction (%)")
    for idx, value in enumerate(size_reduction):
        axes[0, 1].text(idx, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)

    axes[1, 0].bar(labels, speedup, color=bar_colors)
    axes[1, 0].axhline(1.0, color="#222222", linewidth=0.9)
    axes[1, 0].set_title("Latency Speed-up vs Baseline")
    axes[1, 0].set_ylabel("Speed-up (x)")
    for idx, value in enumerate(speedup):
        axes[1, 0].text(idx, value, f"{value:.2f}x", ha="center", va="bottom", fontsize=8)

    axes[1, 1].scatter(latency, [row.accuracy_percent for row in rows], s=[max(90, item * 4) for item in size_reduction], c=bar_colors)
    for idx, label in enumerate(labels):
        axes[1, 1].annotate(label, (latency[idx], rows[idx].accuracy_percent), xytext=(6, 6), textcoords="offset points")
    axes[1, 1].set_title("Accuracy vs Latency")
    axes[1, 1].set_xlabel("P50 latency (ms)")
    axes[1, 1].set_ylabel("Accuracy (%)")

    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=20)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.6, linewidth=0.8)

    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return png_path


def print_rows(rows: list[FinalRow]) -> None:
    headers = ["Model (Strategy)", "Accuracy", "Model Size", "Latency"]
    table_rows = [
        [
            row.model_strategy,
            f"{row.accuracy_percent:.2f}%",
            f"{row.model_size_mb:.2f} MB",
            f"{row.latency_p50_ms:.1f} ms",
        ]
        for row in rows
    ]
    widths = [max(len(str(row[idx])) for row in [headers] + table_rows) for idx in range(len(headers))]
    print("\nFinal comparison")
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(" | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers))))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    models_dir = root / "models"
    visual_dir = root / "visualizations"

    rows = read_existing_compression_rows(visual_dir / "compression_results.csv")
    val_ds = make_val_dataset(root, args.img_size, args.batch_size, args.max_val)

    if not args.skip_keras_profile:
        slim_path = models_dir / f"hw_slim_alpha_{args.hardware_alpha_label}.keras"
        if slim_path.exists():
            accuracy, latency = evaluate_keras_model(
                slim_path,
                val_ds,
                args.profile_warmup,
                args.profile_rounds,
            )
            rows.append(
                FinalRow(
                    model_strategy=f"{slim_path.stem} (HW structured channel slimming)",
                    accuracy_percent=accuracy,
                    model_size_mb=slim_path.stat().st_size / (1024 * 1024),
                    latency_p50_ms=latency,
                )
            )
        else:
            print(f"Missing slim Keras model, skipped: {slim_path}")

    if not args.skip_tflite_profile:
        for tflite_path in sorted(models_dir.glob("*.tflite")):
            strategy = "HW INT8 quantization" if "base" in tflite_path.stem else "HW structured slimming + INT8"
            accuracy, latency = evaluate_tflite_model(
                tflite_path,
                val_ds,
                args.profile_warmup,
                args.profile_rounds,
            )
            rows.append(
                FinalRow(
                    model_strategy=f"{tflite_path.stem} ({strategy})",
                    accuracy_percent=accuracy,
                    model_size_mb=tflite_path.stat().st_size / (1024 * 1024),
                    latency_p50_ms=latency,
                )
            )

    print_rows(rows)
    csv_path = write_final_csv(rows, visual_dir)
    png_path = plot_final_table(rows, visual_dir)
    graph_path = plot_baseline_comparison(rows, visual_dir)
    print(f"\nSaved CSV: {csv_path}")
    if png_path:
        print(f"Saved table image: {png_path}")
    if graph_path:
        print(f"Saved baseline comparison graph: {graph_path}")


if __name__ == "__main__":
    main()
