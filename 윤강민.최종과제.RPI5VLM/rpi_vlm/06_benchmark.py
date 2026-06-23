"""Benchmark the final RPS YOLO + VLM pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import config as cfg


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Benchmark RPS pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Validate benchmark setup without models.")
    parser.add_argument("--num-images", type=int, default=100, help="Number of test images.")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up frames to exclude.")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM during benchmark.")
    return parser.parse_args()


def list_test_images(limit: int) -> list[tuple[Path, int]]:
    """YOLO test 이미지와 라벨에서 벤치마크 샘플을 수집한다."""
    image_dir = cfg.RPS_YOLO_DIR / "images" / "test"
    label_dir = cfg.RPS_YOLO_DIR / "labels" / "test"
    samples = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        class_id = 3
        if label_path.exists():
            try:
                class_id = int(label_path.read_text(encoding="utf-8").split()[0])
            except Exception:
                class_id = 3
        samples.append((image_path, class_id))
        if len(samples) >= limit:
            break
    return samples


def model_size_ncnn_mb() -> float | None:
    """NCNN param/bin 파일의 총 크기를 MB로 계산한다."""
    files = [cfg.NCNN_MODEL_DIR / "rps.param", cfg.NCNN_MODEL_DIR / "rps.bin"]
    if not all(p.exists() for p in files):
        return None
    return sum(p.stat().st_size for p in files) / (1024 * 1024)


def ram_usage_mb() -> float | None:
    """현재 프로세스 RAM 사용량을 MB로 반환한다."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def run_benchmark(num_images: int, warmup: int, no_vlm: bool, dry_run: bool) -> dict[str, Any]:
    """오프라인 테스트 이미지로 YOLO-only와 통합 파이프라인 성능을 측정한다."""
    import cv2
    from importlib import import_module

    pipeline_module = import_module("05_pipeline_rpi")
    pipeline = pipeline_module.RPSPipeline(
        cfg.NCNN_MODEL_DIR / "rps.param",
        cfg.NCNN_MODEL_DIR / "rps.bin",
        cfg.VLM_MODEL_DIR / cfg.VLM_MODEL,
        no_vlm=no_vlm,
        dry_run=dry_run,
    )
    samples = list_test_images(num_images)
    if dry_run and not samples:
        dummy = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)
        samples = [(Path("__dummy__"), 3)] * 5
        frames = [dummy.copy() for _ in samples]
    else:
        frames = []
        for path, _ in samples:
            frame = cv2.imread(str(path))
            if frame is not None:
                frames.append(frame)
        samples = samples[: len(frames)]
    if not frames:
        raise FileNotFoundError("No test images found. Run 01_dataset_prepare.py first.")

    yolo_times = []
    pipeline_times = []
    vlm_times = []
    vlm_calls = 0
    correct_rps = 0
    total_rps = 0
    correct_other = 0
    total_other = 0
    fps_timeline = []

    for idx, (frame, (_, true_class)) in enumerate(zip(frames, samples)):
        start = time.perf_counter()
        _ = pipeline.detect(frame)
        yolo_elapsed = time.perf_counter() - start
        start = time.perf_counter()
        result = pipeline.run_frame(frame)
        pipe_elapsed = time.perf_counter() - start
        if idx >= warmup:
            yolo_times.append(yolo_elapsed)
            pipeline_times.append(pipe_elapsed)
            fps_timeline.append(1.0 / max(pipe_elapsed, 1e-6))
            if pipeline.last_vlm_called:
                vlm_calls += 1
                vlm_times.append(pipeline.last_vlm_latency_ms / 1000)
            pred_label = result.get("class", "other")
            pred_id = next((k for k, v in cfg.CLASSES.items() if v == pred_label), 3)
            if true_class in [0, 1, 2]:
                total_rps += 1
                correct_rps += int(pred_id == true_class)
            else:
                total_other += 1
                correct_other += int(pred_id == 3)

    measured = max(1, len(pipeline_times))
    metrics = {
        "yolo_only_fps": 1.0 / float(np.mean(yolo_times)) if yolo_times else None,
        "pipeline_fps": 1.0 / float(np.mean(pipeline_times)) if pipeline_times else None,
        "vlm_call_rate": vlm_calls / measured if measured else None,
        "yolo_latency_ms": float(np.mean(yolo_times) * 1000) if yolo_times else None,
        "vlm_latency_ms": float(np.mean(vlm_times) * 1000) if vlm_times else None,
        "accuracy_rps": correct_rps / total_rps if total_rps else None,
        "accuracy_other": correct_other / total_other if total_other else None,
        "model_size_ncnn_mb": model_size_ncnn_mb(),
        "ram_usage_mb": ram_usage_mb(),
        "fps_timeline": fps_timeline,
    }
    return metrics


def format_value(value: Any, fmt: str = ".1f", default: str = "N/A") -> str:
    """None 값을 N/A로 처리하며 숫자를 지정 포맷으로 변환한다."""
    if value is None:
        return default
    return format(value, fmt)


def write_reports(metrics: dict[str, Any]) -> None:
    """벤치마크 텍스트, JSON, FPS timeline 그래프를 저장한다."""
    report = f"""============================================================
RPi5 RPS Pipeline Benchmark Report
============================================================
YOLO (NCNN INT8, 320x320)
  Latency    : {format_value(metrics.get('yolo_latency_ms'))} ms
  FPS        : {format_value(metrics.get('yolo_only_fps'))}

VLM (Moondream2 Q4 + speculative decoding/fallback)
  Latency    : {format_value(metrics.get('vlm_latency_ms'))} ms when called
  Call Rate  : {format_value((metrics.get('vlm_call_rate') or 0) * 100)}% of total frames

Integrated Pipeline
  Effective FPS : {format_value(metrics.get('pipeline_fps'))}
  Target FPS    : {cfg.TARGET_FPS}
  RAM Usage     : {format_value(metrics.get('ram_usage_mb'))} MB

Accuracy
  mAP50         : N/A
  RPS 3-class   : {format_value((metrics.get('accuracy_rps') or 0) * 100)}%
  Other Reject  : {format_value((metrics.get('accuracy_other') or 0) * 100)}%
============================================================
"""
    (cfg.LOG_DIR / "benchmark_report.txt").write_text(report, encoding="utf-8")
    json_metrics = dict(metrics)
    timeline = json_metrics.pop("fps_timeline", [])
    (cfg.LOG_DIR / "benchmark_report.json").write_text(json.dumps(json_metrics, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 3))
        plt.plot(timeline)
        plt.axhline(cfg.TARGET_FPS, linestyle="--", color="red")
        plt.title("Pipeline FPS timeline")
        plt.xlabel("frame")
        plt.ylabel("FPS")
        plt.tight_layout()
        plt.savefig(cfg.LOG_DIR / "fps_timeline.png")
        plt.close()
    except Exception as exc:
        print(f"[WARN] Could not save FPS plot: {exc}")
    print(report)


def dry_run(args: argparse.Namespace) -> int:
    """드라이런으로 벤치마크 의존성과 제어 흐름을 검증한다."""
    print("[DRY-RUN] validating benchmark")
    for name in ["cv2", "psutil", "matplotlib", "numpy"]:
        try:
            __import__(name)
            print(f"[OK] import {name}")
        except Exception as exc:
            print(f"[WARN] import {name} failed: {exc}")
    metrics = run_benchmark(5, 0, args.no_vlm, dry_run=True)
    write_reports(metrics)
    return 0


def main() -> int:
    """최종 통합 파이프라인 성능을 측정하고 보고서를 생성한다."""
    args = parse_args()
    cfg.ensure_project_dirs()
    if args.dry_run:
        return dry_run(args)
    try:
        metrics = run_benchmark(args.num_images, args.warmup, args.no_vlm, args.dry_run)
        write_reports(metrics)
        return 0
    except Exception as exc:
        print(f"[ERROR] benchmark failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
