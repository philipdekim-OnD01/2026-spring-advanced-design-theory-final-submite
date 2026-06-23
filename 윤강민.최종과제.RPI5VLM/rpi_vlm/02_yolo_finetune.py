"""Fine-tune YOLOv8n on the prepared RPS YOLO dataset."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import config as cfg


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n for RPS detection.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training.")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cuda:0 or cpu.")
    return parser.parse_args()


def print_cuda_report() -> str:
    """CUDA 가시성과 사용할 논리 디바이스 정보를 출력한다."""
    try:
        import torch

        print("CUDA available:", torch.cuda.is_available())
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
        if torch.cuda.is_available():
            print("Visible CUDA device count:", torch.cuda.device_count())
            print("Using logical CUDA device:", torch.cuda.current_device())
            print("Device name:", torch.cuda.get_device_name(0))
            return "cuda:0"
    except Exception as exc:
        print(f"[WARN] torch import/CUDA report failed: {exc}")
    return "cpu"


def resolve_device(cli_device: str | None) -> str:
    """CLI 인자와 CUDA 상태를 바탕으로 학습 디바이스를 결정한다."""
    default_device = print_cuda_report()
    return cli_device or default_device


def validate_dataset() -> Path:
    """YOLO data.yaml 경로가 존재하는지 확인한다."""
    data_yaml = cfg.RPS_YOLO_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing YOLO dataset yaml: {data_yaml}. Run 01_dataset_prepare.py first.")
    return data_yaml


def copy_training_artifacts(results, output_model: Path) -> None:
    """Ultralytics 결과 디렉터리에서 최종 모델과 학습 로그를 표준 위치로 복사한다."""
    save_dir = Path(getattr(results, "save_dir", cfg.LOG_DIR))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    source_model = best if best.exists() else last
    if not source_model.exists():
        raise FileNotFoundError(f"No trained model found under {save_dir / 'weights'}")
    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model, output_model)
    csv_path = save_dir / "results.csv"
    if csv_path.exists():
        shutil.copy2(csv_path, cfg.LOG_DIR / "train_log.csv")
    png_path = save_dir / "results.png"
    if png_path.exists():
        shutil.copy2(png_path, cfg.LOG_DIR / "train_curve.png")
    print(f"[MODEL] saved {output_model}")


def print_validation_metrics(metrics) -> None:
    """검증 결과에서 mAP, precision, recall을 출력한다."""
    box = getattr(metrics, "box", None)
    print("mAP50:", getattr(box, "map50", "N/A"))
    print("mAP50-95:", getattr(box, "map", "N/A"))
    print("Precision:", getattr(box, "mp", "N/A"))
    print("Recall:", getattr(box, "mr", "N/A"))


def dry_run(args: argparse.Namespace) -> int:
    """드라이런으로 학습 환경과 경로를 검증한다."""
    print("[DRY-RUN] validating YOLO fine-tune setup")
    device = resolve_device(args.device)
    print(f"[DRY-RUN] selected device: {device}")
    try:
        import ultralytics  # noqa: F401

        print("[OK] import ultralytics")
    except Exception as exc:
        print(f"[WARN] import ultralytics failed: {exc}")
    try:
        data_yaml = validate_dataset()
        print(f"[OK] dataset yaml: {data_yaml}")
    except Exception as exc:
        print(f"[WARN] {exc}")
    print(f"[DRY-RUN] baseline output: {cfg.YOLO_MODEL_DIR / 'rps_baseline.pt'}")
    return 0


def main() -> int:
    """YOLOv8n 모델을 RPS 데이터셋으로 파인튜닝한다."""
    args = parse_args()
    cfg.ensure_project_dirs()
    if args.dry_run:
        return dry_run(args)
    try:
        from ultralytics import YOLO

        data_yaml = validate_dataset()
        device = resolve_device(args.device)
        print(f"[TRAIN] device={device}, data={data_yaml}")
        model = YOLO(cfg.YOLO_PRETRAIN)
        results = model.train(
            data=str(data_yaml),
            imgsz=cfg.IMG_SIZE,
            epochs=cfg.EPOCHS,
            batch=cfg.BATCH_SIZE,
            lr0=cfg.LR0,
            lrf=cfg.LRF,
            warmup_epochs=cfg.WARMUP_EPOCHS,
            augment=True,
            patience=cfg.EARLY_STOP_PATIENCE,
            device=device,
            project=str(cfg.LOG_DIR / "ultralytics"),
            name="rps_yolov8n",
            exist_ok=True,
        )
        output_model = cfg.YOLO_MODEL_DIR / "rps_baseline.pt"
        copy_training_artifacts(results, output_model)
        print("[VAL] running validation")
        val_metrics = YOLO(str(output_model)).val(data=str(data_yaml), imgsz=cfg.IMG_SIZE, conf=cfg.CONF_LOW, device=device)
        print_validation_metrics(val_metrics)
        return 0
    except Exception as exc:
        print(f"[ERROR] YOLO fine-tuning failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
