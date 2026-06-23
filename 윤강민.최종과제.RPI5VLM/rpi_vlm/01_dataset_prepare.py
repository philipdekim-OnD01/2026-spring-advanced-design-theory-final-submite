"""Convert the local RPS classification dataset into YOLO detection format."""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image

import config as cfg


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Prepare RPS YOLO dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without writing the full dataset.")
    parser.add_argument("--source", type=Path, default=cfg.RAW_DATASET_DIR, help="Local RPS dataset root.")
    parser.add_argument("--roboflow-dir", type=Path, default=None, help="Future Roboflow dataset input root.")
    parser.add_argument("--negative-dir", type=Path, default=None, help="Optional non-RPS images for class 3.")
    parser.add_argument("--augment", action="store_true", help="Save one augmented copy per training image.")
    return parser.parse_args()


def optional_imports() -> dict[str, object | None]:
    """선택 의존성을 불러오고 누락된 항목은 경고로 보고한다."""
    modules: dict[str, object | None] = {}
    for name in ["cv2", "albumentations", "matplotlib.pyplot", "sklearn.model_selection"]:
        try:
            module = __import__(name, fromlist=["*"])
            modules[name] = module
            print(f"[OK] import {name}")
        except Exception as exc:
            modules[name] = None
            print(f"[WARN] import {name} failed: {exc}")
    try:
        __import__("yaml")
        print("[OK] import yaml")
    except Exception as exc:
        print(f"[WARN] import yaml failed: {exc}")
    return modules


def list_images(root: Path) -> list[Path]:
    """디렉터리 아래의 이미지 파일을 정렬된 목록으로 반환한다."""
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def collect_local_samples(dataset_dir: Path) -> list[tuple[Path, int]]:
    """로컬 RPS 폴더 구조에서 이미지 경로와 클래스 ID를 수집한다."""
    samples: list[tuple[Path, int]] = []
    for class_id in [0, 1, 2]:
        class_dir = dataset_dir / str(class_id)
        files = list_images(class_dir)
        print(f"[DATA] class {class_id} ({cfg.CLASSES[class_id]}): {len(files)} images")
        samples.extend((path, class_id) for path in files)
    return samples


def collect_roboflow_samples(dataset_dir: Path) -> list[tuple[Path, int]]:
    """향후 Roboflow 입력 폴더를 사용할 때 기존 YOLO 구조의 이미지를 수집한다."""
    if dataset_dir is None or not dataset_dir.exists():
        return []
    print("[WARN] Roboflow mode currently validates/copies existing YOLO-like folders only.")
    samples: list[tuple[Path, int]] = []
    for image_path in list_images(dataset_dir):
        samples.append((image_path, 3))
    return samples


def prepare_negative_samples(negative_dir: Path | None = None) -> list[tuple[Path, int]]:
    """
    일반 손 제스처 이미지 또는 RPS가 아닌 이미지를 other class로 준비한다.
    """
    candidate_dirs = [p for p in [negative_dir, cfg.RAW_DATASET_DIR / "negative", cfg.RAW_DATASET_DIR / "3"] if p]
    for directory in candidate_dirs:
        files = list_images(directory)
        if files:
            print(f"[DATA] negative samples: {len(files)} images from {directory}")
            return [(path, 3) for path in files]
    print("No negative samples found")
    return []


def stratified_split(samples: list[tuple[Path, int]]) -> dict[str, list[tuple[Path, int]]]:
    """클래스 비율을 유지하며 train/val/test로 데이터를 분할한다."""
    rng = random.Random(cfg.RANDOM_STATE)
    grouped: dict[int, list[tuple[Path, int]]] = {}
    for sample in samples:
        grouped.setdefault(sample[1], []).append(sample)

    splits = {"train": [], "val": [], "test": []}
    for class_samples in grouped.values():
        rng.shuffle(class_samples)
        n = len(class_samples)
        n_train = int(n * cfg.TRAIN_RATIO)
        n_val = int(n * cfg.VAL_RATIO)
        splits["train"].extend(class_samples[:n_train])
        splits["val"].extend(class_samples[n_train : n_train + n_val])
        splits["test"].extend(class_samples[n_train + n_val :])

    for split_samples in splits.values():
        rng.shuffle(split_samples)
    return splits


def yolo_label_for_full_image(class_id: int) -> str:
    """전체 이미지를 손 ROI로 가정한 YOLO 라벨 문자열을 생성한다."""
    return f"{class_id} 0.500000 0.500000 1.000000 1.000000\n"


def create_augmentation():
    """Albumentations 기반 학습 이미지 증강 파이프라인을 생성한다."""
    try:
        import albumentations as A

        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomScale(scale_limit=0.1, p=0.5),
            ]
        )
    except Exception as exc:
        print(f"[WARN] Albumentations unavailable, augmentation disabled: {exc}")
        return None


def prepare_output_dirs(output_dir: Path) -> None:
    """YOLO 데이터셋 출력 디렉터리를 초기화한다."""
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_sample(
    image_path: Path,
    class_id: int,
    split: str,
    index: int,
    output_dir: Path,
    augment=False,
    augmenter=None,
) -> None:
    """이미지를 320x320으로 저장하고 대응하는 YOLO 라벨 파일을 생성한다."""
    import numpy as np

    image = Image.open(image_path).convert("RGB").resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
    stem = f"{split}_{index:06d}_{image_path.stem}"
    image_out = output_dir / "images" / split / f"{stem}.jpg"
    label_out = output_dir / "labels" / split / f"{stem}.txt"
    image.save(image_out, quality=95)
    label_out.write_text(yolo_label_for_full_image(class_id), encoding="utf-8")

    if augment and split == "train" and augmenter is not None:
        augmented = augmenter(image=np.array(image))["image"]
        aug_out = output_dir / "images" / split / f"{stem}_aug.jpg"
        aug_label = output_dir / "labels" / split / f"{stem}_aug.txt"
        Image.fromarray(augmented).save(aug_out, quality=95)
        aug_label.write_text(yolo_label_for_full_image(class_id), encoding="utf-8")


def write_data_yaml(output_dir: Path) -> None:
    """Ultralytics 학습용 data.yaml 파일을 작성한다."""
    import yaml

    data = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": cfg.CLASSES,
        "nc": len(cfg.CLASSES),
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def save_distribution_plot(samples: Iterable[tuple[Path, int]], output_path: Path) -> None:
    """클래스별 이미지 개수 막대그래프를 저장한다."""
    try:
        import matplotlib.pyplot as plt

        counts = Counter(class_id for _, class_id in samples)
        names = [cfg.CLASSES[i] for i in sorted(cfg.CLASSES)]
        values = [counts.get(i, 0) for i in sorted(cfg.CLASSES)]
        plt.figure(figsize=(7, 4))
        plt.bar(names, values)
        plt.title("RPS class distribution")
        plt.ylabel("images")
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"[LOG] saved {output_path}")
    except Exception as exc:
        print(f"[WARN] Could not save distribution plot: {exc}")


def dry_run(args: argparse.Namespace) -> int:
    """드라이런으로 경로, 의존성, 분할 로직을 검증한다."""
    print("[DRY-RUN] validating dataset preparation")
    optional_imports()
    print(f"BASE_DIR: {cfg.BASE_DIR}")
    print(f"RAW_DATASET_DIR exists: {args.source.exists()} ({args.source})")
    samples = collect_local_samples(args.source)
    samples += prepare_negative_samples(args.negative_dir)
    if not samples:
        print("[ERROR] No images found.")
        return 1
    splits = stratified_split(samples)
    print("[DRY-RUN] split counts:", {k: len(v) for k, v in splits.items()})
    print("[DRY-RUN] output:", cfg.RPS_YOLO_DIR)
    return 0


def main() -> int:
    """RPS 데이터셋을 YOLO 형식으로 변환하는 메인 실행 함수이다."""
    args = parse_args()
    cfg.ensure_project_dirs()
    if args.dry_run:
        return dry_run(args)

    try:
        samples = collect_roboflow_samples(args.roboflow_dir) if args.roboflow_dir else collect_local_samples(args.source)
        samples += prepare_negative_samples(args.negative_dir)
        if not samples:
            raise FileNotFoundError(f"No images found in {args.source}")
        if cfg.RPS_YOLO_DIR.exists():
            shutil.rmtree(cfg.RPS_YOLO_DIR)
        prepare_output_dirs(cfg.RPS_YOLO_DIR)
        splits = stratified_split(samples)
        augmenter = create_augmentation() if args.augment else None
        for split, split_samples in splits.items():
            print(f"[WRITE] {split}: {len(split_samples)} images")
            for idx, (image_path, class_id) in enumerate(split_samples):
                write_sample(image_path, class_id, split, idx, cfg.RPS_YOLO_DIR, args.augment, augmenter)
        write_data_yaml(cfg.RPS_YOLO_DIR)
        save_distribution_plot(samples, cfg.LOG_DIR / "dataset_stats.png")
        print(f"[DONE] YOLO dataset saved to {cfg.RPS_YOLO_DIR}")
        return 0
    except Exception as exc:
        print(f"[ERROR] dataset preparation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
