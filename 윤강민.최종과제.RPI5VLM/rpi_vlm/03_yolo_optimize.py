"""Optimize the trained RPS YOLO model for Raspberry Pi 5 deployment."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Any

import config as cfg


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Prune, quantize, and export RPS YOLO.")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without optimization.")
    parser.add_argument("--prune-amount", type=float, default=cfg.PRUNE_AMOUNT, help="L1 pruning amount for Conv2d.")
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


def model_size_mb(path: Path) -> str:
    """파일 또는 디렉터리 크기를 MB 문자열로 반환한다."""
    if not path.exists():
        return "N/A"
    if path.is_dir():
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    else:
        size = path.stat().st_size
    return f"{size / (1024 * 1024):.2f}"


def measure_sparsity(model: Any) -> float:
    """Conv2d 가중치 기준 전체 sparsity를 계산한다."""
    import torch

    total = 0
    zeros = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            weight = module.weight.detach()
            total += weight.numel()
            zeros += int(torch.count_nonzero(weight == 0).item())
    return zeros / total if total else 0.0


def prune_yolo_model(input_path: Path, output_path: Path, amount: float) -> None:
    """YOLO 모델의 Conv2d 가중치에 L1 비구조적 pruning을 적용한다."""
    import torch
    import torch.nn.utils.prune as prune
    from ultralytics import YOLO

    if not input_path.exists():
        raise FileNotFoundError(f"Missing baseline model: {input_path}")
    yolo = YOLO(str(input_path))
    for module in yolo.model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    sparsity = measure_sparsity(yolo.model)
    print(f"[PRUNE] measured Conv2d sparsity: {sparsity:.2%}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yolo.save(str(output_path))
    print(f"[PRUNE] saved {output_path}")


def export_onnx_and_quantize(model_path: Path) -> Path | None:
    """Ultralytics ONNX export 후 onnxsim과 ONNX Runtime INT8 양자화를 수행한다."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxsim import simplify
        from ultralytics import YOLO
        import onnx

        yolo = YOLO(str(model_path))
        exported = Path(yolo.export(format="onnx", imgsz=cfg.IMG_SIZE, simplify=False, dynamic=False))
        simplified = cfg.ONNX_MODEL_DIR / "rps_simplified.onnx"
        model = onnx.load(str(exported))
        sim_model, ok = simplify(model)
        if ok:
            onnx.save(sim_model, str(simplified))
        else:
            print("[WARN] onnxsim check failed; using raw ONNX export")
            simplified = exported
        int8_path = cfg.ONNX_MODEL_DIR / "rps_int8.onnx"
        quantize_dynamic(str(simplified), str(int8_path), weight_type=QuantType.QInt8)
        print(f"[PTQ] saved {int8_path}")
        return int8_path
    except Exception as exc:
        print(f"[WARN] INT8 ONNX quantization skipped: {exc}")
        return None


def export_ncnn(model_path: Path) -> tuple[Path | None, Path | None]:
    """Ultralytics NCNN export를 실행하고 param/bin 파일을 표준 위치로 복사한다."""
    try:
        from ultralytics import YOLO

        yolo = YOLO(str(model_path))
        exported = Path(yolo.export(format="ncnn", imgsz=cfg.IMG_SIZE))
        candidates = [exported] if exported.is_dir() else [exported.parent]
        candidates += list(model_path.parent.glob("*ncnn*"))
        param_files = []
        bin_files = []
        for directory in candidates:
            if directory.exists():
                param_files.extend(directory.rglob("*.param"))
                bin_files.extend(directory.rglob("*.bin"))
        if not param_files or not bin_files:
            raise FileNotFoundError("NCNN .param/.bin files were not produced.")
        cfg.NCNN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        param_out = cfg.NCNN_MODEL_DIR / "rps.param"
        bin_out = cfg.NCNN_MODEL_DIR / "rps.bin"
        shutil.copy2(param_files[0], param_out)
        shutil.copy2(bin_files[0], bin_out)
        runtime_dir = cfg.NCNN_MODEL_DIR / "rps_ncnn_model"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(param_files[0], runtime_dir / "model.ncnn.param")
        shutil.copy2(bin_files[0], runtime_dir / "model.ncnn.bin")
        metadata = next((p for p in candidates if (p / "metadata.yaml").exists()), None)
        if metadata is not None:
            shutil.copy2(metadata / "metadata.yaml", runtime_dir / "metadata.yaml")
        print(f"[NCNN] saved {param_out} and {bin_out}")
        return param_out, bin_out
    except Exception as exc:
        print(f"[WARN] NCNN export skipped: {exc}")
        return None, None


def smoke_validate_model(model_path: Path, min_conf: float = 0.5, limit: int = 12) -> bool:
    """Small deployment sanity check to avoid exporting a detector with no confident test predictions."""
    try:
        from ultralytics import YOLO

        image_dir = cfg.RPS_YOLO_DIR / "images" / "test"
        images = [p for p in sorted(image_dir.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}][:limit]
        if not images:
            print("[WARN] no test images available for optimization smoke validation")
            return True
        model = YOLO(str(model_path), task="detect")
        confident = 0
        for image in images:
            result = model.predict(str(image), imgsz=cfg.IMG_SIZE, conf=0.01, verbose=False)[0]
            boxes = getattr(result, "boxes", [])
            if boxes is not None and len(boxes) and float(boxes[0].conf[0].item()) >= min_conf:
                confident += 1
        ratio = confident / len(images)
        print(f"[SMOKE] {model_path.name}: {confident}/{len(images)} images >= {min_conf:.2f} confidence")
        return ratio >= 0.5
    except Exception as exc:
        print(f"[WARN] optimization smoke validation failed for {model_path}: {exc}")
        return False


def cpu_inference_ms(model_path: Path) -> str:
    """가능하면 CPU 단일 더미 추론 latency를 측정한다."""
    try:
        import numpy as np
        from ultralytics import YOLO

        if not model_path.exists():
            return "N/A"
        model = YOLO(str(model_path))
        dummy = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)
        model.predict(dummy, imgsz=cfg.IMG_SIZE, device="cpu", verbose=False)
        start = time.perf_counter()
        model.predict(dummy, imgsz=cfg.IMG_SIZE, device="cpu", verbose=False)
        return f"{(time.perf_counter() - start) * 1000:.1f}"
    except Exception:
        return "N/A"


def write_comparison() -> None:
    """모델 크기와 가능한 CPU 추론 시간을 비교표로 저장한다."""
    rows = [
        ("baseline .pt", cfg.YOLO_MODEL_DIR / "rps_baseline.pt", "N/A"),
        ("pruned .pt", cfg.YOLO_MODEL_DIR / "rps_pruned.pt", "N/A"),
        ("INT8 .onnx", cfg.ONNX_MODEL_DIR / "rps_int8.onnx", "N/A"),
        ("NCNN .param", cfg.NCNN_MODEL_DIR, "N/A"),
    ]
    lines = [
        "| model         | size_MB | mAP50 | cpu_inference_ms |",
        "|---------------|---------|-------|------------------|",
    ]
    for name, path, map50 in rows:
        latency = cpu_inference_ms(path) if path.suffix == ".pt" else "N/A"
        lines.append(f"| {name:<13} | {model_size_mb(path):>7} | {map50:<5} | {latency:<16} |")
    out = cfg.LOG_DIR / "model_comparison.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[LOG] saved {out}")


def append_comparison_note(note: str) -> None:
    """Append deployment fallback notes to the model comparison report."""
    out = cfg.LOG_DIR / "model_comparison.txt"
    with out.open("a", encoding="utf-8") as f:
        f.write(f"\nNOTE: {note}\n")


def dry_run(args: argparse.Namespace) -> int:
    """드라이런으로 최적화 입력과 의존성 상태를 검증한다."""
    print("[DRY-RUN] validating YOLO optimization setup")
    print_cuda_report()
    print(f"[DRY-RUN] prune amount: {args.prune_amount}")
    print(f"[DRY-RUN] baseline exists: {(cfg.YOLO_MODEL_DIR / 'rps_baseline.pt').exists()}")
    for name in ["torch", "ultralytics", "onnx", "onnxruntime", "onnxsim"]:
        try:
            __import__(name)
            print(f"[OK] import {name}")
        except Exception as exc:
            print(f"[WARN] import {name} failed: {exc}")
    return 0


def main() -> int:
    """Pruning, ONNX INT8 PTQ, NCNN export, 비교표 생성을 순서대로 실행한다."""
    args = parse_args()
    cfg.ensure_project_dirs()
    if args.dry_run:
        return dry_run(args)
    try:
        baseline = cfg.YOLO_MODEL_DIR / "rps_baseline.pt"
        pruned = cfg.YOLO_MODEL_DIR / "rps_pruned.pt"
        print_cuda_report()
        prune_yolo_model(baseline, pruned, args.prune_amount)
        deployment_model = pruned
        fallback_note = ""
        if not smoke_validate_model(pruned):
            deployment_model = baseline
            fallback_note = "Pruned model failed confidence smoke validation; ONNX/NCNN deployment artifacts were exported from the baseline model."
            print(f"[WARN] {fallback_note}")
        export_onnx_and_quantize(deployment_model)
        export_ncnn(deployment_model)
        write_comparison()
        if fallback_note:
            append_comparison_note(fallback_note)
        return 0
    except Exception as exc:
        print(f"[ERROR] optimization failed: {exc}")
        write_comparison()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
