#!/usr/bin/env python3
"""Compare GPU-only and strict-DLA power using AGX Orin tegrastats rails."""

from __future__ import annotations

import argparse
import csv
import json
import re
import socket
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from benchmark_trt_native_int8 import (
    NativeInt8Inference,
    load_calibration_scales,
    load_manifest,
    quantize_image,
)


RAIL_PATTERNS = {
    "vdd_gpu_soc_mw": re.compile(r"VDD_GPU_SOC\s+(\d+)mW/"),
    "vdd_cpu_cv_mw": re.compile(r"VDD_CPU_CV\s+(\d+)mW/"),
    "vin_sys_5v0_mw": re.compile(r"VIN_SYS_5V0\s+(\d+)mW/"),
    "vddq_vdd2_1v8ao_mw": re.compile(r"VDDQ_VDD2_1V8AO\s+(\d+)mW/"),
}
GPU_PATTERN = re.compile(r"GR3D_FREQ\s+(\d+)%")


@dataclass(frozen=True)
class Target:
    device: str
    engine: Path
    calib_cache: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-engine", required=True, type=Path)
    parser.add_argument("--gpu-calib-cache", required=True, type=Path)
    parser.add_argument("--dla-engine", required=True, type=Path)
    parser.add_argument("--dla-calib-cache", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--duration", default=20.0, type=float)
    parser.add_argument("--idle-duration", default=10.0, type=float)
    parser.add_argument("--cooldown", default=5.0, type=float)
    parser.add_argument("--interval-ms", default=200, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--warmup", default=500, type=int)
    parser.add_argument("--img-size", default=64, type=int)
    return parser.parse_args()


def start_tegrastats(interval_ms: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        ["/usr/bin/tegrastats", "--interval", str(interval_ms)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def stop_tegrastats(process: subprocess.Popen[str]) -> list[str]:
    process.terminate()
    try:
        stdout, stderr = process.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=3)
    if process.returncode not in (0, -15):
        raise RuntimeError(f"tegrastats failed ({process.returncode}): {stderr.strip()}")
    return [line for line in stdout.splitlines() if line.strip()]


def parse_samples(
    lines: list[str], *, repeat: int, sequence: int, device: str, phase: str
) -> list[dict[str, int | str]]:
    samples: list[dict[str, int | str]] = []
    for sample_index, line in enumerate(lines):
        values: dict[str, int | str] = {
            "repeat": repeat,
            "sequence": sequence,
            "device": device,
            "phase": phase,
            "sample_index": sample_index,
        }
        for name, pattern in RAIL_PATTERNS.items():
            match = pattern.search(line)
            if match is None:
                break
            values[name] = int(match.group(1))
        else:
            gpu_match = GPU_PATTERN.search(line)
            values["gr3d_freq_percent"] = int(gpu_match.group(1)) if gpu_match else -1
            values["compute_rails_mw"] = (
                int(values["vdd_gpu_soc_mw"]) + int(values["vdd_cpu_cv_mw"])
            )
            samples.append(values)
    if not samples:
        raise RuntimeError(f"No tegrastats power samples parsed for {device} {phase}")
    return samples


def measure_idle(args: argparse.Namespace, repeat: int, sequence: int, device: str) -> list[dict]:
    time.sleep(args.cooldown)
    process = start_tegrastats(args.interval_ms)
    try:
        time.sleep(args.idle_duration)
    finally:
        lines = stop_tegrastats(process)
    return parse_samples(lines, repeat=repeat, sequence=sequence, device=device, phase="idle")


def measure_load(
    args: argparse.Namespace,
    repeat: int,
    sequence: int,
    target: Target,
    runner: NativeInt8Inference,
    packed_input,
) -> tuple[list[dict], int, float]:
    for _ in range(args.warmup):
        runner.infer(packed_input)

    process = start_tegrastats(args.interval_ms)
    count = 0
    start = time.perf_counter()
    try:
        deadline = start + args.duration
        while time.perf_counter() < deadline:
            runner.infer(packed_input)
            count += 1
    finally:
        elapsed = time.perf_counter() - start
        lines = stop_tegrastats(process)
    samples = parse_samples(
        lines, repeat=repeat, sequence=sequence, device=target.device, phase="load"
    )
    return samples, count, elapsed


def mean(samples: list[dict], field: str) -> float:
    return statistics.fmean(float(sample[field]) for sample in samples)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "gpu": Target("gpu", args.gpu_engine, args.gpu_calib_cache),
        "dla0": Target("dla0", args.dla_engine, args.dla_calib_cache),
    }
    runners = {name: NativeInt8Inference(target.engine) for name, target in targets.items()}
    test_path, _ = load_manifest(args.manifest)[0]
    packed_inputs = {}
    for name, target in targets.items():
        scales = load_calibration_scales(target.calib_cache)
        input_scale = scales.get(runners[name].input_name)
        if input_scale is None:
            raise KeyError(f"No input calibration scale for {name}")
        packed_inputs[name] = quantize_image(
            args.dataset_dir / test_path, args.img_size, input_scale
        )

    all_samples: list[dict] = []
    run_rows: list[dict] = []
    sequence = 0
    for repeat in range(1, args.repeats + 1):
        order = ("gpu", "dla0") if repeat % 2 else ("dla0", "gpu")
        for name in order:
            sequence += 1
            target = targets[name]
            print(f"[{sequence}/{args.repeats * 2}] repeat={repeat} device={name} idle")
            idle = measure_idle(args, repeat, sequence, name)
            print(f"[{sequence}/{args.repeats * 2}] repeat={repeat} device={name} load")
            load, count, elapsed = measure_load(
                args, repeat, sequence, target, runners[name], packed_inputs[name]
            )
            all_samples.extend(idle)
            all_samples.extend(load)

            qps = count / elapsed
            row: dict[str, float | int | str] = {
                "repeat": repeat,
                "sequence": sequence,
                "device": name,
                "idle_samples": len(idle),
                "load_samples": len(load),
                "inferences": count,
                "elapsed_s": elapsed,
                "throughput_qps": qps,
            }
            for field in (*RAIL_PATTERNS, "compute_rails_mw"):
                idle_mean = mean(idle, field)
                load_mean = mean(load, field)
                row[f"idle_{field}"] = idle_mean
                row[f"load_{field}"] = load_mean
                row[f"delta_{field}"] = load_mean - idle_mean
            row["load_energy_mj_per_inference"] = row["load_compute_rails_mw"] / qps
            row["delta_energy_mj_per_inference"] = row["delta_compute_rails_mw"] / qps
            run_rows.append(row)

    sample_fields = [
        "repeat", "sequence", "device", "phase", "sample_index",
        *RAIL_PATTERNS, "compute_rails_mw", "gr3d_freq_percent",
    ]
    run_fields = list(run_rows[0])
    write_csv(args.output_dir / "power_raw_samples.csv", all_samples, sample_fields)
    write_csv(args.output_dir / "power_runs.csv", run_rows, run_fields)

    summary_rows: list[dict] = []
    metrics = [field for field in run_fields if field not in {
        "repeat", "sequence", "device", "idle_samples", "load_samples", "inferences"
    }]
    for device in ("gpu", "dla0"):
        selected = [row for row in run_rows if row["device"] == device]
        summary: dict[str, float | int | str] = {"device": device, "repeats": len(selected)}
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    write_csv(args.output_dir / "power_summary.csv", summary_rows, list(summary_rows[0]))

    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "protocol": {
            "duration_s": args.duration,
            "idle_duration_s": args.idle_duration,
            "cooldown_s": args.cooldown,
            "interval_ms": args.interval_ms,
            "repeats": args.repeats,
            "order": "GPU,DLA / DLA,GPU / alternating",
            "input": str(args.dataset_dir / test_path),
        },
        "engines": {name: str(target.engine) for name, target in targets.items()},
        "rail_note": (
            "VDD_GPU_SOC supplies GPU and SoC; VDD_CPU_CV supplies CPU and CV cores "
            "including DLA/PVA. compute_rails_mw is their sum."
        ),
    }
    (args.output_dir / "power_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
