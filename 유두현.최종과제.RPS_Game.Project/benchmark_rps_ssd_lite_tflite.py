import statistics
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import tflite_runtime.interpreter as tflite


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT.parent / "models"
MODELS = [
    MODEL_DIR / "rps_ssd_lite_fp32.tflite",
    MODEL_DIR / "rps_ssd_lite_int8.tflite",
]


def make_input(input_detail):
    shape = input_detail["shape"]
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return np.random.random(shape).astype(np.float32)

    scale, zero_point = input_detail["quantization"]
    fp = np.random.random(shape).astype(np.float32)
    q = np.round(fp / scale + zero_point)
    q = np.clip(q, np.iinfo(dtype).min, np.iinfo(dtype).max)
    return q.astype(dtype)


def benchmark(model_path, warmup=10, repeat=100):
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    sample = make_input(input_detail)

    interpreter.set_tensor(input_detail["index"], sample)
    for _ in range(warmup):
        interpreter.invoke()

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        interpreter.set_tensor(input_detail["index"], sample)
        interpreter.invoke()
        times.append((time.perf_counter() - start) * 1000)

    ps_cpu_pct, ps_rss_kb = read_ps_usage()

    return {
        "model": model_path.name,
        "bytes": model_path.stat().st_size,
        "input_shape": input_detail["shape"].tolist(),
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "input_quant": input_detail["quantization"],
        "output_shape": output_detail["shape"].tolist(),
        "output_dtype": np.dtype(output_detail["dtype"]).name,
        "output_quant": output_detail["quantization"],
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "ps_cpu_pct": ps_cpu_pct,
        "ps_rss_kb": ps_rss_kb,
        "ps_rss_mb": ps_rss_kb / 1024,
    }


def read_ps_usage():
    output = subprocess.check_output(
        ["ps", "-p", str(os.getpid()), "-o", "%cpu=,rss="],
        text=True,
    ).strip()
    cpu_text, rss_text = output.split()
    return float(cpu_text), int(rss_text)


def main():
    results = [benchmark(path) for path in MODELS]
    for result in results:
        print(result)

    if len(results) == 2:
        fp32, int8 = results
        print()
        print(f"size_reduction_pct={(1 - int8['bytes'] / fp32['bytes']) * 100:.1f}")
        print(f"mean_speedup_x={fp32['mean_ms'] / int8['mean_ms']:.2f}")


if __name__ == "__main__":
    main()
