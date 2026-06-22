#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda)}"
CONDA_ENV="${CONDA_ENV:-rps_int8_nvdla}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda/lib64}"
ENV_LIB_DIR="${CONDA_ENV_PREFIX:-${HOME}/miniconda3/envs/${CONDA_ENV}}/lib"
LD_PATH="${ENV_LIB_DIR}:${CUDA_LIB_DIR}"
DATASET_DIR="${PROJECT_DIR}/RPS_Dataset"
RESULT_CSV="${RESULT_CSV:-${PROJECT_DIR}/results/native_strict_sweep.csv}"
CALIB_SAMPLES="${CALIB_SAMPLES:-512}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-16}"
WARMUP="${WARMUP:-100}"
REPEAT="${REPEAT:-3}"

cd "${PROJECT_DIR}"
mkdir -p engines results logs

run_py() {
  "${CONDA_BIN}" run -n "${CONDA_ENV}" env LD_LIBRARY_PATH="${LD_PATH}" python "$@"
}

run_one() {
  local artifact="$1"
  local tag="$2"
  local artifact_dir="${PROJECT_DIR}/artifacts/${artifact}"
  local onnx="${artifact_dir}/model_float32_nchw_b1.onnx"
  local manifest="${artifact_dir}/manifest.csv"
  local gpu_engine="${PROJECT_DIR}/engines/${tag}_gpu_native_int8_b1.engine"
  local dla_engine="${PROJECT_DIR}/engines/${tag}_dla0_strict_native_int8_b1.engine"

  echo "[${tag}] building GPU native INT8 engine"
  run_py scripts/build_trt_engine.py \
    --onnx "${onnx}" --engine "${gpu_engine}" \
    --precision int8 --device gpu --native-int8-io \
    --dataset-dir "${DATASET_DIR}" --manifest "${manifest}" \
    --calib-samples "${CALIB_SAMPLES}" --calib-batch-size "${CALIB_BATCH_SIZE}"

  echo "[${tag}] benchmarking GPU native INT8 engine"
  run_py scripts/benchmark_trt_native_int8.py \
    --engine "${gpu_engine}" --calib-cache "${gpu_engine%.engine}.calib.cache" \
    --dataset-dir "${DATASET_DIR}" --manifest "${manifest}" \
    --output-csv "${RESULT_CSV}" --device gpu \
    --model-name "rps_dla32_${tag}_int8" --warmup "${WARMUP}" --repeat "${REPEAT}"

  echo "[${tag}] building DLA0 strict native INT8 engine"
  run_py scripts/build_trt_engine.py \
    --onnx "${onnx}" --engine "${dla_engine}" \
    --precision int8 --device dla --dla-core 0 --native-int8-io --strict-dla \
    --dataset-dir "${DATASET_DIR}" --manifest "${manifest}" \
    --calib-samples "${CALIB_SAMPLES}" --calib-batch-size "${CALIB_BATCH_SIZE}"

  echo "[${tag}] benchmarking DLA0 strict native INT8 engine"
  run_py scripts/benchmark_trt_native_int8.py \
    --engine "${dla_engine}" --calib-cache "${dla_engine%.engine}.calib.cache" \
    --dataset-dir "${DATASET_DIR}" --manifest "${manifest}" \
    --output-csv "${RESULT_CSV}" --device dla0 \
    --model-name "rps_dla32_${tag}_int8" --warmup "${WARMUP}" --repeat "${REPEAT}"
}

if (($# == 0)); then
  echo "Usage: $0 artifact:tag [...]" >&2
  exit 2
fi

for item in "$@"; do
  run_one "${item%%:*}" "${item##*:}"
done

echo "wrote ${RESULT_CSV}"
