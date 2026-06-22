#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <tag> <engine> <dataset_dir> <manifest> <output_csv> [extra benchmark args...]" >&2
  exit 2
fi

TAG="$1"
ENGINE="$2"
DATASET_DIR="$3"
MANIFEST="$4"
OUTPUT_CSV="$5"
shift 5

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${ROOT_DIR}/reports"
mkdir -p "${REPORT_DIR}"

read -r -a PYTHON_CMD <<< "${PYTHON_BIN:-python3}"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-benchmark_trt.py}"
BENCH="${ROOT_DIR}/scripts/${BENCHMARK_SCRIPT}"
[[ -f "${BENCH}" ]] || { echo "Missing benchmark script: ${BENCH}" >&2; exit 1; }

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --output="${REPORT_DIR}/nsys_${TAG}" \
  "${PYTHON_CMD[@]}" "${BENCH}" \
  --engine "${ENGINE}" \
  --dataset-dir "${DATASET_DIR}" \
  --manifest "${MANIFEST}" \
  --output-csv "${OUTPUT_CSV}" \
  "$@"

ncu \
  --force-overwrite \
  --set full \
  --target-processes all \
  --launch-skip 10 \
  --launch-count 20 \
  --export "${REPORT_DIR}/ncu_${TAG}" \
  "${PYTHON_CMD[@]}" "${BENCH}" \
  --engine "${ENGINE}" \
  --dataset-dir "${DATASET_DIR}" \
  --manifest "${MANIFEST}" \
  --output-csv "${OUTPUT_CSV}" \
  "$@"

echo "Nsight reports written under ${REPORT_DIR}"
