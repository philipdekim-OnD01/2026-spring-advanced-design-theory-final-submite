#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENGINE_DIR="${ENGINE_DIR:-${PROJECT_DIR}/engines}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/power_raw}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
TEGRASTATS="${TEGRASTATS:-/usr/bin/tegrastats}"
INTERVAL_MS="${INTERVAL_MS:-200}"
IDLE_SECONDS="${IDLE_SECONDS:-8}"
MONITOR_SECONDS="${MONITOR_SECONDS:-18}"
WARMUP_MS="${WARMUP_MS:-3000}"
DURATION_SECONDS="${DURATION_SECONDS:-12}"

MODELS=(w0_25 w0_5 w1_0 w2_0 w4_0 w8_0 w16_0)
mkdir -p "${OUTPUT_DIR}"

run_monitor() {
  local seconds="$1"
  local output="$2"
  local status=0
  timeout "${seconds}s" "${TEGRASTATS}" --interval "${INTERVAL_MS}" > "${output}" || status=$?
  if [[ "${status}" -ne 0 && "${status}" -ne 124 ]]; then
    return "${status}"
  fi
}

measure_one() {
  local repeat="$1"
  local sequence="$2"
  local model="$3"
  local device="$4"
  local tag="r${repeat}_s$(printf '%02d' "${sequence}")_${model}_${device}"
  local engine
  if [[ "${device}" == "gpu" ]]; then
    engine="${ENGINE_DIR}/${model}_gpu_native_int8_b1.engine"
  else
    engine="${ENGINE_DIR}/${model}_dla0_strict_native_int8_b1.engine"
  fi
  [[ -f "${engine}" ]] || { echo "Missing engine: ${engine}" >&2; return 1; }

  echo "[${tag}] idle baseline"
  run_monitor "${IDLE_SECONDS}" "${OUTPUT_DIR}/${tag}_idle.log"

  echo "[${tag}] saturated load"
  run_monitor "${MONITOR_SECONDS}" "${OUTPUT_DIR}/${tag}_load.log" &
  local monitor_pid=$!
  sleep 1
  "${TRTEXEC}" \
    --loadEngine="${engine}" \
    --warmUp="${WARMUP_MS}" \
    --duration="${DURATION_SECONDS}" \
    --iterations=1 \
    --noDataTransfers \
    --avgRuns=999999 \
    > "${OUTPUT_DIR}/${tag}_trtexec.log" 2>&1
  wait "${monitor_pid}"
  grep -q "&&&& PASSED" "${OUTPUT_DIR}/${tag}_trtexec.log"
}

sequence=0
for model in "${MODELS[@]}"; do
  for device in gpu dla0; do
    sequence=$((sequence + 1))
    measure_one 1 "${sequence}" "${model}" "${device}"
  done
done

for ((index=${#MODELS[@]}-1; index>=0; index--)); do
  model="${MODELS[index]}"
  for device in dla0 gpu; do
    sequence=$((sequence + 1))
    measure_one 2 "${sequence}" "${model}" "${device}"
  done
done

for model in "${MODELS[@]}"; do
  for device in gpu dla0; do
    sequence=$((sequence + 1))
    measure_one 3 "${sequence}" "${model}" "${device}"
  done
done

{
  date --iso-8601=seconds
  uname -a
  "${TRTEXEC}" --version 2>&1 | head -n 1
  nvpmodel -q 2>/dev/null || true
} > "${OUTPUT_DIR}/environment.txt"

echo "Completed ${sequence} power runs under ${OUTPUT_DIR}"
