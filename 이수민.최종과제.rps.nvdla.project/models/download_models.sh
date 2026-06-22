#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-${ROOT_DIR}/source}"
MODE="${1:-all}"
RELEASE_BASE="https://github.com/s0o0omin/2026-spring-advanced-design-theory-final-submite/releases/download/rps-nvdla-models-v1"
ONNX_ARCHIVE="rps_nvdla_onnx_models_v1.tar.gz"
TFLITE_ARCHIVE="rps_nvdla_tflite_models_v1.tar.gz"
ONNX_SHA256="82f07e065e1e9eb75fdf0600aa0c80a0828d1844f43e7915748c3ea9c2876d33"
TFLITE_SHA256="bd3a56f6e5e5d240fdbc0d5aeca73d2c1a37b3852f85ac5e8d79e4922dfa1f1a"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

verify_sha256() {
  local file="$1"
  local expected="$2"
  local actual
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "${file}" | awk '{print $1}')"
  else
    actual="$(shasum -a 256 "${file}" | awk '{print $1}')"
  fi
  [[ "${actual}" == "${expected}" ]] || {
    echo "Checksum mismatch for ${file}" >&2
    return 1
  }
}

download_one() {
  local archive="$1"
  local checksum="$2"
  local temp="${TMP_DIR}/${archive}"
  curl --fail --location --retry 3 "${RELEASE_BASE}/${archive}" --output "${temp}"
  verify_sha256 "${temp}" "${checksum}"
  mkdir -p "${DEST}"
  tar -xzf "${temp}" -C "${DEST}"
}

case "${MODE}" in
  onnx)
    download_one "${ONNX_ARCHIVE}" "${ONNX_SHA256}"
    ;;
  tflite)
    download_one "${TFLITE_ARCHIVE}" "${TFLITE_SHA256}"
    ;;
  all)
    download_one "${ONNX_ARCHIVE}" "${ONNX_SHA256}"
    download_one "${TFLITE_ARCHIVE}" "${TFLITE_SHA256}"
    ;;
  *)
    echo "Usage: $0 [onnx|tflite|all]" >&2
    exit 2
    ;;
esac

echo "Models extracted under ${DEST}/artifacts"
