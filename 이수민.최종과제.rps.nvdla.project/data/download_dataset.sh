#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-${ROOT_DIR}/source/RPS_Dataset}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

if [[ -d "${DEST}" ]]; then
  echo "Dataset already exists: ${DEST}"
  exit 0
fi

git clone --depth 1 https://github.com/s0o0omin/RP2-RPS-QAT-Lab.git "${TMP_DIR}/repo"
cp -R "${TMP_DIR}/repo/RPS_Dataset" "${DEST}"
echo "Dataset downloaded to ${DEST}"
