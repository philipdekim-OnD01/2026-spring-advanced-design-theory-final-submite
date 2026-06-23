#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
REQ_FILE="${ROOT_DIR}/00_setup/requirements.txt"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "${REQ_FILE}"

echo "Environment installed at ${VENV_DIR}"
echo "Next: python ${ROOT_DIR}/00_setup/download_models.py --export-yolo"

