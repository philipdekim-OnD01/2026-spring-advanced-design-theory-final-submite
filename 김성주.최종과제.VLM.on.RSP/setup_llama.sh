#!/usr/bin/env bash
# Clone and build llama.cpp into this folder so the VLM scripts can find
# llama.cpp/build/bin/llama-mtmd-cli and llama-server.
set -euo pipefail

cd "$(dirname "$0")"

REPO_URL="${LLAMA_REPO_URL:-https://github.com/ggml-org/llama.cpp.git}"
# Pin a known-good commit for reproducibility; override with LLAMA_REF=<tag/commit>.
REPO_REF="${LLAMA_REF:-master}"

if [ ! -f llama.cpp/CMakeLists.txt ]; then
  echo "[setup] cloning llama.cpp ($REPO_REF) ..."
  rm -rf llama.cpp
  git clone "$REPO_URL" llama.cpp
fi

cd llama.cpp
echo "[setup] checking out $REPO_REF ..."
git fetch --all --tags --quiet
git checkout "$REPO_REF"

echo "[setup] building (Release) ..."
cmake -B build
cmake --build build --config Release -j

echo "[setup] done. binaries:"
ls -1 build/bin/llama-mtmd-cli build/bin/llama-server 2>/dev/null || true
