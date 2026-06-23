#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec llama.cpp/build/bin/llama-server \
  --hf-repo ggml-org/SmolVLM-500M-Instruct-GGUF:Q8_0 \
  --host 127.0.0.1 \
  --port 8082 \
  -t 4 \
  -c 768 \
  --no-warmup \
  --verbosity 1
