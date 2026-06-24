#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <engine> <output_json>" >&2
  exit 2
fi

ENGINE="$1"
OUTPUT_JSON="$2"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

"${TRTEXEC}" \
  --loadEngine="${ENGINE}" \
  --dumpLayerInfo \
  --profilingVerbosity=detailed \
  --exportLayerInfo="${OUTPUT_JSON}" \
  --skipInference
