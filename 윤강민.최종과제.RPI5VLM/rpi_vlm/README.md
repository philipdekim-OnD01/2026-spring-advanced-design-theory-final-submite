# RPi5 YOLO + Click-to-VLM Pipeline

This folder contains a staged Raspberry Pi 5-oriented pipeline:

1. YOLOv8n ONNX object detection runs continuously on a camera/video/image source.
2. Clicking inside a bounding box crops that object and sends it to SmolVLM GGUF.
3. VLM inference runs asynchronously in a background thread, so display/YOLO keeps running.

## Setup

```bash
cd rpi_vlm
bash 00_setup/setup_env.sh
source .venv/bin/activate
python 00_setup/download_models.py --export-yolo --download-smolvlm
```

By default, `download_models.py` fetches the verified ONNX YOLOv8n model and
the official SmolVLM Q8_0 GGUF bundle:

```text
models/SmolVLM-256M-Instruct-Q8_0.gguf
models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf
```

Optional local SmolVLM GGUF conversion requires a compatible `llama.cpp` checkout:

```bash
python 00_setup/download_models.py \
  --download-smolvlm \
  --convert-smolvlm \
  --llama-cpp /path/to/llama.cpp \
  --quant Q4_K_M
```

## Stage Commands

터미널 하나 열어서 해당 작업 실행 (VLM 계속 돌려놓기)
```bash
python 04_vlm_server.py \
  --model models/SmolVLM-256M-Instruct-Q8_0.gguf \
  --mmproj models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
  --threads 4 --threads-batch 4
```

터미널 하나 더 열어서 해당 작업 실행 (YOLO 및 클릭시 VLM 호출하는 main pipeline)
```bash
python 05_pipeline_main.py \
  --source 0 \
  --yolo models/yolov8n.onnx \
  --vlm-server-url http://127.0.0.1:8099/v1/chat/completions \
  --threads 4 --crop-max-side 512
```

Behavior:

- `q`: quit
- mouse click inside box: crop and send to VLM in background
- duplicate click on a box already running VLM is ignored
- overlapping boxes: smallest containing box wins
- use `--vlm-server-url` for normal real VLM operation. It keeps SmolVLM
  resident in memory and avoids reloading GGUF/mmproj for every click.
- if no server URL is provided, real VLM falls back to a background subprocess
  per click. That is stable but much slower and intended only as a fallback.