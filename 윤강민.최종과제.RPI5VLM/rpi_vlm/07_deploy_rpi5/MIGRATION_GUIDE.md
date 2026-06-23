# Raspberry Pi 5 Migration Guide

Target: Raspberry Pi OS 64-bit Bookworm on Raspberry Pi 5.

## 1. Copy Project Files

From the dev machine:

```bash
rsync -av rpi_vlm/ pi@<pi-host>:~/rpi_vlm/
```

Also copy prepared model files:

```bash
rsync -av rpi_vlm/models/ pi@<pi-host>:~/rpi_vlm/models/
```

Prefer exporting YOLO ONNX and converting SmolVLM GGUF on the dev machine, then
copying the files to the Pi.

## 2. Install System Packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libopenblas-dev libopencv-dev
```

## 3. Python Environment

```bash
cd ~/rpi_vlm
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r 00_setup/requirements-rpi5.txt
```

If `llama-cpp-python` has no suitable wheel, build from source:

```bash
CMAKE_ARGS="-DGGML_NATIVE=ON -DGGML_ARM_DOTPROD=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
  pip install --no-binary llama-cpp-python llama-cpp-python==0.3.9
```

## 4. Verify Architecture and Imports

```bash
python 07_deploy_rpi5/check_arch.py
```

Expected:

- `machine: aarch64`
- NEON/ASIMD present
- `onnxruntime` import OK
- `llama_cpp` import OK

## 5. Smoke Tests

```bash
python 01_camera_capture.py --source 0 --no-window --save models/camera_smoke.jpg
python 02_yolo_detector.py --image models/camera_smoke.jpg --model models/yolov8n.onnx
python 04_vlm_server.py \
  --model models/SmolVLM-256M-Instruct-Q8_0.gguf \
  --mmproj models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
  --threads 4 \
  --threads-batch 4
```

In another shell:

```bash
python 04_vlm_inference.py \
  --image models/camera_smoke.jpg \
  --server-url http://127.0.0.1:8099/v1/chat/completions
```

## 6. Run Pipeline

```bash
python 05_pipeline_main.py \
  --source 0 \
  --yolo models/yolov8n.onnx \
  --vlm-server-url http://127.0.0.1:8099/v1/chat/completions
```

Press `q` to quit. Click inside a YOLO box to start background VLM inference.
The recommended path is a resident VLM server. If `--vlm-server-url` is omitted,
the pipeline falls back to one subprocess per click, which reloads the model and
is much slower.

## Notes

- Active cooling is strongly recommended.
- VLM inference will likely dominate latency on CPU-only Pi 5.
- `--mock-vlm` and `--mock-yolo` are useful to isolate UI/camera problems.
