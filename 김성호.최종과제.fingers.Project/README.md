# Finger Count (1–5) — On-Device Webcam Classifier

Real-time **finger counting (1–5)** from a webcam, built with **MobileNetV2 transfer learning**, quantized to **INT8 TFLite**, and deployed on a **Raspberry Pi** (`tflite_runtime` / `ai_edge_litert`).

The whole pipeline — **train → fine-tune → quantize → deploy** — runs from a few small scripts, and is designed so that **training and inference share the exact same preprocessing** (color + square crop), which was the key to making it work in a real environment.

## Results

| Item | Value |
|------|-------|
| Model | MobileNetV2 (α=1.0, ImageNet), input 160×160 color |
| Validation accuracy (30 imgs) | **100%** |
| INT8 accuracy (post-quant) | **100%** (lossless) |
| Model size | 8.7 MB → **2.68 MB** (≈3.25× smaller) |
| Inference | ~9.7 ms |
| Dataset | 150 self-captured color square photos (~30/class) |

> 100% is measured on a held-out split of the same capture distribution. Real-world robustness (different lighting/background) improves with more captured data.

## How it works

1. **Transfer learning** — start from ImageNet-pretrained MobileNetV2 (`include_top=False`) and add a small head (`GAP → Dropout → Dense(5, softmax)`).
2. **Two-phase training** — (1) train the head with the base frozen, then (2) **fine-tune** the base at a low learning rate (BatchNorm frozen).
3. **Data augmentation** — random flip / rotation / translation / zoom / brightness / contrast, applied during training only.
4. **INT8 post-training quantization** — full-integer quantization calibrated on the real photos; I/O kept float32 for simple board integration.
5. **Domain match** — capture, training, and inference all use **color + square crop**, so the model sees the same kind of image at train and run time.

**Why MobileNetV2?** It is reliably deployable on the Raspberry Pi (XNNPACK-compatible). MobileNetV3 / EfficientNetV2 produced XNNPACK `prepare` failures after TFLite conversion, so accuracy is pushed via input resolution / width (α) / more data rather than a heavier backbone.

## Repository layout

```
.
├── train_fingers.py            # Transfer-learning trainer (head + fine-tune), selectable backbone
├── quantize_fingers.py         # INT8 quantization + benchmark (auto-detects model input size)
├── run_fingers_board_real.py   # Webcam real-time inference (display auto-detect, runtime auto-select)
├── capture_fingers.py          # Capture your own training images from a webcam (ROI box)
├── requirements.txt
├── files/
│   ├── fingers_real/<1..5>/    # training images (ImageFolder, color square) — not committed
│   └── save/                   # outputs: fingers_model.keras, fingers_int8.tflite, labels.txt
└── results/benchmark.csv       # size / accuracy / speed comparison
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # training (TensorFlow)
# Inference only (e.g. Raspberry Pi):
#   pip install ai-edge-litert numpy && sudo apt install -y python3-opencv
```

## Usage

### 1. (Optional) Capture your own data
```bash
python capture_fingers.py --roi 0.7
# Fill the green box with your hand, press 1–5 to label, u to undo, q to quit.
# ~50–300 images per class recommended.
```

### 2. Train (transfer learning + fine-tune), then quantize
```bash
python train_fingers.py --backbone mobilenetv2 --alpha 1.0 --img-size 160
python quantize_fingers.py --model files/save/fingers_model.keras --custom-dir files/fingers_real
```
Outputs: `files/save/fingers_model.keras` → `files/save/fingers_int8.tflite` + `labels.txt`.
Accuracy knobs: `--img-size 192/224`, `--alpha 1.3`, more data.

### 3. Run inference (webcam)
```bash
python run_fingers_board_real.py --color --roi 1.0
# --roi 1.0 center-crops the webcam frame to a square to match the square training images.
```

## Raspberry Pi deployment

Copy three files to the Pi and run:

```bash
# from your PC
scp files/save/fingers_int8.tflite files/save/labels.txt run_fingers_board_real.py pi@<host>:~/fingers/

# on the Pi (skip install if tflite_runtime is already present)
sudo apt install -y python3-opencv
pip install ai-edge-litert numpy
python run_fingers_board_real.py --color --roi 1.0 --model fingers_int8.tflite --labels labels.txt
```

The `.tflite` file is architecture-independent — no re-conversion needed. The runtime is auto-selected in the order `ai_edge_litert → tflite_runtime → tensorflow.lite`.

## Scripts reference

| Script | Key options |
|--------|-------------|
| `train_fingers.py` | `--backbone {mobilenetv2,mobilenetv3large,mobilenetv3small,efficientnetv2b0}` `--alpha` `--img-size` `--epochs-head` `--epochs-finetune` |
| `quantize_fingers.py` | `--model` `--custom-dir` `--gray` `--int8-io` `--rep-samples` |
| `run_fingers_board_real.py` | `--color` `--roi` `--camera` `--headless` `--model` `--labels` |
| `capture_fingers.py` | `--roi` `--img-size` `--out-dir` `--camera` |

## Acknowledgements

Built on top of an on-device AI lightweighting course lab (TensorFlow Model Optimization). The board inference structure follows the course's Raspberry Pi example.

## License

MIT
