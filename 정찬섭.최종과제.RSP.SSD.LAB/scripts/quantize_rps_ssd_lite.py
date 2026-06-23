from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent

REPO_ROOT = ROOT.parent

SOURCE_H5 = REPO_ROOT / "models" / "ssd_lite_rps.h5"
REPRESENTATIVE_IMAGE_DIR = REPO_ROOT / "assets" / "representative_images"

OUT_DIR = REPO_ROOT / "models"
FP32_TFLITE = OUT_DIR / "rps_ssd_lite_fp32.tflite"
INT8_TFLITE = OUT_DIR / "rps_ssd_lite_int8.tflite"
SAVED_MODEL_DIR = OUT_DIR / "ssd_lite_rps_saved_model"

IMG_SIZE = 64


def representative_dataset():
    image_paths = sorted(REPRESENTATIVE_IMAGE_DIR.glob("*.png"))[:200]
    if not image_paths:
        raise RuntimeError(f"No representative images found under {REPRESENTATIVE_IMAGE_DIR}")

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        yield [np.expand_dims(img, axis=0)]


def write_tflite(saved_model_dir, output_path, quantize=False):
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    return output_path.stat().st_size


def describe_model(path):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "inputs": [
            {
                "name": d["name"],
                "shape": d["shape"].tolist(),
                "dtype": np.dtype(d["dtype"]).name,
                "quantization": d["quantization"],
            }
            for d in inputs
        ],
        "outputs": [
            {
                "name": d["name"],
                "shape": d["shape"].tolist(),
                "dtype": np.dtype(d["dtype"]).name,
                "quantization": d["quantization"],
            }
            for d in outputs
        ],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(SOURCE_H5, compile=False)
    model.export(SAVED_MODEL_DIR)

    fp32_size = write_tflite(SAVED_MODEL_DIR, FP32_TFLITE, quantize=False)
    int8_size = write_tflite(SAVED_MODEL_DIR, INT8_TFLITE, quantize=True)

    print("Created:")
    print(f"- {FP32_TFLITE} ({fp32_size:,} bytes)")
    print(f"- {INT8_TFLITE} ({int8_size:,} bytes)")
    print()
    print("FP32 details:", describe_model(FP32_TFLITE))
    print("INT8 details:", describe_model(INT8_TFLITE))
    print()
    print(f"Size reduction: {(1 - int8_size / fp32_size) * 100:.1f}%")


if __name__ == "__main__":
    main()
