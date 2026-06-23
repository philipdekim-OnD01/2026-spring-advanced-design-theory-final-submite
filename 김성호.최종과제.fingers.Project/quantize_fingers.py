"""INT8-quantize the trained finger model (.keras), convert to .tflite + benchmark.

Based on the EX_05~08 pattern. By default this is full integer (INT8) quantization, and for
compatibility with the board execution code the input/output stays float32 (since preprocessing
is inside the model, the board feeds 0~255 directly).
Passing --int8-io makes the input/output int8 as well (applying scale/zero-point).

Outputs:
    files/save/fingers_float.tflite   (non-quantized, baseline for size/speed comparison)
    files/save/fingers_int8.tflite    (INT8, for deployment)
    results/benchmark.csv

Usage examples:
    python quantize_fingers.py
    python quantize_fingers.py --int8-io          # int8 for input/output too
    python quantize_fingers.py --rep-samples 300  # number of representative data samples
"""

import argparse
import glob
import gzip
import os
import sys
from pathlib import Path


def _ensure_cuda_libs() -> None:
    """Inject the venv's nvidia-*-cu12 library paths into LD_LIBRARY_PATH (call before importing tensorflow)."""
    try:
        import nvidia
    except ImportError:
        return
    base = os.path.dirname(nvidia.__file__)
    libdirs = sorted({os.path.dirname(p) for p in glob.glob(base + '/*/lib/*.so*')})
    if not libdirs:
        return
    want = ':'.join(libdirs)
    current = os.environ.get('LD_LIBRARY_PATH', '')
    if want in current:
        return
    os.environ['LD_LIBRARY_PATH'] = (want + ':' + current).rstrip(':')
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_cuda_libs()

import cv2
import numpy as np
import tensorflow as tf

keras = tf.keras

IMG_SIZE = 96   # must be identical to train_fingers.py (the model input size)
CLASS_NAMES = ['1', '2', '3', '4', '5']


# --- Data loader (same rules as train_fingers.py, duplicated here so this file runs standalone) ---
def _label_from_filename(path: str) -> int:
    stem = Path(path).stem
    try:
        return int(stem[-2])
    except (ValueError, IndexError):
        return -1


def load_split(data_dir: str, split: str, img_size: int = 96):
    pattern = os.path.join(data_dir, split, '*.png')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No images found in '{pattern}'. Make sure you have downloaded "
            f"Kaggle 'koryakinp/fingers' into files/fingers/{{train,test}}.")
    X, y = [], []
    for f in files:
        cnt = _label_from_filename(f)
        if cnt < 1 or cnt > 5:
            continue
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        img = np.stack([img, img, img], axis=-1)
        X.append(img)
        y.append(cnt - 1)
    return np.asarray(X, dtype=np.uint8), np.asarray(y, dtype=np.int32)


def load_custom_dir(custom_dir: str, gray: bool = False, img_size: int = 96):
    """ImageFolder load: <dir>/<class>/*.{jpg,png} -> (X_uint8 3ch, y, class_names).

    If gray=True, grayscale->3 channels (for the koryakinp/fine-tune grayscale model); if False, color RGB."""
    base = Path(custom_dir)
    class_names = sorted(d.name for d in base.iterdir() if d.is_dir())
    if not class_names:
        raise FileNotFoundError(f"No class folders found in '{custom_dir}'.")
    exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    X, y = [], []
    for idx, c in enumerate(class_names):
        files = sorted({f for e in exts for f in (base / c).glob(e)})
        for f in files:
            if gray:
                g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if g is None:
                    continue
                g = cv2.resize(g, (img_size, img_size))
                img = np.stack([g, g, g], axis=-1)
            else:
                img = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(idx)
    return np.asarray(X, np.uint8), np.asarray(y, np.int32), class_names


def eval_tflite(model_bytes: bytes, X, y, n_eval=None):
    """Measure tflite model accuracy + average inference time (ms). Handles both float/int8 input/output."""
    # The XNNPACK delegate can fail to prepare on some ops (e.g. MobileNetV3 hard-swish),
    # so the measurement interpreter is created without the default delegates (deployment is separate)
    try:
        interp = tf.lite.Interpreter(
            model_content=model_bytes,
            experimental_op_resolver_type=(
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES))
    except Exception:
        interp = tf.lite.Interpreter(model_content=model_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp['quantization']
    out_scale, out_zp = out['quantization']

    n = len(X) if n_eval is None else min(n_eval, len(X))
    correct = 0
    import time
    # warmup
    for _ in range(3):
        sample = X[0:1].astype(np.float32)
        if inp['dtype'] in (np.int8, np.uint8):
            sample = np.round(sample / in_scale + in_zp).astype(inp['dtype'])
        else:
            sample = sample.astype(inp['dtype'])
        interp.set_tensor(inp['index'], sample)
        interp.invoke()

    t0 = time.perf_counter()
    for i in range(n):
        img = X[i:i + 1].astype(np.float32)                    # 0~255
        if inp['dtype'] in (np.int8, np.uint8):
            img = np.round(img / in_scale + in_zp).astype(inp['dtype'])
        else:
            img = img.astype(inp['dtype'])
        interp.set_tensor(inp['index'], img)
        interp.invoke()
        o = interp.get_tensor(out['index'])[0].astype(np.float32)
        if out['dtype'] in (np.int8, np.uint8):
            o = (o - out_zp) * out_scale
        if int(np.argmax(o)) == int(y[i]):
            correct += 1
    ms = (time.perf_counter() - t0) / n * 1000.0
    return correct / n, ms


def _convert_float(model):
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def _convert_int8(model, rep_gen, int8_io: bool):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if int8_io:
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


def main() -> None:
    parser = argparse.ArgumentParser(description='Finger model INT8 quantization + benchmark')
    parser.add_argument('--model', default='files/save/fingers_model.keras')
    parser.add_argument('--data-dir', default='files/fingers')
    parser.add_argument('--custom-dir', default=None,
                        help='Root of self-captured/real photo data (ImageFolder). Use the same as during training')
    parser.add_argument('--gray', action='store_true',
                        help='Grayscale->3 channel load (required when calibrating a model fine-tuned in grayscale)')
    parser.add_argument('--out-dir', default='files/save')
    parser.add_argument('--results', default='results/benchmark.csv')
    parser.add_argument('--rep-samples', type=int, default=200, help='Number of representative data samples (for calibration)')
    parser.add_argument('--eval-samples', type=int, default=1000, help='Number of samples for accuracy/speed measurement')
    parser.add_argument('--int8-io', action='store_true', help='Convert input/output to int8 as well')
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    def rel(p):
        return p if os.path.isabs(p) else str(root / p)

    model_path = rel(args.model)
    data_dir = rel(args.data_dir)
    out_dir = Path(rel(args.out_dir))
    results_path = Path(rel(args.results))
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'tensorflow {tf.__version__}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\nTrain it first with 'python train_fingers.py'.")

    # 1. Load model -> auto-detect input size (automatically adapts to whatever --img-size was used in training)
    model = keras.models.load_model(model_path)
    img_size = int(model.input_shape[1])
    if args.custom_dir:
        custom_dir = rel(args.custom_dir)
        X, y, _ = load_custom_dir(custom_dir, gray=args.gray, img_size=img_size)
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]
        n_test = max(1, int(len(X) * 0.15))
        Xte, yte = X[:n_test], y[:n_test]
        Xtr = X[n_test:]
    else:
        Xte, yte = load_split(data_dir, 'test', img_size)
        Xtr, _ = load_split(data_dir, 'train', img_size)
    print(f'model: {model_path} (input {img_size}px) / test: {Xte.shape} / train(rep): {Xtr.shape}')

    # 2. Representative data generator (raw 0~255 float32, (1,96,96,3))
    rep_idx = np.linspace(0, len(Xtr) - 1, min(args.rep_samples, len(Xtr))).astype(int)

    def rep_gen():
        for i in rep_idx:
            yield [Xtr[i:i + 1].astype(np.float32)]

    # 3. Convert
    print('\n[Convert] float tflite ...')
    float_bytes = _convert_float(model)
    print('[Convert] INT8 tflite ...')
    int8_bytes = _convert_int8(model, rep_gen, args.int8_io)

    float_path = out_dir / 'fingers_float.tflite'
    int8_path = out_dir / 'fingers_int8.tflite'
    float_path.write_bytes(float_bytes)
    int8_path.write_bytes(int8_bytes)

    # 4. Benchmark (accuracy/size/speed)
    keras_acc = float(model.evaluate(
        tf.data.Dataset.from_tensor_slices((Xte, yte)).batch(64)
        .map(lambda a, b: (tf.cast(a, tf.float32), b)), verbose=0)[1])

    rows = []
    for name, data, path in [('float', float_bytes, float_path),
                             ('int8', int8_bytes, int8_path)]:
        acc, ms = eval_tflite(data, Xte, yte, n_eval=args.eval_samples)
        rows.append({
            'model': name,
            'size_kb': round(len(data) / 1024, 1),
            'gzip_kb': round(len(gzip.compress(data)) / 1024, 1),
            'accuracy': round(acc, 4),
            'infer_ms': round(ms, 3),
            'path': str(path),
        })

    # 5. Print + save CSV
    print(f'\nKeras(baseline) test accuracy: {keras_acc:.4f}')
    print(f"{'model':<7}{'size_kb':>10}{'gzip_kb':>10}{'accuracy':>10}{'infer_ms':>10}")
    for r in rows:
        print(f"{r['model']:<7}{r['size_kb']:>10}{r['gzip_kb']:>10}"
              f"{r['accuracy']:>10}{r['infer_ms']:>10}")

    import csv
    with open(results_path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=['model', 'size_kb', 'gzip_kb',
                                                'accuracy', 'infer_ms', 'path'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nSaved: {float_path}')
    print(f'Saved: {int8_path}  <- for board deployment')
    print(f'Saved: {results_path}')


if __name__ == '__main__':
    main()
