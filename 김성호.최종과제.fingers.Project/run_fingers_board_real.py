"""Real-time webcam finger count (1-5) recognition — execution code for board deployment.

Follows the structure of EX_04_Board_RPS_PreTrained_DenseNet.py but replaces it with 5 classes,
and automatically handles the model's input/output quantization info (float/int8).
  - Input size/format/dtype are read automatically from the model (input_details)
  - Since training is grayscale (3-channel replication) based, inference is matched the same way: grayscale->3-channel
  - On the board, tflite_runtime is used; if absent, tensorflow.lite is used automatically (for dev PC testing)

By default, it uses fingers_int8.tflite in the same folder as the script.
(If absent, it automatically falls back to fingers_float.tflite)

Usage examples:
    python run_fingers_board.py
    python run_fingers_board.py --model fingers_float.tflite --camera 0
    python run_fingers_board.py --color     # Input as color without grayscale conversion (when retrained in color)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Automatic inference runtime selection: ai_edge_litert(LiteRT, recommended) → tflite_runtime → tensorflow.lite
# All three share the same Interpreter API (allocate_tensors/get_*_details/set_tensor/invoke/get_tensor)
try:
    from ai_edge_litert.interpreter import Interpreter as _Interpreter

    def make_interpreter(model_path):
        return _Interpreter(model_path=model_path)
    BACKEND = 'ai_edge_litert'
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite

        def make_interpreter(model_path):
            return tflite.Interpreter(model_path=model_path)
        BACKEND = 'tflite_runtime'
    except ImportError:
        import tensorflow as tf

        def make_interpreter(model_path):
            return tf.lite.Interpreter(model_path=model_path)
        BACKEND = 'tensorflow.lite'


def load_labels(path: str):
    """Load labels.txt (one class name per line). If absent, default to 1-5."""
    p = Path(path)
    if p.exists():
        return [ln.strip() for ln in p.read_text(encoding='utf-8').splitlines() if ln.strip()]
    return ['1', '2', '3', '4', '5']


def setup_display() -> bool:
    """Determine whether a GUI window (cv2.imshow) can be shown, and if possible, correct DISPLAY.

    SSH/console (TTY) sessions usually have an empty DISPLAY, which causes cv2.imshow to abort
    the process with a Qt (xcb) error (cannot be caught as a Python exception). The reason the
    EX_04 example 'worked correctly' is that it was run from the board's desktop terminal
    (DISPLAY=:0); it is not a code difference but a difference in whether the execution session
    has a display.

    - If DISPLAY is empty and the board desktop is up (X socket /tmp/.X11-unix/X0),
      set DISPLAY to that display to show the window on the board monitor.
    - Test actual window creation in a child process; if connection fails, return False
      (in this case the caller falls back to headless → console output instead of abort).
    """
    if not os.environ.get('DISPLAY'):
        socket_dir = Path('/tmp/.X11-unix')
        socks = sorted(socket_dir.glob('X*')) if socket_dir.exists() else []
        if not socks:
            return False
        os.environ['DISPLAY'] = ':' + socks[0].name[1:]   # e.g. X0 -> ':0'
        xauth = Path.home() / '.Xauthority'
        if 'XAUTHORITY' not in os.environ and xauth.exists():
            os.environ['XAUTHORITY'] = str(xauth)

    # Check only whether window creation is possible in a child process, so as not to kill the parent
    probe = ("import cv2; cv2.namedWindow('p', cv2.WINDOW_AUTOSIZE); "
             "cv2.destroyAllWindows()")
    try:
        r = subprocess.run([sys.executable, '-c', probe], env=os.environ,
                           timeout=20, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description='Real-time webcam finger (1-5) recognition')
    parser.add_argument('--model', default='fingers_int8.tflite')
    parser.add_argument('--labels', default='labels.txt')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--color', action='store_true',
                        help='Input as color without grayscale conversion (when retrained in color)')
    parser.add_argument('--roi', type=float, default=0.0,
                        help='Central square ROI ratio (0-1). For self-captured models, use the same value as capture (e.g. 0.7)')
    parser.add_argument('--headless', action='store_true',
                        help='Output only prediction results to the console without showing a window (for environments without a display)')
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    def resolve_model(spec: str) -> str:
        """Resolve the model file path. Search in the order absolute/relative/script folder/filename,
        and if the default int8 model is absent, automatically fall back to the float model."""
        candidates = []
        p = Path(spec)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates += [Path.cwd() / spec, root / spec, root / p.name]
        # Automatic fallback (int8 <-> float) for when both variants are present in the same folder
        for a, b in (('int8', 'float'), ('float', 'int8')):
            if a in p.name:
                candidates.append(root / p.name.replace(a, b))
        for c in candidates:
            if c.exists():
                return str(c)
        tried = '\n  '.join(str(c) for c in candidates)
        raise FileNotFoundError(
            f'Model file not found: {spec}\nPaths searched:\n  {tried}')

    model_path = resolve_model(args.model)
    labels_path = args.labels if Path(args.labels).is_absolute() else str(root / args.labels)

    labels = load_labels(labels_path)
    # Display color per class (BGR): red/orange/yellow/green/blue
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)]

    # Model loading
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    H, W = int(inp['shape'][1]), int(inp['shape'][2])
    in_dtype = inp['dtype']
    in_scale, in_zp = inp['quantization']
    out_scale, out_zp = out['quantization']

    print(f'backend: {BACKEND}')
    print(f'model input: {(H, W)} dtype={np.dtype(in_dtype).name} '
          f'quant=(scale={in_scale}, zero={in_zp})')

    def predict(frame):
        # Crop only the central square ROI for classification (same conditions as capture → removes background/hand-size variation)
        if args.roi and args.roi > 0:
            h, w = frame.shape[:2]
            side = int(min(h, w) * args.roi)
            y0, x0 = (h - side) // 2, (w - side) // 2
            frame = frame[y0:y0 + side, x0:x0 + side]
        # Match the training domain: grayscale->3-channel (with the color option, keep RGB as is)
        if args.color:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (W, H))
        x = np.expand_dims(img, 0).astype(np.float32)            # (1,H,W,3), 0-255

        # Input dtype handling: if int8/uint8, quantize; if float, keep as is (preprocessing is inside the model)
        if in_dtype in (np.int8, np.uint8):
            x = np.round(x / in_scale + in_zp).astype(in_dtype)
        else:
            x = x.astype(in_dtype)

        interpreter.set_tensor(inp['index'], x)
        interpreter.invoke()
        o = interpreter.get_tensor(out['index'])[0].astype(np.float32)
        if out['dtype'] in (np.int8, np.uint8):
            o = (o - out_zp) * out_scale
        ans = int(np.argmax(o))
        conf = float(o[ans])
        return ans, conf

    # Decide display mode: if GUI is possible, use a window; otherwise headless (console). Prevents abort.
    headless = args.headless or not setup_display()
    if headless:
        print('Display mode: headless (console output) — no usable GUI display available.')
        print('  To show a window on the board monitor, run it from the board desktop terminal, or')
        print('  specify an X display. e.g.: DISPLAY=:0 python3 run_fingers_board.py')
        print('  Press Ctrl-C to exit.')
    else:
        print(f"Display mode: GUI window (DISPLAY={os.environ.get('DISPLAY')}) — press q to exit")

    # Camera setup (same as EX_04_Board)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera ({args.camera}).')

    start = time.time()
    last_log = 0.0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            ans, conf = predict(frame)
            now = time.time()
            fps = 1.0 / max(now - start, 1e-6)
            start = now

            if headless:
                # Print one updating line to the console about every 0.5 seconds
                if now - last_log >= 0.5:
                    last_log = now
                    print(f'\rFingers: {labels[ans]:>3}  conf={conf*100:5.1f}%  '
                          f'FPS={fps:4.1f}', end='', flush=True)
            else:
                if args.roi and args.roi > 0:
                    rh, rw = frame.shape[:2]
                    rs = int(min(rh, rw) * args.roi)
                    ry, rx = (rh - rs) // 2, (rw - rs) // 2
                    cv2.rectangle(frame, (rx, ry), (rx + rs, ry + rs), (0, 255, 0), 2)
                text = f'{labels[ans]} ({conf*100:.0f}%)'
                color = colors[ans % len(colors)]
                # Show the predicted number at the bottom-right, right-aligned (so it does not overlap FPS)
                fh, fw = frame.shape[:2]
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)
                org = (fw - tw - 10, fh - 10)
                cv2.putText(frame, text, org,
                            cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                # FPS is fixed at the top-left
                cv2.putText(frame, f'FPS: {fps:.1f}', (20, 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                cv2.imshow('fingers', frame)
                if cv2.waitKey(10) == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if headless:
            print()
        else:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
