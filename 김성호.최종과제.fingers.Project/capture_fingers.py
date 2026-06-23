"""Tool for directly capturing finger (1-5) training images with a webcam.

Fill your hand inside the central ROI (square) box on screen and press keys 1-5 to save under that class.
Inference (run_fingers_board*.py --roi --color) uses the same ROI/color, so the training-inference conditions match.
-> Compared to koryakinp (dark noisy background / grayscale), this produces data tailored to the real webcam domain.

Storage structure (ImageFolder, color):
    files/fingers_custom/1/*.png ... files/fingers_custom/5/*.png

Usage examples:
    python capture_fingers.py
    python capture_fingers.py --out-dir files/fingers_custom --roi 0.7 --camera 0

Controls:
    1-5 : save the current ROI as that finger count (hold to save continuously)
    u   : undo the last save (delete the file)
    q   : quit
Recommended: 150-300 images per class, captured while slightly varying hand position/angle/distance/lighting.
"""

import argparse
from pathlib import Path

import cv2

CLASSES = ['1', '2', '3', '4', '5']


def main() -> None:
    ap = argparse.ArgumentParser(description='Capture webcam finger (1-5) training images')
    ap.add_argument('--out-dir', default='files/fingers_custom')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--roi', type=float, default=0.7, help='Central square ROI ratio (0-1)')
    ap.add_argument('--img-size', type=int, default=96, help='Save size (px)')
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else root / args.out_dir
    for c in CLASSES:
        (out_dir / c).mkdir(parents=True, exist_ok=True)

    def count(c):
        return len(list((out_dir / c).glob('*.png')))

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera ({args.camera}).')

    print('Controls: 1-5 save / u undo / q quit')
    last_saved = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            side = int(min(h, w) * args.roi)
            y0, x0 = (h - side) // 2, (w - side) // 2
            roi = frame[y0:y0 + side, x0:x0 + side].copy()   # for saving (original before drawing the box)

            # Screen overlay: ROI box + per-class running count + instructions
            disp = frame.copy()
            cv2.rectangle(disp, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)
            counts = '  '.join(f'{c}:{count(c)}' for c in CLASSES)
            cv2.putText(disp, counts, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)
            cv2.putText(disp, '1-5:save  u:undo  q:quit', (10, h - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
            cv2.imshow('capture', disp)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            if key == ord('u') and last_saved is not None and last_saved.exists():
                last_saved.unlink()
                print(f'\nUndo: {last_saved.name}')
                last_saved = None
                continue
            for d in CLASSES:
                if key == ord(d):
                    img = cv2.resize(roi, (args.img_size, args.img_size))
                    fn = out_dir / d / f'{d}_{count(d):05d}.png'
                    cv2.imwrite(str(fn), img)
                    last_saved = fn
                    print(f'\rSaved {fn.name} (class {d}: {count(d)} images)', end='', flush=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print('\nPer-class count:', {c: count(c) for c in CLASSES})


if __name__ == '__main__':
    main()
