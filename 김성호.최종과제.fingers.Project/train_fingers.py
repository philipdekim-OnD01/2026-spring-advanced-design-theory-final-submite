"""Finger (1-5) classification — transfer-learning trainer (accuracy-first, selectable backbone).

Transfer-learns from real captured color data (files/fingers_real/<1-5>/*.jpg, square)
using an ImageNet-pretrained backbone. A single run performs both stages:
  [Stage 1] feature extraction — freeze base, train head (Dense)
  [Stage 2] fine-tune — fine-tune base (BatchNorm frozen, low lr)

Deployment contract (run_fingers_board*.py): input NxNx3, 0-255 (preprocessing inside the model), argmax over N classes.
Per-backbone preprocessing (all included inside the model -> the board feeds 0-255 as-is, no code changes):
  - mobilenetv2        : Rescaling [-1,1]
  - mobilenetv3*/effv2 : in-model preprocessing (include_preprocessing, 0-255 input)

Usage examples:
  python train_fingers.py                                       # default mobilenetv3large @160
  python train_fingers.py --backbone efficientnetv2b0 --img-size 192
  python train_fingers.py --backbone mobilenetv2 --alpha 1.0 --img-size 160
  python train_fingers.py --gray                                # train in grayscale (for grayscale deployment)
"""

import argparse
import glob
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

import collections

import cv2
import numpy as np
import tensorflow as tf

keras = tf.keras

_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')


def load_imagefolder(data_dir: str, img_size: int, gray: bool = False):
    """Load <data_dir>/<class>/*.{jpg,png} -> (X_uint8 3ch, y, class_names)."""
    base = Path(data_dir)
    class_names = sorted(d.name for d in base.iterdir() if d.is_dir())
    if not class_names:
        raise FileNotFoundError(
            f"No class folders (1-5) found in '{data_dir}'.")
    X, y = [], []
    print('Number of photos per class:')
    for idx, c in enumerate(class_names):
        files = sorted({f for e in _EXTS for f in (base / c).glob(e)})
        cnt = 0
        for f in files:
            if gray:
                g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if g is None:
                    continue
                g = cv2.resize(g, (img_size, img_size))
                img = np.stack([g, g, g], axis=-1)
            else:
                im = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if im is None:
                    continue
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                img = cv2.resize(im, (img_size, img_size))
            X.append(img)
            y.append(idx)
            cnt += 1
        print(f'  {c}: {cnt} images')
    return np.asarray(X, np.uint8), np.asarray(y, np.int32), class_names


def stratified_split(y, val_frac: float, seed: int = 42):
    """Split off a val_frac fraction per class into the validation set (keeps class balance)."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        te += list(idx[:n_val])
        tr += list(idx[n_val:])
    rng.shuffle(tr)
    rng.shuffle(te)
    return np.array(tr), np.array(te)


def make_augmenter():
    return keras.Sequential([
        keras.layers.RandomFlip('horizontal'),       # left/right of the hand is irrelevant to the count
        keras.layers.RandomRotation(0.12),
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.RandomZoom(0.15),
        keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
        keras.layers.RandomContrast(0.15),
    ], name='augmenter')


def to_ds(X, y, batch, augment, augmenter):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        ds = ds.shuffle(min(2048, len(X)))
    ds = ds.batch(batch).map(lambda a, b: (tf.cast(a, tf.float32), b),
                             num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda a, b: (augmenter(a, training=True), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(backbone: str, img_size: int, alpha: float, n_classes: int):
    """Build the model with per-backbone preprocessing included inside it. Returns (model, base_layer)."""
    name = backbone.lower()
    shape = (img_size, img_size, 3)
    inputs = keras.Input(shape)
    if name == 'mobilenetv2':
        x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
        base = keras.applications.MobileNetV2(input_shape=shape, include_top=False,
                                              weights='imagenet', alpha=alpha)
    elif name == 'mobilenetv3large':
        x = inputs
        base = keras.applications.MobileNetV3Large(input_shape=shape, include_top=False,
                                                   weights='imagenet', include_preprocessing=True)
    elif name == 'mobilenetv3small':
        x = inputs
        base = keras.applications.MobileNetV3Small(input_shape=shape, include_top=False,
                                                   weights='imagenet', include_preprocessing=True)
    elif name == 'efficientnetv2b0':
        x = inputs
        base = keras.applications.EfficientNetV2B0(input_shape=shape, include_top=False,
                                                   weights='imagenet', include_preprocessing=True)
    else:
        raise ValueError(f'Unknown backbone: {backbone}')
    base.trainable = False
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inputs, out, name=f'fingers_{name}'), base


def main() -> None:
    ap = argparse.ArgumentParser(description='Finger (1-5) transfer learning (accuracy-first, selectable backbone)')
    ap.add_argument('--data-dir', default='files/fingers_real', help='ImageFolder(<dir>/<1~5>)')
    ap.add_argument('--backbone', default='mobilenetv2',
                    choices=['mobilenetv2', 'mobilenetv3large', 'mobilenetv3small', 'efficientnetv2b0'],
                    help='For Pi deployment, mobilenetv2 is recommended (XNNPACK compatible). The others are experimental (possible TFLite/deployment issues)')
    ap.add_argument('--alpha', type=float, default=1.0, help='mobilenetv2 width(0.35/0.5/0.75/1.0/1.3/1.4)')
    ap.add_argument('--img-size', type=int, default=160)
    ap.add_argument('--gray', action='store_true', help='train in grayscale (for grayscale deployment)')
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--epochs-head', type=int, default=30)
    ap.add_argument('--epochs-finetune', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr-head', type=float, default=1e-3)
    ap.add_argument('--lr-finetune', type=float, default=1e-4)
    ap.add_argument('--no-augment', action='store_true')
    ap.add_argument('--model-out', default='files/save/fingers_model.keras')
    ap.add_argument('--labels-out', default='files/save/labels.txt')
    args = ap.parse_args()

    root = Path(__file__).resolve().parent

    def rel(p):
        return p if os.path.isabs(p) else str(root / p)

    data_dir = rel(args.data_dir)
    model_out = Path(rel(args.model_out))
    labels_out = Path(rel(args.labels_out))
    model_out.parent.mkdir(parents=True, exist_ok=True)

    gpus = tf.config.list_physical_devices('GPU')
    print(f'tensorflow {tf.__version__} / GPU: {gpus if gpus else "none (CPU)"}')
    print(f'backbone={args.backbone} img_size={args.img_size} alpha={args.alpha} gray={args.gray}')

    # 1. Load data + class-balanced split
    X, y, class_names = load_imagefolder(data_dir, args.img_size, gray=args.gray)
    tr, te = stratified_split(y, args.val_frac)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    print(f'classes={class_names} / train {Xtr.shape} / val {Xte.shape}')

    aug = make_augmenter()
    train_ds = to_ds(Xtr, ytr, args.batch_size, not args.no_augment, aug)
    val_ds = to_ds(Xte, yte, args.batch_size, False, aug)

    # 2. Model + stage 1 (head)
    model, base = build_model(args.backbone, args.img_size, args.alpha, len(class_names))
    model.compile(optimizer=keras.optimizers.Adam(args.lr_head),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(f'\n[Stage 1] head training (base frozen) — params {model.count_params():,}')
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor='val_accuracy', patience=8, restore_best_weights=True)])

    # 3. Stage 2 fine-tune (fine-tune base, BN frozen)
    if args.epochs_finetune > 0:
        base.trainable = True
        for l in base.layers:
            if isinstance(l, keras.layers.BatchNormalization):
                l.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(args.lr_finetune),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print('\n[Stage 2] fine-tune (fine-tune base, BN frozen)')
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune, verbose=2,
                  callbacks=[
                      keras.callbacks.EarlyStopping(
                          monitor='val_accuracy', patience=12, restore_best_weights=True),
                      keras.callbacks.ReduceLROnPlateau(
                          monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
                  ])

    # 4. Evaluation + per-class confusion
    probs = model.predict(np.asarray(Xte, np.float32), verbose=0)
    preds = probs.argmax(1)
    acc = float((preds == yte).mean())
    print(f'\nVal accuracy: {acc:.4f}  (n={len(yte)})')
    wrong = [(class_names[a], class_names[b]) for a, b in zip(yte.tolist(), preds.tolist()) if a != b]
    if wrong:
        print('Confusion (actual->predicted, errors):')
        for (a, b), n in sorted(collections.Counter(wrong).items()):
            print(f'  {a}->{b}: {n}')
    else:
        print('All validation samples correct.')

    # 5. Save
    model.save(str(model_out))
    labels_out.write_text('\n'.join(class_names), encoding='utf-8')
    print(f'Saved model -> {model_out}')
    print(f'Saved labels-> {labels_out}')
    gray_flag = ' --gray' if args.gray else ''
    print(f'\nNext (quantization): python quantize_fingers.py --model {args.model_out} '
          f'--custom-dir {args.data_dir}{gray_flag}')


if __name__ == '__main__':
    main()
