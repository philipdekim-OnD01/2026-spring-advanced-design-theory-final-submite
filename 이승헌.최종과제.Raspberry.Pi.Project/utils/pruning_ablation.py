"""Tier-1 pruning ablation helpers (unstructured vs structured channel masking)."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _is_kernel_layer(layer) -> bool:
    from tensorflow_model_optimization.python.core.keras.compat import keras

    return isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense))


def unwrap_kernel_layer(layer):
    """Return inner Conv2D/Dense if wrapped (e.g. QuantizeWrapperV2)."""
    if hasattr(layer, "layer") and _is_kernel_layer(layer.layer):
        return layer.layer
    if _is_kernel_layer(layer):
        return layer
    return None


def _kernel_weight_index(weights) -> Optional[int]:
    """Find kernel index in get_weights(); skip 1D QAT quant params."""
    best_idx = None
    best_ndim = 0
    for i, w in enumerate(weights):
        arr = np.array(w)
        if arr.ndim >= 2 and arr.ndim >= best_ndim:
            best_idx = i
            best_ndim = arr.ndim
    return best_idx


def _get_kernel_array(layer) -> Optional[np.ndarray]:
    weights = layer.get_weights()
    if not weights:
        return None
    idx = _kernel_weight_index(weights)
    if idx is None:
        return None
    return np.array(weights[idx])


def _set_kernel_array(layer, kernel: np.ndarray) -> None:
    weights = layer.get_weights()
    if not weights:
        return
    idx = _kernel_weight_index(weights)
    if idx is None:
        return
    weights = list(weights)
    weights[idx] = kernel
    layer.set_weights(weights)


def _get_bn_gamma(bn_layer) -> Optional[np.ndarray]:
    weights = bn_layer.get_weights()
    if not weights:
        return None
    return np.array(weights[0])


def _is_bn_layer(layer) -> bool:
    return "BatchNormalization" in type(layer).__name__


def find_layer_by_suffix(model, suffix: str):
    for layer in model.layers:
        if layer.name == suffix:
            return layer
    for layer in model.layers:
        if layer.name.endswith(suffix):
            return layer
    for layer in model.layers:
        if suffix in layer.name:
            return layer
    return None


def count_weight_sparsity(model) -> Tuple[int, int, float]:
    """Return (zero_weights, total_weights, sparsity_ratio)."""
    zero_count = 0
    total_count = 0
    for layer in model.layers:
        inner = unwrap_kernel_layer(layer)
        if inner is None:
            continue
        w = _get_kernel_array(inner)
        if w is None:
            continue
        total_count += w.size
        zero_count += int(np.sum(w == 0))
    if total_count == 0:
        return 0, 0, 0.0
    return zero_count, total_count, zero_count / total_count


def unstructured_magnitude_prune(model, sparsity: float) -> float:
    """Zero out the smallest ``sparsity`` fraction of weights globally."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1).")

    magnitudes = []
    layers_with_kernels = []
    for layer in model.layers:
        inner = unwrap_kernel_layer(layer)
        if inner is None:
            continue
        w = _get_kernel_array(inner)
        if w is None:
            continue
        magnitudes.append(np.abs(w).ravel())
        layers_with_kernels.append(inner)

    if not magnitudes:
        return 0.0

    threshold = np.percentile(np.concatenate(magnitudes), sparsity * 100.0)
    for inner in layers_with_kernels:
        w = _get_kernel_array(inner)
        if w is None:
            continue
        mask = np.abs(w) >= threshold
        _set_kernel_array(inner, w * mask)

    _, _, ratio = count_weight_sparsity(model)
    return ratio


def _mask_from_gamma(gamma: np.ndarray, percent: float) -> np.ndarray:
    n = len(gamma)
    n_prune = int(n * percent)
    if n_prune <= 0:
        return np.ones(n, dtype=bool)
    threshold = np.sort(np.abs(gamma))[n_prune]
    keep = np.abs(gamma) > threshold
    if keep.sum() == 0:
        keep[int(np.argmax(np.abs(gamma)))] = True
    return keep


def get_block_shared_masks(model, percent: float) -> Dict[str, np.ndarray]:
    """
    Network Slimming style channel masks on ResNet bottleneck BN layers.

    Each bottleneck BN gets its OWN mask from its OWN gamma ranking:
    ``_1_bn`` masks conv1-out / conv2-in, ``_2_bn`` masks conv2-out / conv3-in.
    (Sharing one mask across both BNs prunes conv2-out by an unrelated ranking.)
    """
    masks: Dict[str, np.ndarray] = {}
    block_prefixes = set()

    for layer in model.layers:
        if not _is_bn_layer(layer):
            continue
        if layer.name.endswith("_1_bn"):
            block_prefixes.add(layer.name[: -len("_1_bn")])

    for prefix in sorted(block_prefixes):
        for suffix in ("_1_bn", "_2_bn"):
            bn = find_layer_by_suffix(model, f"{prefix}{suffix}")
            if bn is None:
                continue
            gamma = _get_bn_gamma(bn)
            if gamma is None:
                continue
            masks[f"{prefix}{suffix}"] = _mask_from_gamma(gamma, percent)

    conv1_bn = find_layer_by_suffix(model, "conv1_bn")
    if conv1_bn is not None:
        gamma = _get_bn_gamma(conv1_bn)
        if gamma is not None:
            masks["conv1_bn"] = _mask_from_gamma(gamma, percent)

    return masks


def report_gamma_sparsity(model, masks: Dict[str, np.ndarray]) -> dict:
    """
    Pre-prune sanity check for Network Slimming: if the L1 phase actually worked,
    the channels we are about to prune should have |gamma| ~ 0. If pruned |gamma|
    is still large, the L1 sparsity step did nothing and accuracy will collapse.
    """
    kept_g, pruned_g = [], []
    for bn_name, keep in masks.items():
        bn = find_layer_by_suffix(model, bn_name)
        if bn is None:
            continue
        gamma = _get_bn_gamma(bn)
        if gamma is None or len(gamma) != len(keep):
            continue
        ag = np.abs(gamma)
        kept_g.append(ag[keep])
        pruned_g.append(ag[~keep])

    kept = np.concatenate(kept_g) if kept_g else np.array([0.0])
    pruned = np.concatenate(pruned_g) if pruned_g else np.array([0.0])
    all_g = np.concatenate([kept, pruned])
    report = {
        "pruned_gamma_mean": float(pruned.mean()),
        "pruned_gamma_max": float(pruned.max()),
        "kept_gamma_mean": float(kept.mean()),
        "frac_gamma_below_005": float(np.mean(all_g < 0.05)),
    }
    print(
        f"  gamma check | pruned |g| mean={report['pruned_gamma_mean']:.3f} "
        f"max={report['pruned_gamma_max']:.3f} | "
        f"kept |g| mean={report['kept_gamma_mean']:.3f} | "
        f"frac|g|<0.05={report['frac_gamma_below_005']:.2f}"
    )
    if report["pruned_gamma_max"] > 0.3:
        print(
            "  [WARN] pruned channels still have large gamma -> L1 sparsity weak. "
            "Increase l1_lambda / l1_sparsity_epochs, or lower prune percent."
        )
    return report


def structured_channel_stats(masks: Dict[str, np.ndarray]) -> Tuple[int, int, float]:
    total = sum(len(v) for v in masks.values())
    kept = sum(int(v.sum()) for v in masks.values())
    pruned = total - kept
    ratio = pruned / total if total else 0.0
    return kept, total, ratio


def _bn_gamma_l1_penalty(model):
    """Sum of |gamma| over all BatchNormalization layers (Network Slimming)."""
    import tensorflow as tf

    penalty = tf.constant(0.0, dtype=tf.float32)
    for layer in model.layers:
        if not _is_bn_layer(layer):
            continue
        gamma = getattr(layer, "gamma", None)
        if gamma is not None:
            penalty = penalty + tf.reduce_sum(tf.abs(gamma))
    return penalty


def _l1_train_step(self, data):
    """Custom train_step: task loss + L1 penalty on BN gamma (Network Slimming)."""
    import tensorflow as tf

    x, y = data
    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        loss = loss + self._l1_lambda * _bn_gamma_l1_penalty(self)
    grads = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}


def _build_l1_training_data(
    train_data,
    train_labels,
    train_gen,
    batch_size: int,
):
    """Prefer in-memory arrays + tf.data (no augmentation) for faster L1 phase."""
    import tensorflow as tf

    if train_data is not None and train_labels is not None:
        ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        ds = ds.shuffle(min(len(train_labels), 2048), seed=1)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
    return train_gen


def train_network_slimming_sparsity(
    model,
    train_gen,
    test_data,
    test_labels,
    epochs: int = 5,
    l1_lambda: float = 1e-5,
    learning_rate: float = 1e-5,
    train_data=None,
    train_labels=None,
    batch_size: int = 32,
) -> None:
    """
    Network Slimming step 2: fine-tune with L1 penalty on BN gamma.
    Loss = task_loss + l1_lambda * sum(|gamma|)

    Uses model.fit (progress bar) instead of a slow per-batch .numpy() loop.
    Pass train_data/train_labels to skip ImageDataGenerator augmentation (~2x faster).
    """
    from tensorflow_model_optimization.python.core.keras.compat import keras

    training_data = _build_l1_training_data(
        train_data, train_labels, train_gen, batch_size
    )
    using_arrays = train_data is not None and train_labels is not None
    if using_arrays:
        print(f"  L1 training data: in-memory arrays (batch_size={batch_size}, no augmentation)")
    else:
        print("  L1 training data: ImageDataGenerator (augmentation on)")

    # The base model freezes BatchNormalization (trainable=False). If left frozen,
    # gamma is NOT in trainable_variables, so the L1 penalty has no gradient path
    # and never sparsifies anything (the whole point of Network Slimming). Unfreeze
    # BN for the L1 phase so gamma can actually be driven toward zero, then restore.
    bn_layers = [l for l in model.layers if _is_bn_layer(l)]
    bn_trainable_state = [l.trainable for l in bn_layers]
    for l in bn_layers:
        l.trainable = True
    n_unfrozen = sum(1 for s in bn_trainable_state if not s)
    print(f"  BN unfrozen for L1 phase: {n_unfrozen}/{len(bn_layers)} layers")

    original_train_step = model.train_step
    model._l1_lambda = l1_lambda
    model.train_step = types.MethodType(_l1_train_step, model)
    try:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        n_gamma = sum(1 for v in model.trainable_variables if "gamma" in v.name)
        print(f"  gamma tensors in trainable_variables: {n_gamma}")
        history = model.fit(
            training_data,
            epochs=epochs,
            validation_data=(test_data, test_labels),
            verbose=1,
        )
    finally:
        model.train_step = original_train_step
        for l, s in zip(bn_layers, bn_trainable_state):
            l.trainable = s

    reg_val = float(_bn_gamma_l1_penalty(model).numpy()) * l1_lambda
    val_accs = history.history.get("val_accuracy", [])
    if val_accs:
        print(
            f"  L1 sparsity done: final val_acc={val_accs[-1]:.4f}, "
            f"L1_reg={reg_val:.6f}"
        )


def _zero_bn_gamma_beta(bn_layer, keep_mask: np.ndarray) -> None:
    """Network Slimming: zero gamma/beta only (keep BN running stats)."""
    weights = bn_layer.get_weights()
    if len(weights) < 2:
        return
    gamma = np.array(weights[0]).copy()
    beta = np.array(weights[1]).copy()
    if gamma.shape[0] == len(keep_mask):
        gamma[~keep_mask] = 0.0
        beta[~keep_mask] = 0.0
    bn_layer.set_weights([gamma, beta, *weights[2:]])


def _zero_conv_out_channels(conv_layer, keep_mask: np.ndarray) -> None:
    w = _get_kernel_array(conv_layer)
    if w is None:
        return
    w = w.copy()
    if w.ndim == 4:
        if w.shape[-1] != len(keep_mask):
            return
        w[:, :, :, ~keep_mask] = 0.0
    elif w.ndim == 2:
        if w.shape[-1] != len(keep_mask):
            return
        w[:, ~keep_mask] = 0.0
    else:
        return
    _set_kernel_array(conv_layer, w)


def _zero_conv_in_channels(conv_layer, keep_mask: np.ndarray) -> None:
    w = _get_kernel_array(conv_layer)
    if w is None:
        return
    w = w.copy()
    if w.ndim == 4:
        if w.shape[-2] != len(keep_mask):
            return
        w[:, :, ~keep_mask, :] = 0.0
    elif w.ndim == 2:
        if w.shape[0] != len(keep_mask):
            return
        w[~keep_mask, :] = 0.0
    else:
        return
    _set_kernel_array(conv_layer, w)


def _apply_block_mask(
    model, prefix: str, mask1: np.ndarray, mask2: np.ndarray
) -> None:
    """
    ResNet50 bottleneck dataflow: conv1 -> bn1 -> conv2 -> bn2 -> conv3.
    mask1 (bn1): bn1, conv1-out, conv2-in.
    mask2 (bn2): bn2, conv2-out, conv3-in.
    """
    bn1 = find_layer_by_suffix(model, f"{prefix}_1_bn")
    if bn1 is not None:
        _zero_bn_gamma_beta(bn1, mask1)
    bn2 = find_layer_by_suffix(model, f"{prefix}_2_bn")
    if bn2 is not None:
        _zero_bn_gamma_beta(bn2, mask2)

    conv1 = unwrap_kernel_layer(find_layer_by_suffix(model, f"{prefix}_1_conv"))
    if conv1 is not None:
        _zero_conv_out_channels(conv1, mask1)

    conv2 = unwrap_kernel_layer(find_layer_by_suffix(model, f"{prefix}_2_conv"))
    if conv2 is not None:
        _zero_conv_in_channels(conv2, mask1)
        _zero_conv_out_channels(conv2, mask2)

    conv3 = unwrap_kernel_layer(find_layer_by_suffix(model, f"{prefix}_3_conv"))
    if conv3 is not None:
        _zero_conv_in_channels(conv3, mask2)


def apply_structured_channel_masks(model, masks: Dict[str, np.ndarray]) -> None:
    """Apply channel masks (Tier 1: shape unchanged, pruned channels zeroed)."""
    if "conv1_bn" in masks:
        keep = masks["conv1_bn"]
        bn = find_layer_by_suffix(model, "conv1_bn")
        if bn is not None:
            _zero_bn_gamma_beta(bn, keep)
        conv1 = unwrap_kernel_layer(find_layer_by_suffix(model, "conv1_conv"))
        if conv1 is not None:
            _zero_conv_out_channels(conv1, keep)

    seen_blocks = set()
    for bn_name in masks:
        if not bn_name.endswith("_1_bn") or bn_name == "conv1_bn":
            continue
        prefix = bn_name[: -len("_1_bn")]
        if prefix in seen_blocks:
            continue
        seen_blocks.add(prefix)
        mask1 = masks[bn_name]
        mask2 = masks.get(f"{prefix}_2_bn", mask1)
        _apply_block_mask(model, prefix, mask1, mask2)


def finetune_model(
    model,
    train_gen,
    test_data,
    test_labels,
    epochs: int = 3,
    learning_rate: float = 1e-5,
):
    from tensorflow_model_optimization.python.core.keras.compat import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model.fit(
        train_gen,
        epochs=epochs,
        validation_data=(test_data, test_labels),
        verbose=1,
    )


def evaluate_model(model, test_data, test_labels) -> Tuple[float, float]:
    loss, acc = model.evaluate(test_data, test_labels, verbose=0)
    return float(loss), float(acc)


def _ratio_tag(ratio: float) -> str:
    return f"{int(round(ratio * 100)):02d}"


def artifact_stem(method: str, ratio: float) -> str:
    tag = _ratio_tag(ratio)
    if method == "unstructured":
        return f"pruned_unstructured_sparsity{tag}"
    if method == "structured":
        return f"pruned_structured_channel{tag}"
    raise ValueError(f"unknown method: {method}")


def save_pruned_artifacts(model, save_dir: str, method: str, ratio: float) -> Dict[str, object]:
    """Save fine-tuned QAT model (.h5) and convert to TFLite (.tflite)."""
    import tensorflow as tf
    from tensorflow_model_optimization.quantization.keras import quantize_scope

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact_stem(method, ratio)
    h5_path = out_dir / f"{stem}.h5"
    tflite_path = out_dir / f"{stem}.tflite"

    with quantize_scope():
        model.save(str(h5_path))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Op 버전 호환성: 구형 TFLite도 로드 가능하도록
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    tflite_bytes = converter.convert()
    tflite_path.write_bytes(tflite_bytes)

    return {
        "h5_path": str(h5_path),
        "tflite_path": str(tflite_path),
        "tflite_size_mb": round(len(tflite_bytes) / (1024 * 1024), 2),
    }


def _result_row(
    method: str,
    ratio: float,
    acc_before_ft: float,
    acc_after_ft: float,
    weight_sparsity: float,
    channel_prune_ratio: float,
    model,
    save_dir: Optional[str],
) -> dict:
    row = {
        "method": method,
        "ratio": ratio,
        "acc_before_ft": acc_before_ft,
        "acc_after_ft": acc_after_ft,
        "weight_sparsity": weight_sparsity,
        "channel_prune_ratio": channel_prune_ratio,
        "params": model.count_params(),
        "h5_path": None,
        "tflite_path": None,
        "tflite_size_mb": None,
    }
    if save_dir:
        row.update(save_pruned_artifacts(model, save_dir, method, ratio))
    return row


def run_unstructured_ablation(
    load_model_fn,
    train_gen,
    test_data,
    test_labels,
    ratios: Optional[List[float]] = None,
    finetune_epochs: int = 3,
    save_dir: Optional[str] = None,
) -> List[dict]:
    """Magnitude-based unstructured pruning ablation."""
    if ratios is None:
        ratios = [0.3, 0.5, 0.7]

    results: List[dict] = []
    for sparsity in ratios:
        print(f"\n=== Unstructured pruning: sparsity={sparsity:.0%} ===")
        model = load_model_fn()
        unstructured_magnitude_prune(model, sparsity)
        _, acc_b = evaluate_model(model, test_data, test_labels)
        print(f"  before fine-tune: val_acc={acc_b:.4f}")
        finetune_model(model, train_gen, test_data, test_labels, epochs=finetune_epochs)
        _, acc_a = evaluate_model(model, test_data, test_labels)
        _, _, weight_sp = count_weight_sparsity(model)
        results.append(
            _result_row(
                "unstructured",
                sparsity,
                acc_b,
                acc_a,
                weight_sp,
                0.0,
                model,
                save_dir,
            )
        )
    return results


def run_structured_ablation(
    load_model_fn,
    train_gen,
    test_data,
    test_labels,
    percents: Optional[List[float]] = None,
    l1_sparsity_epochs: int = 5,
    l1_lambda: float = 1e-5,
    l1_learning_rate: float = 1e-4,
    finetune_epochs: int = 5,
    save_dir: Optional[str] = None,
    train_data=None,
    train_labels=None,
    batch_size: int = 32,
) -> List[dict]:
    """
    Network Slimming structured pruning ablation.

    1. L1 sparsity training on BN gamma
    2. Channel prune by |gamma|
    3. Fine-tune
    """
    if percents is None:
        percents = [0.3, 0.5]

    results: List[dict] = []
    for percent in percents:
        print(f"\n=== Structured pruning (Network Slimming): channel={percent:.0%} ===")
        model = load_model_fn()
        print(
            f"  Step 1/3: L1 sparsity training ({l1_sparsity_epochs} epochs, "
            f"lambda={l1_lambda}, lr={l1_learning_rate})"
        )
        train_network_slimming_sparsity(
            model,
            train_gen,
            test_data,
            test_labels,
            epochs=l1_sparsity_epochs,
            l1_lambda=l1_lambda,
            learning_rate=l1_learning_rate,
            train_data=train_data,
            train_labels=train_labels,
            batch_size=batch_size,
        )
        print("  Step 2/3: channel pruning by |gamma|")
        masks = get_block_shared_masks(model, percent)
        report_gamma_sparsity(model, masks)
        apply_structured_channel_masks(model, masks)
        _, _, ch_prune_ratio = structured_channel_stats(masks)
        _, acc_b = evaluate_model(model, test_data, test_labels)
        print(f"  before fine-tune: val_acc={acc_b:.4f}")
        print(f"  Step 3/3: fine-tune ({finetune_epochs} epochs)")
        finetune_model(model, train_gen, test_data, test_labels, epochs=finetune_epochs)
        _, acc_a = evaluate_model(model, test_data, test_labels)
        _, _, weight_sp = count_weight_sparsity(model)
        results.append(
            _result_row(
                "structured",
                percent,
                acc_b,
                acc_a,
                weight_sp,
                ch_prune_ratio,
                model,
                save_dir,
            )
        )
    return results


def run_tier1_ablation(
    load_model_fn,
    train_gen,
    test_data,
    test_labels,
    baseline_acc: float,
    unstructured_ratios: Optional[List[float]] = None,
    structured_percents: Optional[List[float]] = None,
    finetune_epochs: int = 3,
    l1_sparsity_epochs: int = 5,
    l1_lambda: float = 1e-5,
    structured_finetune_epochs: Optional[int] = None,
    save_dir: Optional[str] = None,
    train_data=None,
    train_labels=None,
    batch_size: int = 32,
) -> List[dict]:
    """Run both ablations (convenience wrapper). Prefer separate cell functions."""
    if structured_finetune_epochs is None:
        structured_finetune_epochs = max(finetune_epochs, 5)

    results: List[dict] = [
        {
            "method": "baseline",
            "ratio": 0.0,
            "acc_before_ft": baseline_acc,
            "acc_after_ft": baseline_acc,
            "weight_sparsity": 0.0,
            "channel_prune_ratio": 0.0,
            "params": None,
            "h5_path": None,
            "tflite_path": None,
            "tflite_size_mb": None,
        }
    ]
    results.extend(
        run_unstructured_ablation(
            load_model_fn,
            train_gen,
            test_data,
            test_labels,
            ratios=unstructured_ratios,
            finetune_epochs=finetune_epochs,
            save_dir=save_dir,
        )
    )
    results.extend(
        run_structured_ablation(
            load_model_fn,
            train_gen,
            test_data,
            test_labels,
            percents=structured_percents,
            l1_sparsity_epochs=l1_sparsity_epochs,
            l1_lambda=l1_lambda,
            finetune_epochs=structured_finetune_epochs,
            save_dir=save_dir,
            train_data=train_data,
            train_labels=train_labels,
            batch_size=batch_size,
        )
    )
    return results


def results_to_dataframe(results: List[dict]):
    import pandas as pd

    return pd.DataFrame(results)
