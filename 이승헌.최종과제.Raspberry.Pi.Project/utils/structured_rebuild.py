"""Tier 3: rebuild a physically smaller ResNet50 from structured-pruned (masked) model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.pruning_ablation import (
    apply_structured_channel_masks,
    evaluate_model,
    find_layer_by_suffix,
    finetune_model,
    get_block_shared_masks,
    structured_channel_stats,
    train_network_slimming_sparsity,
)

STAGE_BLOCKS = {2: 3, 3: 4, 4: 6, 5: 3}
STAGE_FILTERS = {2: 64, 3: 128, 4: 256, 5: 512}


def _is_quant_wrapper(layer) -> bool:
    return "QuantizeWrapper" in type(layer).__name__


def _get_conv_weights(layer) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (kernel, bias) from Conv2D or QuantizeWrapperV2."""
    if _is_quant_wrapper(layer):
        weights = layer.get_weights()
        kernel = np.array(weights[0])
        bias = np.array(weights[1]) if len(weights) > 1 and np.array(weights[1]).ndim == 1 else None
        return kernel, bias
    inner = layer.layer if hasattr(layer, "layer") else layer
    weights = inner.get_weights()
    if not weights:
        return None, None
    kernel = np.array(weights[0])
    bias = np.array(weights[1]) if len(weights) > 1 else None
    return kernel, bias


def _set_conv_weights(layer, kernel: np.ndarray, bias: Optional[np.ndarray] = None) -> None:
    if _is_quant_wrapper(layer):
        if bias is not None:
            layer.set_weights([kernel, bias])
        else:
            layer.set_weights([kernel])
        return
    inner = layer.layer if hasattr(layer, "layer") else layer
    if bias is not None:
        inner.set_weights([kernel, bias])
    else:
        inner.set_weights([kernel])


def _slice_conv_kernel(
    kernel: np.ndarray,
    in_idx: Optional[np.ndarray] = None,
    out_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    w = kernel
    if in_idx is not None:
        w = w[:, :, in_idx, :]
    if out_idx is not None:
        w = w[:, :, :, out_idx]
    return w


def _slice_bn_weights(bn_layer, keep_idx: np.ndarray) -> List[np.ndarray]:
    weights = [np.array(w) for w in bn_layer.get_weights()]
    return [w[keep_idx] for w in weights]


def _keep_indices_from_bn(bn_layer, eps: float = 1e-8) -> np.ndarray:
    gamma = np.array(bn_layer.get_weights()[0])
    return np.where(np.abs(gamma) > eps)[0]


def channel_plan_from_masks(masks: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Build rebuild channel plan from Network Slimming keep masks."""
    if "conv1_bn" not in masks:
        raise ValueError("conv1_bn mask not found")

    stem_idx = np.where(masks["conv1_bn"])[0]
    plan: Dict[str, object] = {
        "stem": int(len(stem_idx)),
        "stem_idx": stem_idx,
        "blocks": {},
    }

    for bn_name, keep in masks.items():
        if not bn_name.endswith("_1_bn") or bn_name == "conv1_bn":
            continue
        prefix = bn_name[: -len("_1_bn")]
        # bn1 governs conv1-out / conv2-in; bn2 governs conv2-out / conv3-in.
        # These are independent reducible widths in a bottleneck, so keep two idx.
        idx1 = np.where(keep)[0]
        mask2 = masks.get(f"{prefix}_2_bn", keep)
        idx2 = np.where(mask2)[0]
        plan["blocks"][prefix] = (int(len(idx1)), int(len(idx2)))
        plan[f"{prefix}_idx1"] = idx1
        plan[f"{prefix}_idx2"] = idx2
    return plan


def extract_channel_plan(model, eps: float = 1e-8) -> Dict[str, object]:
    """
    Read kept channels from BN gamma magnitudes (e.g. Tier-1 masked model).
    Prefer ``channel_plan_from_masks`` after L1 + ``get_block_shared_masks``.
    """
    conv1_bn = find_layer_by_suffix(model, "conv1_bn")
    if conv1_bn is None:
        raise ValueError("conv1_bn not found")
    stem_idx = _keep_indices_from_bn(conv1_bn, eps=eps)
    plan: Dict[str, object] = {"stem": int(len(stem_idx)), "stem_idx": stem_idx, "blocks": {}}

    for stage, n_blocks in STAGE_BLOCKS.items():
        for b in range(1, n_blocks + 1):
            prefix = f"conv{stage}_block{b}"
            bn1 = find_layer_by_suffix(model, f"{prefix}_1_bn")
            if bn1 is None:
                raise ValueError(f"{prefix}_1_bn not found")
            idx1 = _keep_indices_from_bn(bn1, eps=eps)
            bn2 = find_layer_by_suffix(model, f"{prefix}_2_bn")
            idx2 = _keep_indices_from_bn(bn2, eps=eps) if bn2 is not None else idx1
            plan["blocks"][prefix] = (int(len(idx1)), int(len(idx2)))
            plan[f"{prefix}_idx1"] = idx1
            plan[f"{prefix}_idx2"] = idx2
    return plan


def _block_io_channels(stage: int, block: int, stem: int) -> Tuple[int, int]:
    out_ch = STAGE_FILTERS[stage] * 4
    if block == 1:
        in_ch = stem if stage == 2 else STAGE_FILTERS[stage - 1] * 4
    else:
        in_ch = out_ch
    return in_ch, out_ch


def _unpack_inner(value) -> Tuple[int, int]:
    """blocks[prefix] is (k1, k2); tolerate a legacy single int."""
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def build_slim_resnet50(
    channel_plan: Dict[str, object],
    img_size: int = 64,
    num_classes: int = 3,
):
    """Build a smaller ResNet50 (fp32) matching the channel plan."""
    from tensorflow_model_optimization.python.core.keras.compat import keras

    stem = int(channel_plan["stem"])
    blocks = channel_plan["blocks"]

    inputs = keras.Input(shape=(img_size, img_size, 3), name="input")
    x = keras.applications.resnet.preprocess_input(inputs)

    x = keras.layers.ZeroPadding2D(padding=(3, 3), name="conv1_pad")(x)
    x = keras.layers.Conv2D(
        stem, 7, strides=2, use_bias=False, name="conv1_conv"
    )(x)
    x = keras.layers.BatchNormalization(name="conv1_bn")(x)
    x = keras.layers.Activation("relu", name="conv1_relu")(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1), name="pool1_pad")(x)
    x = keras.layers.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    for stage, n_blocks in STAGE_BLOCKS.items():
        for b in range(1, n_blocks + 1):
            prefix = f"conv{stage}_block{b}"
            inner_k1, inner_k2 = _unpack_inner(blocks[prefix])
            in_ch, out_ch = _block_io_channels(stage, b, stem)
            strides = 2 if b == 1 and stage > 2 else 1
            if b == 1 and stage == 2:
                strides = 1
            x = _slim_bottleneck(x, in_ch, inner_k1, inner_k2, out_ch, strides, prefix)

    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="dense")(x)
    return keras.Model(inputs, outputs, name="slim_resnet50")


def _slim_bottleneck(
    x, in_ch: int, inner_k1: int, inner_k2: int, out_ch: int, strides: int, name: str
):
    from tensorflow_model_optimization.python.core.keras.compat import keras

    if strides != 1 or in_ch != out_ch:
        shortcut = keras.layers.Conv2D(
            out_ch, 1, strides=strides, use_bias=False, name=f"{name}_0_conv"
        )(x)
        shortcut = keras.layers.BatchNormalization(name=f"{name}_0_bn")(shortcut)
    else:
        shortcut = x

    # inner_k1: conv1-out (= conv2-in), inner_k2: conv2-out (= conv3-in)
    y = keras.layers.Conv2D(inner_k1, 1, use_bias=False, name=f"{name}_1_conv")(x)
    y = keras.layers.BatchNormalization(name=f"{name}_1_bn")(y)
    y = keras.layers.Activation("relu", name=f"{name}_1_relu")(y)

    y = keras.layers.Conv2D(
        inner_k2, 3, strides=strides, padding="same", use_bias=False, name=f"{name}_2_conv"
    )(y)
    y = keras.layers.BatchNormalization(name=f"{name}_2_bn")(y)
    y = keras.layers.Activation("relu", name=f"{name}_2_relu")(y)

    y = keras.layers.Conv2D(out_ch, 1, use_bias=False, name=f"{name}_3_conv")(y)
    y = keras.layers.BatchNormalization(name=f"{name}_3_bn")(y)

    y = keras.layers.Add(name=f"{name}_add")([shortcut, y])
    return keras.layers.Activation("relu", name=f"{name}_out")(y)


def transfer_weights(source_model, target_model, channel_plan: Dict[str, object]) -> None:
    """Copy sliced weights from masked QAT model into compact fp32 model."""
    stem_idx = channel_plan["stem_idx"]

    # Stem
    src_conv = find_layer_by_suffix(source_model, "conv1_conv")
    tgt_conv = target_model.get_layer("conv1_conv")
    k, b = _get_conv_weights(src_conv)
    tgt_conv.set_weights([_slice_conv_kernel(k, out_idx=stem_idx)])

    src_bn = find_layer_by_suffix(source_model, "conv1_bn")
    target_model.get_layer("conv1_bn").set_weights(_slice_bn_weights(src_bn, stem_idx))

    for stage, n_blocks in STAGE_BLOCKS.items():
        for b in range(1, n_blocks + 1):
            prefix = f"conv{stage}_block{b}"
            inner_idx1 = channel_plan[f"{prefix}_idx1"]
            inner_idx2 = channel_plan[f"{prefix}_idx2"]
            in_ch, out_ch = _block_io_channels(stage, b, int(channel_plan["stem"]))
            strides = 2 if b == 1 and stage > 2 else 1
            if b == 1 and stage == 2:
                strides = 1
            has_proj = strides != 1 or in_ch != out_ch

            if has_proj:
                proj_in_idx = stem_idx if stage == 2 and b == 1 else None
                _transfer_conv_bn(
                    source_model,
                    target_model,
                    f"{prefix}_0_conv",
                    f"{prefix}_0_bn",
                    in_idx=proj_in_idx,
                    out_idx=None,
                )

            block_in_idx = stem_idx if stage == 2 and b == 1 else None
            _transfer_conv_bn(
                source_model,
                target_model,
                f"{prefix}_1_conv",
                f"{prefix}_1_bn",
                in_idx=block_in_idx,
                out_idx=inner_idx1,
            )
            _transfer_conv_bn(
                source_model,
                target_model,
                f"{prefix}_2_conv",
                f"{prefix}_2_bn",
                in_idx=inner_idx1,
                out_idx=inner_idx2,
            )
            _transfer_conv_bn(
                source_model,
                target_model,
                f"{prefix}_3_conv",
                f"{prefix}_3_bn",
                in_idx=inner_idx2,
                out_idx=None,
            )

    # Classifier head (GAP output dim unchanged = 2048)
    src_dense = find_layer_by_suffix(source_model, "dense")
    tgt_dense = target_model.get_layer("dense")
    k, b = _get_conv_weights(src_dense)
    tgt_dense.set_weights([k, b])


def _transfer_conv_bn(
    source_model,
    target_model,
    conv_suffix: str,
    bn_suffix: str,
    in_idx: Optional[np.ndarray],
    out_idx: Optional[np.ndarray],
) -> None:
    src_conv = find_layer_by_suffix(source_model, conv_suffix)
    tgt_conv = target_model.get_layer(conv_suffix)
    k, _ = _get_conv_weights(src_conv)
    new_k = _slice_conv_kernel(k, in_idx=in_idx, out_idx=out_idx)
    tgt_conv.set_weights([new_k])

    src_bn = find_layer_by_suffix(source_model, bn_suffix)
    tgt_bn = target_model.get_layer(bn_suffix)
    if out_idx is not None:
        tgt_bn.set_weights(_slice_bn_weights(src_bn, out_idx))
    else:
        tgt_bn.set_weights(src_bn.get_weights())


def save_rebuilt_artifacts(model, save_dir: str, tag: str) -> Dict[str, object]:
    """Save rebuilt fp32 .h5 and TFLite."""
    import tensorflow as tf

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / f"rebuilt_structured_{tag}.h5"
    tflite_path = out_dir / f"rebuilt_structured_{tag}.tflite"

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
        "params": model.count_params(),
    }


def _rebuild_and_finetune(
    source_model,
    plan: Dict[str, object],
    train_gen,
    test_data,
    test_labels,
    finetune_epochs: int,
    save_dir: Optional[str],
    tag: str,
    channel_prune_ratio: float,
    source_acc_after_mask: Optional[float] = None,
) -> Tuple[object, dict]:
    from tensorflow_model_optimization.python.core.keras.compat import keras

    print(f"Channel plan: stem={plan['stem']}, blocks={plan['blocks']}")

    slim = build_slim_resnet50(plan)
    print(
        f"Slim model params: {slim.count_params():,} "
        f"(source: {source_model.count_params():,})"
    )

    transfer_weights(source_model, slim, plan)

    slim.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    _, acc_before = slim.evaluate(test_data, test_labels, verbose=0)
    print(f"  after weight transfer: val_acc={acc_before:.4f}")
    if source_acc_after_mask is not None:
        print(f"  (masked source before rebuild: val_acc={source_acc_after_mask:.4f})")

    if finetune_epochs > 0:
        print(f"  Step 4/4: fine-tune ({finetune_epochs} epochs)")
        finetune_model(slim, train_gen, test_data, test_labels, epochs=finetune_epochs)

    _, acc_after = slim.evaluate(test_data, test_labels, verbose=0)
    print(f"  after fine-tune: val_acc={acc_after:.4f}")

    result = {
        "channel_plan": plan,
        "channel_prune_ratio": channel_prune_ratio,
        "acc_masked_source": source_acc_after_mask,
        "acc_after_transfer": float(acc_before),
        "acc_after_ft": float(acc_after),
        "params": slim.count_params(),
        "source_params": source_model.count_params(),
    }
    if save_dir:
        result.update(save_rebuilt_artifacts(slim, save_dir, tag))
        print(f"  saved: {result.get('h5_path')}")
        print(f"  TFLite: {result.get('tflite_size_mb')} MB")

    return slim, result


def run_structured_rebuild(
    load_model_fn,
    train_gen,
    test_data,
    test_labels,
    channel_percent: float = 0.3,
    l1_sparsity_epochs: int = 3,
    l1_lambda: float = 1e-5,
    l1_learning_rate: float = 1e-4,
    finetune_epochs: int = 10,
    save_dir: Optional[str] = None,
    tag: Optional[str] = None,
    train_data=None,
    train_labels=None,
    batch_size: int = 32,
) -> Tuple[object, dict]:
    """
    Tier 3 pipeline (single pass):

    1. Load original QAT model
    2. L1 sparsity on BN gamma
    3. Channel plan from |gamma| + apply masks on source (required for transfer)
    4. Rebuild slim ResNet50 + weight transfer + fine-tune + save
    """
    if tag is None:
        tag = f"channel{int(round(channel_percent * 100)):02d}"

    print(f"\n=== Tier 3 Structured Rebuild: channel={channel_percent:.0%} ===")
    model = load_model_fn()

    print(
        f"  Step 1/4: L1 sparsity training "
        f"({l1_sparsity_epochs} epochs, lambda={l1_lambda}, lr={l1_learning_rate})"
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

    print("  Step 2/4: channel plan from |gamma| + apply masks")
    masks = get_block_shared_masks(model, channel_percent)
    _, _, ch_prune_ratio = structured_channel_stats(masks)
    plan = channel_plan_from_masks(masks)
    print(f"  channel prune ratio: {ch_prune_ratio:.1%}")
    apply_structured_channel_masks(model, masks)
    _, acc_masked = evaluate_model(model, test_data, test_labels)
    print(f"  masked source (before rebuild): val_acc={acc_masked:.4f}")

    print("  Step 3/4: rebuild slim network + weight transfer")
    return _rebuild_and_finetune(
        model,
        plan,
        train_gen,
        test_data,
        test_labels,
        finetune_epochs,
        save_dir,
        tag,
        ch_prune_ratio,
        source_acc_after_mask=acc_masked,
    )


def rebuild_from_pruned_model(
    pruned_model,
    train_gen,
    test_data,
    test_labels,
    finetune_epochs: int = 5,
    save_dir: Optional[str] = None,
    tag: str = "channel30",
) -> Tuple[object, dict]:
    """Legacy: rebuild from an already masked Tier-1 pruned model."""
    plan = extract_channel_plan(pruned_model)
    print("  Step: rebuild slim network + weight transfer (from Tier-1 pruned model)")
    return _rebuild_and_finetune(
        pruned_model,
        plan,
        train_gen,
        test_data,
        test_labels,
        finetune_epochs,
        save_dir,
        tag,
        0.0,
    )
