"""Explicit CPU single-box baseline, never a multi-object detector."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .core import Box, iou, load


def backend() -> Any:
    """Import optional TensorFlow lazily and select CPU."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ValueError("Install uv sync --extra ml for TensorFlow.") from exc
    tf.config.set_visible_devices([], "GPU")
    return tf


def build(
    height: int = 380,
    width: int = 676,
    depth: int = 10,
    filters: int = 8,
    seed: int = 42,
) -> Any:
    """Original Conv/BN/pool stack, bounded sigmoid fractional xyxy outputs.

    Defaults retain original depth/width; small smoke depth is explicit. Four
    independent output values do not guarantee correct corner order.
    """
    if (
        not 16 <= height <= 1024
        or not 16 <= width <= 1024
        or not 1 <= depth <= 10
        or not 1 <= filters <= 8
    ):
        raise ValueError("Dimensions16..1024, depth1..10, filters1..8 required.")
    tf = backend()
    tf.keras.utils.set_random_seed(seed)
    layers = tf.keras.layers
    inputs = layers.Input((height, width, 3))
    x = inputs
    for i in range(depth):
        x = layers.Conv2D(filters * 2**i, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    output = layers.Dense(4, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def train(
    training: list[Box],
    validation: list[Box],
    output: Path,
    epochs: int = 1,
    height: int = 380,
    width: int = 676,
    depth: int = 10,
    filters: int = 8,
    seed: int = 42,
) -> dict[str, float | int]:
    """Explicit CPU fitting with group-disjoint validation and mean pair IoU."""
    if output.exists():
        raise ValueError("Output exists; choose a new directory.")
    if epochs < 1:
        raise ValueError("Epochs must be positive.")
    if {b.group for b in training} & {b.group for b in validation} or {
        b.image for b in training
    } & {b.image for b in validation}:
        raise ValueError("Image/source groups overlap between splits.")
    x, y = load(training, height, width)
    vx, vy = load(validation, height, width)
    model = build(height, width, depth, filters, seed)
    for _ in range(epochs):
        for start in range(0, len(x), 4):
            loss = float(
                model.train_on_batch(x[start : start + 4], y[start : start + 4])[0]
            )
            if not math.isfinite(loss):
                raise ValueError("Nonfinite training loss; nothing saved.")
    loss, mae = map(float, model.evaluate(vx, vy, verbose=0))
    prediction = model(vx, training=False).numpy()
    if not math.isfinite(loss) or not math.isfinite(mae):
        raise ValueError("Nonfinite validation metrics.")
    overlaps = [
        iou(
            (float(a[0]), float(a[1]), float(a[2]), float(a[3])),
            (float(b[0]), float(b[1]), float(b[2]), float(b[3])),
        )
        for a, b in zip(vy, prediction, strict=True)
    ]
    result = {
        "validation_loss": loss,
        "validation_mae": mae,
        "validation_mean_iou": sum(overlaps) / len(overlaps),
        "validation_images": len(vx),
        "validation_groups": len({b.group for b in validation}),
    }
    output.mkdir(parents=True)
    model.save(output / "model.keras")
    (output / "validation.json").write_text(json.dumps(result))
    return result
