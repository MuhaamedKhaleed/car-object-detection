"""Validate single-box annotations, normalize by actual image size and split groups."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@dataclass(frozen=True)
class Box:
    """One image, pixel xyxy box and explicit video/source group."""

    image: Path
    xyxy: tuple[float, float, float, float]
    group: str


def read(path: Path) -> list[Box]:
    """Read image,xmin,ymin,xmax,ymax,group CSV. Never drop extra boxes."""
    result = []
    seen = set()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        headers = rows.fieldnames or []
        if len(headers) != len(set(headers)) or not {
            "image",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "group",
        } <= set(headers):
            raise ValueError(
                "CSV needs unique image,xmin,ymin,xmax,ymax,group headers."
            )
        for i, row in enumerate(rows, 2):
            if None in row or any(v is None or not v.strip() for v in row.values()):
                raise ValueError(f"Missing/malformed fields at row{i}.")
            image = (path.parent / row["image"].strip()).resolve()
            if image in seen:
                raise ValueError(
                    "Multiple boxes/repeated image: single-box baseline cannot "
                    "represent them. Use a detector or review annotations."
                )
            seen.add(image)
            try:
                coords = tuple(float(row[k]) for k in ["xmin", "ymin", "xmax", "ymax"])
            except ValueError as exc:
                raise ValueError(f"Nonnumeric coordinates at row{i}.") from exc
            a, b, c, d = coords
            with Image.open(image) as img:
                width, height = img.size
                img.verify()
            if (
                not all(math.isfinite(x) for x in coords)
                or not 0 <= a < c <= width
                or not 0 <= b < d <= height
            ):
                raise ValueError(f"Box outside image or degenerate at row{i}.")
            result.append(Box(image, (a, b, c, d), row["group"].strip()))
    if not result:
        raise ValueError("No annotation rows.")
    return result


def split(
    boxes: list[Box], fraction: float = 0.2, seed: int = 42
) -> tuple[list[Box], list[Box]]:
    """Keep explicit source/video groups together, based on supplied IDs."""
    groups = sorted({b.group for b in boxes})
    if len(groups) < 2 or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Need at least two groups and fraction in (0,1).")
    random.Random(seed).shuffle(groups)
    n = max(1, min(len(groups) - 1, math.ceil(len(groups) * fraction)))
    val = set(groups[:n])
    return [b for b in boxes if b.group not in val], [
        b for b in boxes if b.group in val
    ]


def load(
    boxes: list[Box], height: int = 380, width: int = 676
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """RGB float32 resize and fractional xyxy labels, not raw-size coordinate mix."""
    if not boxes or not 16 <= height <= 1024 or not 16 <= width <= 1024:
        raise ValueError("Nonempty boxes and dimensions16..1024 required.")
    images = []
    coords = []
    for box in boxes:
        with Image.open(box.image) as image:
            w, h = image.size
            images.append(
                np.asarray(
                    image.convert("RGB").resize(
                        (width, height), Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                )
                / 255
            )
            a, b, c, d = box.xyxy
            coords.append([a / w, b / h, c / w, d / h])
    return np.stack(images), np.asarray(coords, dtype=np.float32)


def iou(
    actual: tuple[float, float, float, float],
    predicted: tuple[float, float, float, float],
) -> float:
    """Single-pair IoU, invalid predicted order gives0, not mAP/detection recall."""
    if not all(math.isfinite(x) for x in (*actual, *predicted)):
        raise ValueError("Coordinates must be finite.")
    a, b, c, d = actual
    e, f, g, h = predicted
    if a >= c or b >= d:
        raise ValueError("Ground-truth box must have positive area.")
    if e >= g or f >= h:
        return 0.0
    intersection = max(0, min(c, g) - max(a, e)) * max(0, min(d, h) - max(b, f))
    return float(intersection / ((c - a) * (d - b) + (g - e) * (h - f) - intersection))
