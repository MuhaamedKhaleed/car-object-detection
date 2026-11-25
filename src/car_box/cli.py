"""Thin CLI for explicit single-box research, no automatic training."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
from dotenv import dotenv_values

from .core import load, read, split

app = typer.Typer(
    help="Single-box car baseline checks and CPU train, not multi-object detection."
)
logger = logging.getLogger(__name__)


def run(action: Callable[[], Any]) -> None:
    """JSON stdout, expected input errors stderr/exit2."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        typer.echo(json.dumps(action()))
    except (ValueError, OSError, TypeError) as exc:
        logger.error("%s", exc)
        raise typer.Exit(2) from exc


def resolve(data: Path | None, config: Path | None) -> Path:
    """Read explicit CSV or opt-in dotenv BOX_CSV."""
    if data:
        return data
    if config:
        if not config.is_file():
            raise ValueError("Config missing; supply existing dotenv.")
        value = dotenv_values(config).get("BOX_CSV")
        if value:
            return Path(value)
    raise ValueError("Supply --data or --config containing BOX_CSV.")


@app.command()
def check_data(data: Path | None = None, config: Path | None = None) -> None:
    """Validate every image/box and group split, refuse lost annotations."""

    def action() -> dict[str, int]:
        boxes = read(resolve(data, config))
        training, validation = split(boxes)
        load(boxes, 32, 32)
        return {
            "images": len(boxes),
            "train_images": len(training),
            "validation_images": len(validation),
            "groups": len({b.group for b in boxes}),
        }

    run(action)


@app.command()
def train(
    data: Path,
    output: Path,
    epochs: int = 1,
    height: int = 380,
    width: int = 676,
    depth: int = 10,
    filters: int = 8,
    seed: int = 42,
) -> None:
    """Explicit CPU fitting; defaults are large, smoke model knobs explicit."""

    def action() -> dict[str, float | int]:
        from .model import train as fit

        training, validation = split(read(data), seed=seed)
        return fit(
            training, validation, output, epochs, height, width, depth, filters, seed
        )

    run(action)


@app.command()
def predict(image: Path, model: Path) -> None:
    """Predict one car box from an explicit trusted local trained artifact."""

    def action() -> dict[str, Any]:
        from .model import predict as infer

        return infer(image, model)

    run(action)
