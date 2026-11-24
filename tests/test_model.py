import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from car_box.cli import app
from car_box.core import load, read, split
from car_box.model import build, train

tf = pytest.importorskip("tensorflow")


def test_actual_small_stack_train_save_reload(tmp_path: Path) -> None:
    for name in ["a", "b"]:
        Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(
            tmp_path / f"{name}.png"
        )
    csv = tmp_path / "data.csv"
    csv.write_text(
        "image,xmin,ymin,xmax,ymax,group\na.png,0,0,16,16,v1\nb.png,0,0,32,32,v2\n"
    )
    training, validation = split(read(csv))
    x, _ = load(validation, 32, 32)
    model = build(32, 32, 2, 2)
    assert model(x).shape == (1, 4)
    scores = train(training, validation, tmp_path / "artifact", 1, 32, 32, 2, 2)
    assert scores["validation_images"] == 1 and 0 <= scores["validation_mean_iou"] <= 1
    assert np.isfinite(scores["validation_loss"])
    assert json.loads((tmp_path / "artifact" / "validation.json").read_text()) == scores
    restored = tf.keras.models.load_model(
        tmp_path / "artifact" / "model.keras", safe_mode=True
    )
    assert restored(x).shape == (1, 4)
    result = CliRunner().invoke(
        app,
        [
            "train",
            str(csv),
            str(tmp_path / "cli"),
            "--height",
            "32",
            "--width",
            "32",
            "--depth",
            "2",
            "--filters",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["validation_groups"] == 1
    with pytest.raises(ValueError):
        train(training, validation, tmp_path / "artifact")
    with pytest.raises(ValueError):
        train(training, training, tmp_path / "new")
    with pytest.raises(ValueError):
        train(training, validation, tmp_path / "new", 0)


@pytest.mark.parametrize(
    "args",
    [(15, 32, 2, 2), (32, 32, 0, 2), (32, 32, 11, 2), (32, 32, 2, 0), (32, 32, 2, 9)],
)
def test_bad_model(args: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError):
        build(*args)
