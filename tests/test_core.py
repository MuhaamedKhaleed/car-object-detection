from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from car_box.core import iou, load, read, split


def fixture(tmp_path: Path) -> Path:
    for name in ["a", "b", "c"]:
        Image.fromarray(np.full((40, 80, 3), 128, dtype=np.uint8)).save(
            tmp_path / f"{name}.png"
        )
    path = tmp_path / "boxes.csv"
    path.write_text(
        "image,xmin,ymin,xmax,ymax,group\na.png,20,10,60,30,video1\nb.png,0,0,40,20,video1\nc.png,0,0,80,40,video2\n"
    )
    return path


def test_load_normalize_group_split(tmp_path: Path) -> None:
    boxes = read(fixture(tmp_path))
    training, validation = split(boxes)
    assert {b.group for b in training}.isdisjoint(b.group for b in validation)
    assert (training, validation) == split(boxes)
    x, y = load(boxes, 16, 32)
    assert x.shape == (3, 16, 32, 3) and y.shape == (3, 4)
    assert x.dtype == np.float32 and y.dtype == np.float32
    assert y[0].tolist() == [0.25, 0.25, 0.75, 0.75]
    assert y[2].tolist() == [0, 0, 1, 1]
    assert x.mean() == pytest.approx(128 / 255)


@pytest.mark.parametrize(
    "row",
    [
        "a.png,0,0,0,20,v",
        "a.png,-1,0,20,20,v",
        "a.png,0,0,81,40,v",
        "a.png,0,0,80,41,v",
        "a.png,nan,0,20,20,v",
        "a.png,hello,0,20,20,v",
        "a.png,0,0,20,20,",
        "missing.png,0,0,20,20,v",
        "a.png,0,0,20,20,v,extra",
    ],
)
def test_bad_annotations(tmp_path: Path, row: str) -> None:
    fixture(tmp_path)
    path = tmp_path / "bad.csv"
    path.write_text("image,xmin,ymin,xmax,ymax,group\n" + row + "\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        read(path)


def test_never_drop_extra_boxes(tmp_path: Path) -> None:
    path = fixture(tmp_path)
    path.write_text(path.read_text() + "a.png,1,1,10,10,video1\n")
    with pytest.raises(ValueError, match="Multiple"):
        read(path)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "image,xmin\na,0\n",
        "image,xmin,ymin,xmax,ymax,group,image\n",
        "image,xmin,ymin,xmax,ymax,group\na.png,0,0\n",
        "image,xmin,ymin,xmax,ymax,group\n",
    ],
)
def test_invalid_csv(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        read(path)


def test_split_load_guards(tmp_path: Path) -> None:
    boxes = read(fixture(tmp_path))
    for fraction in [0, 1, -1, float("nan")]:
        with pytest.raises(ValueError):
            split(boxes, fraction)
    with pytest.raises(ValueError):
        split(boxes[:1])
    with pytest.raises(ValueError):
        load([])
    with pytest.raises(ValueError):
        load(boxes, 15, 32)


def test_iou() -> None:
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0
    assert iou((0, 0, 1, 1), (0, 0, 0.5, 1)) == 0.5
    assert iou((0, 0, 1, 1), (1, 1, 0, 0)) == 0
    with pytest.raises(ValueError):
        iou((0, 0, 0, 1), (0, 0, 1, 1))
    with pytest.raises(ValueError):
        iou((0, 0, 1, 1), (float("nan"), 0, 1, 1))
