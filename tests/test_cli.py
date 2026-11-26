import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from car_box.cli import app

runner = CliRunner()


def test_help_lazy() -> None:
    code = "import sys;import car_box.cli;assert 'tensorflow' not in sys.modules"
    assert (
        subprocess.run([sys.executable, "-c", code], capture_output=True).returncode
        == 0
    )
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert runner.invoke(app, ["train", "--help"]).exit_code == 0


def test_csv_config_errors(tmp_path: Path) -> None:
    for name in ["a", "b"]:
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(
            tmp_path / f"{name}.png"
        )
    data = tmp_path / "boxes.csv"
    data.write_text(
        "image,xmin,ymin,xmax,ymax,group\na.png,0,0,16,16,v1\nb.png,0,0,16,16,v2\n"
    )
    config = tmp_path / ".env"
    config.write_text(f"BOX_CSV={data}\n")
    result = runner.invoke(app, ["check-data", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "images": 2,
        "train_images": 1,
        "validation_images": 1,
        "groups": 2,
    }
    config.write_text("OTHER=none\n")
    assert (
        runner.invoke(
            app, ["check-data", "--data", str(data), "--config", str(config)]
        ).exit_code
        == 0
    )
    for args in [
        [],
        ["--config", str(config)],
        ["--config", str(tmp_path / "missing")],
        ["--data", str(tmp_path / "missing")],
    ]:
        result = runner.invoke(app, ["check-data", *args])
        assert result.exit_code == 2
        assert "Traceback" not in result.output
