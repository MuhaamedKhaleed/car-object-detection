# Single-Box Car Baseline

A strict local image/box pipeline with explicit CPU regression training. This is a **single-box baseline, not a multi-object detector**. It refuses multiple annotations for an image rather than silently deleting cars.

## Install and invented demo

Python3.11/3.12, Linuxx86_64, uv installed:

```sh
uv sync --locked --extra dev
uv run car-box --help
uv run python scripts/synthetic_demo.py
uv run car-box check-data --data synthetic-demo/boxes.csv
uv run car-box check-data --config .env.example
```

Actual output:

```json
{"images": 3, "train_images": 2, "validation_images": 1, "groups": 3}
```

Demo images are invented squares, not cars. The script refuses an existing demo folder. No dataset, pretrained weights, secret, zip extraction or download is included. `--config` explicitly reads `BOX_CSV` from a selected dotenv file; `--data` wins.

## Annotation contract

UTF-8 columns `image,xmin,ymin,xmax,ymax,group`. Images resolve relative to the CSV. Require finite pixel coordinates with positive area inside each image. A repeated image fails even if its boxes differ; do not delete real annotations to satisfy the baseline. Choose a genuine multi-object detector for multi-car images. The model cannot express zero/multiple cars, confidence, classes or object absence.

`group` is an explicit source/video ID. Seeded splitting keeps groups together to reduce adjacent-frame leakage, but incorrect IDs defeat it. No content or near-duplicate/scene identity detection exists. Validation groups are not an untouched final test. RGB images resize bilinearly and scale to [0,1]; labels normalize x by original width and y by original height. The old OpenCV BGR/raw-pixel path is deliberately changed, so old artifacts are incompatible.

## Explicit training

```sh
uv sync --locked --extra dev --extra ml
TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run car-box train synthetic-demo/boxes.csv synthetic-demo/trained --epochs 1 --height 32 --width 32 --depth 2 --filters 2
```

This exact small synthetic command was run. It saves local `model.keras` and `validation.json`, never overwrites an existing directory or starts cloud compute. Default input380x676, depth10, filters8 preserve the original convolution/BatchNorm/pooling stack and dense256/32 head; **the original default network is extremely large** and was not fully trained. Smoke tests reduce input/depth/width explicitly, not a claim of equivalent model quality. CPU-only. Real default training needs a separate resource decision.

Four independent sigmoid coordinates stay within [0,1] but can be misordered. Invalid predicted corner order counts as IoU0. Mean pair IoU, normalized-coordinate MSE and MAE are reported, not object-detection mAP, precision, recall or car-recognition accuracy. No real car dataset, held-out quality or default-network convergence was evaluated. There is no production inference service.

## Checks

```sh
TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run pytest --cov=car_box --cov-report=term-missing
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pre-commit run --all-files
uv run python -m car_box --help
```

26 meaningful tests,97% statement coverage, including actual reduced-stack CNN train/save/reload and explicit CLI, coordinate normalization, multiple-box rejection, group split, invalid images/coordinates and IoU conventions. Light-only install skips the TensorFlow test module explicitly. Coverage verifies engineering, not detection quality.

## Docker

```sh
docker build --tag car-box:ci .
docker run --rm car-box:ci --help
docker run --rm -v "$PWD/synthetic-demo:/data:ro" car-box:ci check-data --data /data/boxes.csv
```

Multi-stage slim non-root **data-check-only** runtime, no TensorFlow/dev packages. Hosted CI runs build and help/mounted CSV checks, not a local Docker-daemon claim. Public-repository-only ordinary Linux workflow, no paid API, hosted training or private billable runners. Linuxx86_64 uv lock; CI pip uses direct pins, not the complete transitive lock. GPU/Windows/macOS/ARM not verified.

## Structure and historical fixes

```text
CSV -> core.read -> image/box guards -> group split
                             -> RGB resize + fractional xyxy
                             -> explicit CPU model.train -> local artifact/metrics
Typer CLI -> JSON output / separate stderr errors
```

The notebook formerly dropped duplicate-image boxes and its generator ignored `df`/`path` arguments, sampling the global training frame/table instead. These fixes are separate from package additions. Coordinate categorical accuracy was replaced by MAE. Outputs cleared, online Colab upload/unzip disabled. **Full notebook remains unrerun**, with historical callers still requiring explicit generator arguments. The CLI is the supported verified path.

Next work: choose a suitable real detector for multi-car/negative examples; licensed grouped train/validation/frozen test; evaluate IoU/mAP, occlusion/scale/night slices and failures. Keep unverified claims out of portfolio summaries.

[Verification](docs/verification.md), [changelog](CHANGELOG.md), [copy-only presentation](docs/presentation.md). No main merge, release/tag, license change or real images/models/data publication.


## License scope

New project-owned package/engineering files use the owner's MIT default. See LICENSE and LICENSE-SCOPE.md. This is not permission to relicense historical notebooks, third-party code, datasets, models or weights. Their provenance and terms still need review. Earlier verification entries saying no license was added describe the initial pilot, not this follow-up.

## Checked synthetic wiring demo

See [commands, saved output and scope](docs/synthetic-demo.md). This is invented-input execution evidence, not model accuracy or a real-data gallery.

## Portable notebook follow-up

The root notebook now uses the supported local package with safe defaults and explicit data/training opt-ins. It replaces the platform-specific experiment rather than reproducing historical results. See `docs/notebook-migration.md` and the notebook settings. Earlier descriptions of live notebook downloads/hard-coded paths are historical and superseded. No dataset or weight rights are granted.

## Real-data CPU example

See [actual results, source terms and explicit download/preparation commands](docs/real-data.md). These are small development runs, not accuracy or production guarantees. No model/data is committed.

## Measured CPU quality follow-up

See [quality experiments, results and limits](docs/quality-cpu.md). These go beyond wiring smokes; some improve measured results, while others expose overfit or a stronger simple baseline. No quality guarantee.
