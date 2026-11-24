# Verification record

Original defects: silently retained only one box per image; generator accepted df/path but sampled global train/path; fixed raw shape assumption; categorical coordinate accuracy not meaningful. Notebook fix refuses repeated images, honors arguments, uses MAE, clears outputs and disables upload/unzip. Full notebook unrerun; historical callers need explicit df/path, so it is not represented as a turnkey path.

New package requires a single-box CSV with explicit source/video group IDs, finite in-bounds coordinates, RGB resize and fractional xyxy labels. This is a behavior/interface change, old BGR/raw-coordinate artifacts incompatible. Multi-car/negative images need a real detector, not annotation deletion. Supplied group IDs cannot be independently verified and content duplicates are not detected. Mean pair IoU is not mAP/recall.

## Local commands

- `uv sync --locked --extra dev --extra ml`.
- `TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run pytest --cov=car_box --cov-report=term-missing`:26passed,97% coverage.
- `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv run mypy src`, `uv run pre-commit run --all-files`:pass.
- `uv run python scripts/synthetic_demo.py`, then `uv run car-box check-data --data synthetic-demo/boxes.csv` and `uv run car-box check-data --config .env.example`:3images,2train,1validation,3groups.
- `TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run car-box train synthetic-demo/boxes.csv synthetic-demo/trained --epochs 1 --height 32 --width 32 --depth 2 --filters 2`:actual finite synthetic training, local save/reload tested.
- Module/console/help commands passed. Light install excludes TensorFlow and explicitly skips its backend module.

Default architecture retains input380x676,10 convolution/BN/pooling blocks doubling from8 filters through4096, then dense256/32. This is huge and not default-network training evidence. Synthetic smoke reduces dimensions/depth/width. Head changes to sigmoid fractional coordinates; independent corners can invert, counted as IoU0. Actual tests include backend forward/train/save/reload, CLI, coordinate bounds/normalization, group-disjoint split, malformed/missing data and IoU failure cases. No real dataset or mAP claim.

Hosted CI owns Docker build/run checks, no local Docker daemon claim. Multi-stage non-root image includes data checks only, no ML/dev dependencies. Public-only ordinary Linux workflow, no paid services or hosted real-data training. uv lock scoped Linuxx86_64, CI pip/direct pins not the full transitive lock.

## Phase notes and open limits

Typed installable src package, explicit CLI, real validation/logging/config, optionalML/dev groups and uv lock, actual tests/quality tooling, public-only CI/data-only container, truthful architecture/output and changelog, separate notebook bugfix and copy-only presentation. No stub helpers, fake history, main merge, tags, license changes or image/model/data publication.

No full historical notebook/default-network training, real-car held-out quality, multi-object detection, negative-image behavior, confidence score, production inference, scene/domain independence, external test cohort, cross-platform/GPU validation or atomic artifact writes. Large defaults can exhaust RAM/time; smoke settings do not validate full-scale use. Models are not published.
