# Single-Box Car Baseline

A strict local image/box pipeline with explicit CPU regression training. **This is a single-box baseline, not a multi-object detector**: it refuses multiple annotations for one image rather than silently deleting cars.

## Results (honest)

- Pretrained detector demo (pinned, hash-checked `hustvl/yolos-tiny`, Apache-2.0) on 24 licensed Open Images car photos: **21/24 photos had a car detected, mean best IoU 0.7484** vs annotations (25 Nov 2026). This is a generic pretrained model, not this repository's own network.
- The repository's own single-box regression model trains and runs end to end but is very weak at demo scale (mean IoU ~0.02 after 2 epochs). More epochs and photos are the knob; a 500-photo route with best-development tracking is in `docs/larger-training.md`.
- No held-out quality or default-network convergence was evaluated for the regression model; mean pair IoU/MSE/MAE are reported, not mAP.
- 26 tests, 97% statement coverage (reduced-stack CNN train/save/reload, coordinate conventions, group splitting). Coverage is engineering evidence, not detection quality.

## Quickstart (fresh clone, about 1 minute)

Needs Python 3.11 or 3.12 (locked wheels have no 3.13+ build), internet for the first install/downloads, ~2 GB free RAM.

```sh
uv sync --python 3.11 --locked --extra detector
uv run --no-sync python scripts/fetch_real_data.py --output ./real-source      # 24 Open Images car photos + boxes
uv run --no-sync python scripts/fetch_detector.py --output ./detector          # pinned Apache-2.0 COCO detector
uv run --no-sync python scripts/pretrained_demo.py --data ./real-source --model ./detector --output ./demo-out
```

Output: `demo-out/detections.json` and `detector-evidence.png`. To train the repository's own model instead: `uv sync --python 3.11 --locked --extra dev --extra ml`, then `uv run --no-sync python scripts/real_data.py --data ./real-source --manifest ./real-source/car-selected.json --epochs 2`.

## Limits

- The model cannot express zero/multiple cars, confidence or classes; invalid predicted corner order counts as IoU 0.
- `group` IDs keep source/video frames together in the split, but wrong IDs defeat it; validation groups are not an untouched final test.
- The default network (380x676 input, depth 10) preserves the original architecture and is extremely large; it was not fully trained. Real training needs a separate resource decision.
- Verified on Linux x86_64; the container is a data-check image without TensorFlow. Free hosted CI covers tests, lint, CLI and Docker build.
- The historical notebook's duplicate-dropping and generator bugs are fixed, but the full notebook was not rerun; the CLI is the supported path. No release, service, real images or models are published.

## License

MIT for the new project-owned code (see `LICENSE` and `LICENSE-SCOPE.md`). Photos, pretrained detector and historical code keep their own terms. Details and evidence: `docs/verification.md`, `docs/pretrained-detector.md`, `docs/real-data.md`, `CHANGELOG.md`.
