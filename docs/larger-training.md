---
id: larger-training
type: doc
aliases: [kaggle-training, quality-gpu]
---
# Larger training run (Kaggle, free tier)

`scripts/quality_gpu.py` trains the repository's own single-box regression
model on 500 real Open Images photos (400 train / 100 development, seeded
grouped split by imageID), with best-development tracking against the
mean-box baseline. It is a larger engineering run of the existing model, not
a benchmark: no negatives, no multi-object mAP, and the development split
selects the best checkpoint (no untouched test set).

## Verified result

Run 2026-11-25 on a free Kaggle CPU session (no accelerator attached;
TensorFlow 2.21.0, torch 2.11.0+cpu, transformers 5.20.2, Python 3.13.16)
from a fresh `git clone --depth 1 --branch car-engineering-pilot`, so an
outsider running the documented commands gets these outputs:

| metric | value |
| --- | --- |
| images | 500 (400 train / 100 development) |
| epochs | 30 |
| best development mean IoU | 0.6462 |
| training mean-box development IoU | 0.6212 |
| elapsed | 670.5 s (CPU) |

Development mean IoU per epoch moved 0.003 (epoch 1) to 0.636 (epoch 30),
crossing the mean-box baseline between epochs 25 and 30. Outputs:
`report.json`, `best.keras`, `training-evidence.png` (development photos
with annotation and predicted boxes). Evidence status: the notebook's saved
private version preserves the rendered cell outputs (printed report and
grids); the raw output files are not attached to that version, and a
durable-artifact save plus a reload-and-predict check of `best.keras`
through the repository's own predict route is still open.

## Data and rights

Open Images validation, single-annotated-car subset: images CC BY 2.0 with
per-image attribution preserved in `real-source/car-selected.json`,
annotations CC BY 4.0. The script downloads from the pinned public URLs at
run time; no images, weights or checkpoints are committed.

## Limits

Single-box regression; split is by imageID, not scene-disjoint; hosted
resized copies are used, not originals; development-selected best with no
held-out test, so 0.6462 is an optimistic engineering signal, not a
generalization estimate.
