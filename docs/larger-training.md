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

Runs on 2026-11-25 on a free Kaggle CPU session (TensorFlow 2.21.0, torch
2.11.0+cpu, transformers 5.20.2, Python 3.13.16) from a fresh
`git clone --depth 1 --branch car-engineering-pilot`. Two runs of the same
command are recorded; they are kept separate and differ:

| metric | first run | second run |
| --- | --- | --- |
| images | 500 (400 train / 100 development) | 500 (400 train / 100 development) |
| epochs | 30 | 30 |
| best development mean IoU | 0.6462 | 0.6616 |
| training mean-box development IoU | 0.6212 | 0.6212 |
| elapsed | 670.5 s | 336.15 s |

The difference between the two runs is not explained. Development mean IoU
per epoch moved from about 0.003 (epoch 1) to 0.634 (epoch 30) in the second
run, with its best (0.6616) at epoch 29.

Outputs: `report.json`, `best.keras`, `training-evidence.png` (development
photos with annotation and predicted boxes). Evidence status: the second run
was saved through the notebook's durable Output, which holds `best.keras`,
`report.json` and the grids. A fresh public-branch reload of that
`best.keras` through the repository's own predict route ran on Kaggle CPU
with exit code 0 and gave IoU 0.9449 on one development sample image. That
is a single example, not a benchmark. The first run's saved version kept only
the rendered report and grids.

## Data and rights

Open Images validation, single-annotated-car subset: images CC BY 2.0 with
per-image attribution preserved in `real-source/car-selected.json`,
annotations CC BY 4.0. The script downloads from the pinned public URLs at
run time; no images, weights or checkpoints are committed.

## Limits

Single-box regression; split is by imageID, not scene-disjoint; hosted
resized copies are used, not originals; development-selected best with no
held-out test, so 0.6462 and 0.6616 are optimistic engineering signals, not a
generalization estimate.
