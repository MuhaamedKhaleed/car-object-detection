# Pretrained detector demo (real car photos)

The repository's single-box regression model is a baseline, not a useful
detector. This route runs a pinned, license-checked pretrained detector on the
same licensed Open Images photos so a fresh clone produces useful real output.

Model: `hustvl/yolos-tiny`, revision `95a90f3c189fbfca3bcfc6d7315b9e84d95dc2de`.
License field on the model card: Apache-2.0 (checked via the Hugging Face API
at fetch time). Generic COCO-trained detector, a separate baseline from this
repository's regression model; it is not trained here and its weights are not
this project's artifact. Model card: https://huggingface.co/hustvl/yolos-tiny

## Run

```sh
uv sync --locked --extra detector
uv run --no-sync python scripts/fetch_real_data.py --output ./real-source
uv run --no-sync python scripts/fetch_detector.py --output ./detector
uv run --no-sync python scripts/pretrained_demo.py --data ./real-source --model ./detector --output ./demo-out
```

`fetch_detector.py` downloads only the three pinned model files, checks SHA-256
(`model.safetensors` hash equals the Hugging Face LFS object id at that
revision) and refuses an existing destination. `pretrained_demo.py` downloads
the documented CVDF-hosted copies of the 24 selected photos, runs detection at
threshold 0.5, and writes `detections.json` plus an annotated
`detector-evidence.png` (green: annotation, red: predicted car boxes).

## Actual recorded result (25 Nov 2026, local CPU, torch 2.4.1+cpu, transformers 4.46.3)

- 24 images, 21 with at least one car detection at threshold 0.5.
- Mean best-car IoU vs the annotated box: 0.7484.
- Elapsed about 22 s on one CPU core after downloads.

A Kaggle run of the same scripts on a fresh clone is recorded in the project
notebook `car-kaggle-run` (private draft in the owner's Kaggle account).

## Limits

Generic COCO detector; COCO "car" is not this repository's annotation cohort,
so these numbers are not comparable with the regression-model IoU runs. The
21/24 and 0.7484 figures are a diagnostic on the same known 24-image
subset used by earlier development runs, not held-out generalization: the
pretrained model's COCO training data and its overlap with these photos were
not audited, and no threshold or model choice was tuned on these 24. Single
annotated car per image by selection. 24 development photos are not a
benchmark mAP. Detection rights: model Apache-2.0; images CC BY 2.0 with
per-image attribution preserved in `real-source/car-selected.json`; Open
Images annotations CC BY 4.0. No weights, photos or data are committed.
