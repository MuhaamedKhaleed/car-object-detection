# Real-data CPU development run

Source: https://storage.googleapis.com/openimages/web/factsfigures_v7.html . Actual recorded result: `real-data-output.json`. This is a small development run, not full training, production readiness or an untouched benchmark. No data/model artifacts are committed. Run from the repository root after installing the locked ML/dev extras. This script starts real CPU training only when explicitly invoked. Outputs use fixed new names under the supplied data folder and refuse rerunning into an existing artifact. Use a new folder for another run.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/real_data.py --data /path/to/prepared-source
```

Use a source manifest prepared from the official Open Images validation boxes and image-license metadata, not an unverified mirror. Supply `--manifest /path/to/car-selected.json`. Each record needs id, box (original normalized bbox CSV row) and attribution (original metadata row). Select sorted first24 images with exactly one car label /m/0k4j and no group/depiction box. Metadata URLs: https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv and https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv . The script explicitly fetches reviewed CVDF-hosted copies and records actual downloaded hashes; they differ from originals. Annotations CC BY4, image metadata CC BY2 with author/landing/license preserved, provider no warranty. No photos committed. This baseline supports one box, not multi-car scenes, negatives or mAP. Split by imageID, not certified scene-disjoint.

## Actual limits

single-box regression, no negatives/multi-object mAP, image groups not scene-disjoint; resized hosted copies not originals
