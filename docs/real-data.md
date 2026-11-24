# Real-data CPU development run

Source: https://storage.googleapis.com/openimages/web/factsfigures_v7.html . Actual recorded result: `real-data-output.json`. This is a small development run, not full training, production readiness or an untouched benchmark. No data/model artifacts are committed. Run from the repository root after installing the locked ML/dev extras. This script starts real CPU training only when explicitly invoked. Outputs use fixed new names under the supplied data folder and refuse rerunning into an existing artifact. Use a new folder for another run.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/real_data.py --data /path/to/prepared-source
```

Use a source manifest prepared from the official Open Images validation boxes and image-license metadata, not an unverified mirror. Supply `--manifest /path/to/car-selected.json`. Each record needs id, box (original normalized bbox CSV row) and attribution (original metadata row). Select sorted first24 images with exactly one car label /m/0k4j and no group/depiction/interior/truncated box. Metadata URLs: https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv and https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv . The script explicitly fetches reviewed CVDF-hosted copies and records actual downloaded hashes; they differ from originals. Annotations CC BY4, image metadata CC BY2 with author/landing/license preserved, provider no warranty. No photos committed. This baseline supports one box, not multi-car scenes, negatives or mAP. Split by imageID, not certified scene-disjoint.

## Actual limits

single-box regression, no negatives/multi-object mAP, image groups not scene-disjoint; resized hosted copies not originals

## Explicit download and preparation

The optional helper downloads only the documented primary-source files, checks SHA-256 and refuses an existing destination. It never runs during installation, import or tests. Source attribution/license cautions above still apply. No model/data is bundled. For an offline copy, add `--cached /path/to/source-files` with the same filenames; checksums are still required.

```sh
uv run --no-sync python scripts/fetch_real_data.py --output ./real-source
uv run --no-sync python scripts/real_data.py --data ./real-source --manifest ./real-source/car-selected.json
```

The hashes pin the exact small development sources used here, not a permission warranty or a quality benchmark. If a source changes, the helper fails rather than silently training on different data.

The original first24 run included some interior/truncated views. Visual review prompted a stricter source-metadata filter; real-data-output.json now records the rerun, still deliberately small and poor quality (mean IoU about0.0187). Lower score is reported, not hidden.

## Docker without a registry account

This repository supplies a Dockerfile, not a published registry image. Build locally once. The default remains the small invented-input wiring demo. No account, paid service or automatic source download is required by the image.

```sh
docker build -f Dockerfile.ml -t project-ml .
docker run --rm --network none project-ml
```

The image also includes `/app/scripts`. After explicit source preparation on the host, mount a new writable working directory and invoke the real script with `--entrypoint python`. Keep model/source mounts read-only when the script does not write beside them. The image uses uid10001, so grant that uid write access to the output folder, do not run the container as root to bypass this. Data/preprocessing scripts currently place artifacts beside source files; mount the working copy read-write and keep your original untouched. The environment disables Hugging Face downloads. Network-none is suitable only after sources/models are local; an explicit fetch helper needs network by design. CPU/dependency builds were checked on GitHub's Linux runner, not every laptop OS.
