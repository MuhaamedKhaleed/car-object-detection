# Measured no-cost CPU quality experiment

Same24development images,19train/5dev;120epochs improves bestdevIoU.3005 but trainingmeanbox baseline.6136 is better. No untouchedtest,mAP or usefulvisuallearning claim.7.4sec518MiB. Input boxes.csv is created by real_data.py in a separate workingfolder.

Run only explicitly, with previously prepared local permitted data and a new output. No automatic downloads or import-time execution. Exact measured result is quality-output.json. This is a reproducible development experiment, not a productionquality guarantee.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/quality_cpu.py --data LOCAL_PREPARED_SOURCE --output NEW_OUTPUT
```

Test sets described as finalized have now been observed. Future choices must not be justified with their scores as if they remained unseen. Retain source terms/attribution from real-data.md. No model or raw text/photos committed.
