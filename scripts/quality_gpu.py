"""Explicit larger Open Images training run (Kaggle/Colab GPU or CPU).

Selects more single-car validation images than the 24-image development route,
downloads the documented CVDF-hosted copies with recorded hashes, and trains
the repository regression model with best-development tracking against the
training-mean box baseline. Uses TensorFlow's available device (GPU when the
runtime provides one). This is a larger measured experiment, not an mAP,
multi-object or production-quality claim. No import-time work or downloads.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Folder from fetch_real_data.py")
    parser.add_argument("--output", required=True, help="New output folder")
    parser.add_argument("--images", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import csv
    import hashlib
    import json
    import time
    import urllib.request
    from collections import defaultdict
    from pathlib import Path

    root = Path(args.data)
    out = Path(args.output)
    if out.exists():
        parser.error("Output exists; use a new directory.")
    out.mkdir(parents=True)

    boxes = defaultdict(list)
    with (root / "car-boxes.csv").open() as source:
        for row in csv.DictReader(source):
            if row["LabelName"] == "/m/0k4j":
                boxes[row["ImageID"]].append(row)
    ids = sorted(
        k
        for k, rows in boxes.items()
        if len(rows) == 1
        and rows[0]["IsGroupOf"] == "0"
        and rows[0]["IsDepiction"] == "0"
        and rows[0]["IsInside"] == "0"
        and rows[0]["IsTruncated"] == "0"
    )[: args.images]
    with (root / "car-attribution.csv").open() as source:
        credits = {
            row["ImageID"]: row
            for row in csv.DictReader(source)
            if row["ImageID"] in ids
        }
    if len(credits) != len(ids) or any(
        row["License"] != "https://creativecommons.org/licenses/by/2.0/"
        for row in credits.values()
    ):
        raise SystemExit("Expected CC BY 2.0 source license metadata missing")

    import numpy as np
    import tensorflow as tf
    from PIL import Image, ImageDraw

    from car_box.core import iou, load, read, split
    from car_box.model import build

    runtime = {
        "tensorflow": tf.__version__,
        "gpus_visible": [d.name for d in tf.config.list_physical_devices("GPU")],
    }

    manifest = []
    start = time.time()
    for key in ids:
        url = "https://open-images-dataset.s3.amazonaws.com/validation/" + key + ".jpg"
        raw = urllib.request.urlopen(url, timeout=60).read()
        f = out / (key + ".jpg")
        f.write_bytes(raw)
        im = Image.open(f)
        w, h = im.size
        b = boxes[key][0]
        manifest.append(
            [
                f.name,
                float(b["XMin"]) * w,
                float(b["YMin"]) * h,
                float(b["XMax"]) * w,
                float(b["YMax"]) * h,
                key,
            ]
        )
        credits[key]["hosted_sha256"] = hashlib.sha256(raw).hexdigest()
    with (out / "boxes.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "xmin", "ymin", "xmax", "ymax", "group"])
        writer.writerows(manifest)
    (out / "attribution.json").write_text(json.dumps(credits, indent=2) + "\n")

    training, dev = split(read(out / "boxes.csv"))
    x, y = load(training, 64, 64)
    vx, vy = load(dev, 64, 64)
    model = build(64, 64, 2, 4, args.seed)
    mean = y.mean(axis=0)
    baseline = float(np.mean([iou(tuple(a), tuple(mean)) for a in vy]))
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(len(x))
        for i in range(0, len(x), 16):
            idx = order[i : i + 16]
            model.train_on_batch(x[idx], y[idx])
        pred = model(vx, training=False).numpy()
        score = float(
            np.mean([iou(tuple(a), tuple(b)) for a, b in zip(vy, pred, strict=False)])
        )
        history.append({"epoch": epoch + 1, "development_mean_iou": score})
        if score > best:
            best = score
            model.save(out / "best.keras")
        if epoch % 5 == 4 or epoch == 0:
            print(history[-1], flush=True)

    pred = model(vx, training=False).numpy()
    cols = 6
    thumb = 220
    shown = dev[:12]
    rows_n = (len(shown) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb, rows_n * (thumb + 16)), "white")
    draw = ImageDraw.Draw(canvas)
    dev_index = {row.image: i for i, row in enumerate(dev)}
    for tile, row in enumerate(shown):
        im = Image.open(row.image).convert("RGB")
        w, h = im.size
        d2 = ImageDraw.Draw(im)
        truth = row.xyxy
        p = pred[dev_index[row.image]]
        box = (p[0] * w, p[1] * h, p[2] * w, p[3] * h)
        d2.rectangle(truth, outline="lime", width=3)
        d2.rectangle(box, outline="red", width=2)
        im.thumbnail((thumb, thumb))
        x0, y0 = (tile % cols) * thumb, (tile // cols) * (thumb + 16)
        canvas.paste(im, (x0, y0 + 16))
        draw.text((x0 + 2, y0 + 2), row.image.name[:14], fill="black")
    canvas.save(str(out / "training-evidence.png"))

    report = {
        "real_data": True,
        "dataset": "Open Images validation single-car subset",
        "source": "https://storage.googleapis.com/openimages/web/factsfigures_v7.html",
        "selection": (
            f"first {len(ids)} sorted images with exactly1 annotated car, no "
            "group/depiction/interior/truncated; seeded grouped split by imageID"
        ),
        "license": "CC BY4 annotations, CC BY2 images with attribution preserved",
        "train_images": len(x),
        "development_images": len(vx),
        "training_mean_box_development_iou": baseline,
        "best_development_iou": best,
        "epochs": args.epochs,
        "runtime": runtime,
        "model": "64x64 input, depth2 filters4 (same reduced stack as quality_cpu)",
        "history": history,
        "elapsed_seconds": round(time.time() - start, 2),
        "limits": (
            "single-box regression, no negatives/multi-object mAP, split by "
            "imageID not scene-disjoint; development-selected best, no untouched "
            "test; hosted resized copies not originals"
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
