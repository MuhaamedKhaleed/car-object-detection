"""Explicit real-data CPU development recipe. Never runs on import."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared local source directory")
    parser.add_argument(
        "--base-model", help="Trusted compatible local DistilBERT directory"
    )
    parser.add_argument(
        "--manifest", help="Reviewed selected Open Images source manifest"
    )
    args = parser.parse_args()
    import csv
    import hashlib
    import json
    import os
    import pathlib
    import time
    import urllib.request

    for k, v in {
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
    }.items():
        os.environ[k] = v
    from PIL import Image

    from car_box.core import read, split
    from car_box.model import train

    root = pathlib.Path(args.data)
    root.mkdir(exist_ok=True)
    rows = json.load(open(args.manifest))
    manifest = []
    # Documented CVDF validation hosted copies; preserve per-image source and license.
    for r in rows:
        url = (
            "https://open-images-dataset.s3.amazonaws.com/validation/"
            + r["id"]
            + ".jpg"
        )
        raw = urllib.request.urlopen(url, timeout=30).read()
        f = root / (r["id"] + ".jpg")
        f.write_bytes(raw)
        im = Image.open(f)
        w, h = im.size
        b = r["box"]
        manifest.append(
            [
                f.name,
                float(b["XMin"]) * w,
                float(b["YMin"]) * h,
                float(b["XMax"]) * w,
                float(b["YMax"]) * h,
                r["id"],
            ]
        )
        r["hosted_url"] = url
        r["hosted_sha256"] = hashlib.sha256(raw).hexdigest()
        r["hosted_size"] = len(raw)
    with (root / "boxes.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "xmin", "ymin", "xmax", "ymax", "group"])
        writer.writerows(manifest)
    training, validation = split(read(root / "boxes.csv"))
    start = time.time()
    scores = train(
        training,
        validation,
        root / "model",
        epochs=2,
        height=64,
        width=64,
        depth=2,
        filters=4,
    )
    report = {
        "real_data": True,
        "dataset": "Open Images validation single-car subset",
        "source": "https://storage.googleapis.com/openimages/web/factsfigures_v7.html",
        "selection": (
            "first24 sorted images with exactly1 annotated car, no "
            "group/depiction/interior/truncated; tiny development split by imageID"
        ),
        "license": (
            "CC BY4 annotations, image metadata CC BY2 with per-image "
            "author/landing preserved; provider no warranty"
        ),
        "training": scores,
        "elapsed_seconds": round(time.time() - start, 2),
        "limits": (
            "single-box regression, no negatives/multi-object mAP, "
            "image groups not scene-disjoint; resized hosted copies not"
            " originals"
        ),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "attribution-manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
