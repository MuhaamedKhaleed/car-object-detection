"""Explicit pretrained-detector demo on the repository's licensed car photos.

Runs the pinned hustvl/yolos-tiny model (Apache-2.0, COCO-trained) downloaded
by scripts/fetch_detector.py on the Open Images validation photos selected by
scripts/fetch_real_data.py (CC BY 2.0 per-image attribution preserved in the
manifest). Reports real detections and IoU against the annotated boxes, and
saves an annotated evidence grid. This pretrained model is a separate baseline,
not this repository's single-box regression model and not trained here.
No import-time work, downloads or hidden execution.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Folder from fetch_real_data.py")
    parser.add_argument("--model", required=True, help="Folder from fetch_detector.py")
    parser.add_argument("--output", required=True, help="New output folder")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    import hashlib
    import json
    import time
    import urllib.request
    from pathlib import Path

    root = Path(args.data)
    out = Path(args.output)
    if out.exists():
        parser.error("Output exists; use a new directory.")
    rows = json.loads((root / "car-selected.json").read_text())

    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    from car_box.core import iou

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForObjectDetection.from_pretrained(args.model)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    car_ids = {k for k, v in id2label.items() if v == "car"}
    if not car_ids:
        parser.error("Model config has no car label; unexpected label set.")

    start = time.time()
    per_image = []
    tiles = []
    for r in rows:
        url = (
            "https://open-images-dataset.s3.amazonaws.com/validation/"
            + r["id"]
            + ".jpg"
        )
        raw = urllib.request.urlopen(url, timeout=60).read()
        img_path = out / "photos"
        img_path.mkdir(parents=True, exist_ok=True)
        f = img_path / (r["id"] + ".jpg")
        f.write_bytes(raw)
        im = Image.open(f).convert("RGB")
        w, h = im.size
        truth = (
            float(r["box"]["XMin"]) * w,
            float(r["box"]["YMin"]) * h,
            float(r["box"]["XMax"]) * w,
            float(r["box"]["YMax"]) * h,
        )
        inputs = processor(images=im, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=args.threshold
        )[0]
        cars = [
            (float(s), tuple(float(v) for v in b))
            for s, label, b in zip(
                result["scores"], result["labels"], result["boxes"], strict=False
            )
            if int(label) in car_ids
        ]
        best = max((iou(truth, b) for _, b in cars), default=0.0)
        per_image.append(
            {
                "id": r["id"],
                "hosted_sha256": hashlib.sha256(raw).hexdigest(),
                "car_detections": len(cars),
                "max_car_score": round(max((s for s, _ in cars), default=0.0), 4),
                "best_car_iou_vs_annotation": round(best, 4),
            }
        )
        draw = ImageDraw.Draw(im)
        draw.rectangle(truth, outline="lime", width=3)
        for score, box in cars[:3]:
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], max(0, box[1] - 12)), f"car {score:.2f}", fill="red")
        tiles.append((r["id"], im))

    detected = [p for p in per_image if p["car_detections"] > 0]
    report = {
        "real_data": True,
        "pretrained_model": "hustvl/yolos-tiny",
        "model_license": "apache-2.0",
        "model_revision": "95a90f3c189fbfca3bcfc6d7315b9e84d95dc2de",
        "threshold": args.threshold,
        "dataset": "Open Images validation single-car subset (CC BY 2.0 images)",
        "images": len(per_image),
        "images_with_car_detected": len(detected),
        "mean_best_car_iou": round(
            sum(p["best_car_iou_vs_annotation"] for p in per_image) / len(per_image), 4
        ),
        "elapsed_seconds": round(time.time() - start, 2),
        "per_image": per_image,
        "limits": (
            "Generic COCO detector, not this repository's regression model; "
            "single annotated car per image by selection; detection quality "
            "shown on 24 development photos is not a benchmark mAP."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "detections.json").write_text(json.dumps(report, indent=2) + "\n")
    cols = 6
    thumb = 220
    rows_n = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb, rows_n * (thumb + 16)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (name, im) in enumerate(tiles):
        t = im.copy()
        t.thumbnail((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 16)
        canvas.paste(t, (x, y + 16))
        draw.text((x + 2, y + 2), name[:14], fill="black")
    canvas.save(str(out / "detector-evidence.png"))
    print(json.dumps({k: v for k, v in report.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
