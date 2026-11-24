"""Explicit source download. No import-time network or data redistribution."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New source directory"
    )
    parser.add_argument(
        "--cached", type=Path, help="Use already downloaded files, without network"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for filename, url, sha in SOURCES:
        target = args.output / filename
        if args.cached:
            shutil.copyfile(args.cached / filename, target)
        else:
            with (
                urllib.request.urlopen(url, timeout=120) as source,
                target.open("wb") as out,
            ):
                shutil.copyfileobj(source, out)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != sha:
            target.unlink()
            raise ValueError("Source checksum mismatch: " + filename)
        manifest.append({"file": filename, "url": url, "sha256": actual})
    (args.output / "source-manifest.json").write_text(json.dumps(manifest, indent=2))
    import csv
    from collections import defaultdict

    boxes = defaultdict(list)
    with (args.output / "car-boxes.csv").open() as source:
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
    )[:24]
    with (args.output / "car-attribution.csv").open() as source:
        credits = {
            row["ImageID"]: row
            for row in csv.DictReader(source)
            if row["ImageID"] in ids
        }
    if len(credits) != 24 or any(
        row["License"] != "https://creativecommons.org/licenses/by/2.0/"
        for row in credits.values()
    ):
        raise ValueError("Expected source license metadata missing")
    selected = [
        {"id": key, "box": boxes[key][0], "attribution": credits[key]} for key in ids
    ]
    (args.output / "car-selected.json").write_text(json.dumps(selected, indent=2))


SOURCES = [
    (
        "car-boxes.csv",
        "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        "d8bbd59410af14835d7733165a7bb8a3f0213981b22dd5077b0b9f7878991ff2",
    ),
    (
        "car-attribution.csv",
        "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
        "ed93a0e121fe345effdfc7359b848dbc64a1ff6778c8c73563157cb500b33a17",
    ),
]

if __name__ == "__main__":
    main()
