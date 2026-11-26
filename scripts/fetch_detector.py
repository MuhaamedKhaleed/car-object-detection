"""Explicit pretrained detector download. No import-time network or weights bundled.

Downloads the pinned hustvl/yolos-tiny revision (Apache-2.0) from Hugging Face
with SHA-256 checks and refuses an existing destination. The pinned revision is
a fixed model snapshot; hashes pin the exact bytes reviewed, not a permission
warranty. Model card: https://huggingface.co/hustvl/yolos-tiny
"""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

REVISION = "95a90f3c189fbfca3bcfc6d7315b9e84d95dc2de"
BASE = "https://huggingface.co/hustvl/yolos-tiny/resolve/" + REVISION + "/"

SOURCES = [
    (
        "config.json",
        "a171607bf33c3c85c2cda36e63f681e8b7fd101396b3f0f7a4aa40214020b86e",
    ),
    (
        "preprocessor_config.json",
        "4309cf0226631c9cd34e6981af842c74f4c9214932741d7fc84b68715c36cf66",
    ),
    (
        "model.safetensors",
        "5a6a017a20cb522dd347271fa5bd670467e456176aaccd940090e50985ac6e74",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New model directory"
    )
    parser.add_argument(
        "--cached", type=Path, help="Use already downloaded files, without network"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for filename, sha in SOURCES:
        target = args.output / filename
        if args.cached:
            shutil.copyfile(args.cached / filename, target)
        else:
            with (
                urllib.request.urlopen(BASE + filename, timeout=300) as source,
                target.open("wb") as out,
            ):
                shutil.copyfileobj(source, out)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != sha:
            target.unlink()
            raise ValueError("Source checksum mismatch: " + filename)
        manifest.append({"file": filename, "url": BASE + filename, "sha256": actual})
    (args.output / "detector-manifest.json").write_text(
        json.dumps(
            {
                "model": "hustvl/yolos-tiny",
                "revision": REVISION,
                "license": "apache-2.0",
                "model_card": "https://huggingface.co/hustvl/yolos-tiny",
                "files": manifest,
                "note": (
                    "Generic COCO-trained detector, separate from this "
                    "repository's single-box regression model."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"downloaded": len(manifest), "revision": REVISION}))


if __name__ == "__main__":
    main()
