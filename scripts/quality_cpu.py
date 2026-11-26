"Explicit measured CPU quality experiment. No import-time work or downloads."


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model", type=Path, help="Translation only: trusted local Marian"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new directory.")
    import os

    for k in ["TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"]:
        os.environ[k] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import json
    import resource
    import time
    from pathlib import Path

    import numpy as np

    from car_box.core import iou, load, read, split
    from car_box.model import build

    root = Path(str(args.output))
    root.mkdir(exist_ok=False)
    training, dev = split(read(args.data / "boxes.csv"))
    x, y = load(training, 64, 64)
    vx, vy = load(dev, 64, 64)
    model = build(64, 64, 2, 4, 42)
    start = time.time()
    best = -1
    history = []
    # Check whether visual learning beats the training-mean position prior.
    mean = y.mean(axis=0)
    baseline = float(np.mean([iou(tuple(a), tuple(mean)) for a in vy]))
    for epoch in range(120):
        order = np.random.default_rng(42 + epoch).permutation(len(x))
        for i in range(0, len(x), 4):
            idx = order[i : i + 4]
            model.train_on_batch(x[idx], y[idx])
        if epoch % 5 == 4:
            pred = model(vx, training=False).numpy()
            score = float(
                np.mean(
                    [iou(tuple(a), tuple(b)) for a, b in zip(vy, pred, strict=False)]
                )
            )
            row = {"epoch": epoch + 1, "development_mean_iou": score}
            history.append(row)
            if score > best:
                best = score
                model.save(root / "best.keras")
            if epoch in [4, 29, 59, 119]:
                print(row, flush=True)
    r = {
        "source": "https://storage.googleapis.com/openimages/web/factsfigures_v7.html",
        "train_images": len(x),
        "development_images": len(vx),
        "training_mean_box_development_iou": baseline,
        "best_development_iou": best,
        "history": history,
        "elapsed_seconds": time.time() - start,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limits": (
            "Same24 development images, no untouchedtest or mAP. "
            "Selection over120epochs optimistic; one toycar in source, "
            "singlebox not general detector."
        ),
    }
    (root / "report.json").write_text(json.dumps(r, indent=2))
    print(json.dumps({k: v for k, v in r.items() if k != "history"}))


if __name__ == "__main__":
    main()
