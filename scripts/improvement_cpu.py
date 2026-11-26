"""Explicit local development experiment. No downloads or import-time work."""


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new directory.")
    """Train-only image residual ridge baseline, selected on dev, not test."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import json
    from pathlib import Path

    import numpy as np

    from car_box.core import iou, load, read, split

    out = args.output
    out.mkdir(exist_ok=False)
    train, dev = split(read(args.data / "boxes.csv"))
    x, y = load(train, 16, 16)
    vx, vy = load(dev, 16, 16)
    x = x.reshape(len(x), -1)
    vx = vx.reshape(len(vx), -1)
    mean = y.mean(0)
    xm = x.mean(0)
    x = x - xm
    vx = vx - xm
    trials = []
    for alpha in [0.01, 0.1, 1, 10, 100, 1000]:
        coef = np.linalg.solve(x @ x.T + alpha * np.eye(len(x)), y - mean)
        for scale in [0.1, 0.25, 0.5, 1.0]:
            pred = np.clip(mean + scale * vx @ x.T @ coef, 0, 1)
            pred = np.concatenate(
                [
                    np.minimum(pred[:, :2], pred[:, 2:]),
                    np.maximum(pred[:, :2], pred[:, 2:]),
                ],
                axis=1,
            )
            scores = [iou(tuple(a), tuple(b)) for a, b in zip(vy, pred, strict=False)]
            trials.append(
                dict(
                    alpha=alpha,
                    residual_scale=scale,
                    dev_iou=float(np.mean(scores)),
                    per_image_iou=scores,
                    predictions=pred.tolist(),
                )
            )
    baseline = float(np.mean([iou(tuple(a), tuple(mean)) for a in vy]))
    best = max(trials, key=lambda r: r["dev_iou"])
    r = dict(
        model=(
            "image-conditioned ridge residual around train mean box; explicit "
            "non-CNN alternative "
        ),
        train_images=len(train),
        dev_images=len(dev),
        baseline_dev_iou=baseline,
        selected=best,
        trials=trials,
        limits=(
            "24 candidate settings on only 5 reused development images; "
            "optimistic selection, not generalization, no new test/mAP or "
            "multiobject claim "
        ),
    )
    (out / "report.json").write_text(json.dumps(r, indent=2))
    print(json.dumps({k: v for k, v in r.items() if k != "trials"}))


if __name__ == "__main__":
    main()
