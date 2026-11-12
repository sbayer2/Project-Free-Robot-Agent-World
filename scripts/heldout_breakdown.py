"""Where on the held-out split does a set of trained seeds gain, and how stable is it?

For each checkpoint, on the test split (gain = predict-the-train-mean MSE / model MSE):
  * essence gain per axis (density, friction, restitution);
  * behavior gain per probe block (drop, tilt, push; 7 fields each);
  * the gain averaged over the last --window epochs of the run's metrics.json
    (test MSE is logged every epoch), which is steadier than the final-epoch
    snapshot when the held-out gain swings from epoch to epoch.

Mac/MLX step (runs the checkpoints); the arithmetic is numpy.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ESSENCE_AXES = ("density", "friction", "restitution")


def load(data: str, split: str, max_views: int):
    from pseudomarble.data.dataset import PseudoMarbleDataset
    ds = PseudoMarbleDataset(data, split=split)
    batch = next(ds.iter_batches(batch_size=len(ds), shuffle=False, with_images=True,
                                 max_views=max_views, as_mlx=True))
    return (batch["images"], np.array(batch["behavior"].tolist(), dtype=np.float64),
            np.array(batch["essence"].tolist(), dtype=np.float64))


def gains(pred: np.ndarray, truth: np.ndarray, mean: np.ndarray, cols) -> float:
    base = ((truth[:, cols] - mean[cols]) ** 2).mean()
    return float(base / ((truth[:, cols] - pred[:, cols]) ** 2).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoints", required=True, help="glob for trained .safetensors")
    ap.add_argument("--max-views", type=int, default=8)
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import mlx.core as mx  # type: ignore

    from pseudomarble.config import ModelConfig
    from pseudomarble.models.mlx_net import build_model
    from pseudomarble.probes import OUTCOME_FIELDS, PROBE_ORDER

    _, beh_tr, ess_tr = load(args.data, "train", args.max_views)
    imgs, beh, ess = load(args.data, "test", args.max_views)
    bm, em = beh_tr.mean(0), ess_tr.mean(0)
    base_b, base_e = ((beh - bm) ** 2).mean(), ((ess - em) ** 2).mean()
    k = len(OUTCOME_FIELDS)
    blocks = {p: list(range(i * k, (i + 1) * k)) for i, p in enumerate(PROBE_ORDER)}

    rows = []
    for ck in sorted(glob.glob(args.checkpoints)):
        m = build_model(ModelConfig())
        m.load_weights(ck)
        out = m(imgs)
        mx.eval(out["behavior"], out["essence"])
        pb = np.array(out["behavior"].tolist(), dtype=np.float64)
        pe = np.array(out["essence"].tolist(), dtype=np.float64)
        hist = json.load(open(os.path.join(os.path.dirname(ck), "metrics.json")))["history"]
        last = hist[-args.window:]
        row = {
            "checkpoint": ck,
            "behavior_gain": gains(pb, beh, bm, list(range(beh.shape[1]))),
            "essence_gain": gains(pe, ess, em, [0, 1, 2]),
            "essence_axis_gain": {a: gains(pe, ess, em, [i]) for i, a in enumerate(ESSENCE_AXES)},
            "behavior_block_gain": {p: gains(pb, beh, bm, c) for p, c in blocks.items()},
            "window_behavior_gain": float(np.mean([base_b / e["behavior_mse"] for e in last])),
            "window_essence_gain": float(np.mean([base_e / e["essence_mse"] for e in last])),
        }
        rows.append(row)
        print(f"{os.path.basename(os.path.dirname(ck)):12s} beh {row['behavior_gain']:.3f} "
              f"(win {row['window_behavior_gain']:.3f})  ess {row['essence_gain']:.3f} "
              f"(win {row['window_essence_gain']:.3f})  "
              + " ".join(f"{a[:4]} {v:.2f}" for a, v in row["essence_axis_gain"].items()) + "  "
              + " ".join(f"{p} {v:.2f}" for p, v in row["behavior_block_gain"].items()), flush=True)

    def ms(vals):
        v = np.asarray(vals, dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std())}

    summary = {
        "behavior_gain": ms([r["behavior_gain"] for r in rows]),
        "window_behavior_gain": ms([r["window_behavior_gain"] for r in rows]),
        "essence_gain": ms([r["essence_gain"] for r in rows]),
        "window_essence_gain": ms([r["window_essence_gain"] for r in rows]),
        "essence_axis_gain": {a: ms([r["essence_axis_gain"][a] for r in rows])
                              for a in ESSENCE_AXES},
        "behavior_block_gain": {p: ms([r["behavior_block_gain"][p] for r in rows]) for p in blocks},
    }
    print("\nsummary:", json.dumps(summary, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"data": args.data, "window": args.window, "per_seed": rows, "summary": summary},
              open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
