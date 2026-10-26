"""F31a measurement for MLX checkpoints: scripts/f31_measure.py's readouts
and report schema, so f31_verdicts.py and f31_coreport.py read MLX roots
unchanged.

f31_measure.py encodes torch checkpoints only, and the MLX trainer logs test
behavior MSE but not held-out gain, so this script recomputes both from the
checkpoint: held-out behavior gain and (when the run has a trajectory head)
held-out trajectory gain, each MSE(predict train-mean) / MSE(model) on the
test split, as train_torch.heldout_gain defines it. The escape epoch reads
the per-epoch gain from the logged test behavior MSE against the same baseline.

    python scripts/f31_measure_mlx.py --data data/pm_f31fix_loud \
        --runs runs/f31fix_mlx/loud_traj_s{0,1,2,3,4} --out runs/f31fix_mlx/loud_traj.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from f31_measure import escape_epoch, heldout_r2, load_world, r2_dict  # noqa: E402
from probe_appearance import pixel_features  # noqa: E402


def heldout_gain(Y: np.ndarray, pred: np.ndarray, splits: np.ndarray) -> float:
    """MSE(predict train-mean) / MSE(model) on the test split."""
    tr, te = splits == "train", splits == "test"
    base = float(((Y[te] - Y[tr].mean(0)) ** 2).mean())
    mse = float(((Y[te] - pred[te]) ** 2).mean())
    return base / mse if mse > 0 else float("inf")


def logged_gains(history: list[dict], baseline: float) -> list[dict]:
    """Per-epoch held-out gain from the MLX trainer's logged test behavior MSE."""
    return [{"epoch": h["epoch"], "heldout_gain": baseline / h["behavior_mse"]}
            for h in history if h.get("behavior_mse")]


def load_config(run: str):
    from pseudomarble.config import ModelConfig

    meta = json.load(open(os.path.join(run, "metrics.json")))
    known = ModelConfig.__dataclass_fields__.keys()
    c = {k: tuple(v) if isinstance(v, list) else v
         for k, v in meta["config"].items() if k in known}
    return ModelConfig(**c), meta


def encode(run: str, imgs: np.ndarray, chunk: int = 32):
    """-> (z, behavior predictions, trajectory predictions or None, cfg, meta)."""
    import mlx.core as mx

    from pseudomarble.models.mlx_net import build_model

    cfg, meta = load_config(run)
    model = build_model(cfg)
    model.load_weights(os.path.join(run, "model.safetensors"))
    zs, beh, traj = [], [], []
    for i in range(0, len(imgs), chunk):
        out = model(mx.array(imgs[i:i + chunk]))
        zs.append(np.array(out["z"]))
        beh.append(np.array(out["behavior"]))
        if out.get("trajectory") is not None:
            traj.append(np.array(out["trajectory"]))
    return (np.concatenate(zs), np.concatenate(beh),
            np.concatenate(traj) if traj else None, cfg, meta)


def targets(data: str, cfg) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalized behavior targets and (if the head is on) trajectory targets,
    in dataset scene order (the order load_world's images use)."""
    from pseudomarble.data.dataset import PseudoMarbleDataset
    from pseudomarble.models.train import trajectory_kwargs

    ds = PseudoMarbleDataset(data)
    Y = np.array([s.behavior_target(True) for s in ds.scenes])
    kw = trajectory_kwargs(cfg)
    if not kw:
        return Y, None
    T = [np.asarray(b["trajectory"]) for b in
         ds.iter_batches(64, shuffle=False, with_images=False, **kw)]
    return Y, np.concatenate(T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    imgs, E, splits = load_world(args.data)
    tr = splits == "train"
    report = {"data": args.data, "pixel_ceiling_r2": r2_dict(pixel_features(imgs), E),
              "runs": {}}
    print(f"[f31-measure-mlx] {args.data}: {len(E)} scenes")
    print("  pixel ceiling R2  " + "  ".join(
        f"{a} {v:+.3f}" for a, v in report["pixel_ceiling_r2"].items()))
    for r in args.runs:
        z, beh, traj, cfg, meta = encode(r, imgs)
        Y, T = targets(args.data, cfg)
        base = float(((Y[~tr] - Y[tr].mean(0)) ** 2).mean())
        row = {"z_r2": r2_dict(z, E), "z_r2_heldout": heldout_r2(z, E, splits),
               "heldout_gain": heldout_gain(Y, beh, splits),
               "heldout_trajectory_gain": (heldout_gain(T, traj, splits)
                                           if traj is not None else None),
               "latent_pr": meta["history"][-1].get("latent_pr"),
               "escape_epoch": escape_epoch(logged_gains(meta["history"], base))}
        report["runs"][r] = row
        tg = row["heldout_trajectory_gain"]
        print(f"  {r}: z R2 " + "  ".join(f"{a} {v:+.3f}" for a, v in row["z_r2"].items())
              + f" | held-out friction {row['z_r2_heldout']['friction']:+.3f}"
              + f" | gain {row['heldout_gain']:.3f}"
              + (f" traj {tg:.3f}" if tg is not None else "")
              + f" PR {row['latent_pr']:.1f} escape@{row['escape_epoch']}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
