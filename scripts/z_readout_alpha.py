"""How much does the trained-z friction readout depend on its ridge strength?

The primary readout (`f31_measure.r2_dict` -> `probe_appearance.kfold_r2`) is a
5-fold ridge at a fixed alpha of 1e-3 on column-standardized z, close to least
squares. Out of fold, least squares pays a variance penalty that grows with the
number of independent directions in z, so its R^2 can move with z's rank even
when the material in z does not. This script reads the same z, on the same
folds, at every alpha of the pilot's grid, plus the pilot's nested readout,
and reports per run:

- active dims (sd above the readout's 1e-9 standardization floor) and the rank
  of standardized z;
- friction R^2 at each alpha, and nested (alpha chosen by inner CV);
- at alpha 1e-3, the worst fold and the share of its squared error carried by
  its three worst scenes.

    python scripts/z_readout_alpha.py --data data/pm_f31fix_loud_cam \
        --runs runs/xv_cam/loud_joint_e50_s{0,1,2,3,4} runs/xv_cam/loud_joint_e150_s{0,1,2,3,4} \
        --out runs/xv_cam/readout_alpha_loud.json

Needs MLX (encodes the checkpoints). Diagnostic, not registered.
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
from crossview_pilot import ALPHAS, nested_r2  # noqa: E402
from oracle_ceiling import ridge_fit_predict  # noqa: E402

FRICTION = 1  # essence axes: density, friction, restitution


def folds_for(n: int, k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """The same folds probe_appearance.kfold_r2 uses."""
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


def std_rank(z: np.ndarray, tol: float = 1e-6) -> tuple[int, int]:
    """(active dims, rank of the column-standardized active dims)."""
    sd = z.std(0)
    act = sd > 1e-9
    zs = (z[:, act] - z[:, act].mean(0)) / sd[act]
    if zs.shape[1] == 0:
        return 0, 0
    sv = np.linalg.svd(zs, compute_uv=False)
    return int(act.sum()), int((sv > sv[0] * tol).sum())


def alpha_sweep(z: np.ndarray, y: np.ndarray, folds: list[np.ndarray],
                alphas=ALPHAS) -> tuple[dict[str, float], np.ndarray]:
    """Pooled out-of-fold R^2 per alpha, and the out-of-fold predictions at alphas[0]."""
    out, first = {}, None
    for a in alphas:
        pred = np.zeros_like(y)
        for i, te in enumerate(folds):
            tr = np.concatenate([f for j, f in enumerate(folds) if j != i])
            pred[te] = ridge_fit_predict(z[tr], y[tr], z[te], alpha=a)
        out[f"{a:g}"] = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        if first is None:
            first = pred
    return out, first


def worst_fold(y: np.ndarray, pred: np.ndarray, folds: list[np.ndarray]) -> dict:
    per = [float(1 - ((y[te] - pred[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
           for te in folds]
    i = int(np.argmin(per))
    err = (y[folds[i]] - pred[folds[i]]) ** 2
    return {"per_fold": per, "worst": i,
            "top3_share": float(np.sort(err)[::-1][:3].sum() / err.sum())}


def encode_z(run: str, imgs: np.ndarray, chunk: int = 32) -> np.ndarray:
    import mlx.core as mx

    from pseudomarble.config import ModelConfig
    from pseudomarble.models.mlx_net import build_model

    meta = json.load(open(os.path.join(run, "metrics.json")))
    known = ModelConfig.__dataclass_fields__.keys()
    cfg = ModelConfig(**{k: tuple(v) if isinstance(v, list) else v
                         for k, v in meta["config"].items() if k in known})
    model = build_model(cfg)
    model.load_weights(os.path.join(run, "model.safetensors"))
    return np.concatenate([np.array(model(mx.array(imgs[i:i + chunk]))["z"])
                           for i in range(0, len(imgs), chunk)])


def main() -> None:
    from f31_measure import load_world

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    imgs, E, _ = load_world(args.data)
    y = E[:, FRICTION]
    folds = folds_for(len(y))
    report = {"data": args.data, "alphas": [f"{a:g}" for a in ALPHAS], "runs": {}}
    for r in args.runs:
        z = encode_z(r, imgs)
        active, rank = std_rank(z)
        sweep, pred = alpha_sweep(z, y, folds)
        row = {"active_dims": active, "std_rank": rank, "friction_r2": sweep,
               "friction_r2_nested": float(nested_r2(z, y[:, None])[0]),
               "fold_detail_alpha_1e-3": worst_fold(y, pred, folds)}
        report["runs"][r] = row
        wf = row["fold_detail_alpha_1e-3"]
        print(f"{r}: active {active} rank {rank} | "
              + " ".join(f"a{a} {v:+.3f}" for a, v in sweep.items())
              + f" | nested {row['friction_r2_nested']:+.3f}"
              + f" | worst fold {wf['per_fold'][wf['worst']]:+.2f}, top-3 scenes"
              f" {wf['top3_share']:.0%} of its error")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
