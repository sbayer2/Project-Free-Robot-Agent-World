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
- friction R^2 at each alpha, and nested (alpha chosen by inner CV over the
  same grid, with the alpha each outer fold picked); --alphas sets the grid,
  default the pilot's, and alpha 0 means the minimum-norm least-squares fit;
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
from crossview_pilot import ALPHAS  # noqa: E402
from oracle_ceiling import ridge_fit_predict  # noqa: E402

FRICTION = 1  # essence axes: density, friction, restitution
PRIMARY_ALPHA = 1e-3  # probe_appearance.kfold_r2 via oracle_ceiling.ridge_fit_predict


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


def fit_predict(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, alpha: float) -> np.ndarray:
    """ridge_fit_predict for alpha > 0; for alpha 0, the minimum-norm least-squares
    fit on the same standardized design (z is rank-deficient, so a plain solve fails)."""
    if alpha > 0:
        return ridge_fit_predict(Xtr, Ytr, Xte, alpha=alpha)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    A = np.concatenate([(Xtr - mu) / sd, np.ones((len(Xtr), 1))], axis=1)
    B = np.concatenate([(Xte - mu) / sd, np.ones((len(Xte), 1))], axis=1)
    return B @ np.linalg.lstsq(A, Ytr, rcond=None)[0]


def nested_grid(X: np.ndarray, y: np.ndarray, alphas=ALPHAS, k: int = 5,
                seed: int = 0) -> tuple[float, list[float]]:
    """crossview_pilot.nested_r2 for one target over any alpha grid: outer k-fold R^2,
    alpha picked by inner 4-fold CV on each training part. -> (R^2, alpha per fold)."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(X)), k)
    pred, picks = np.zeros_like(y), []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        inner = np.array_split(rng.permutation(tr), 4)

        def loss(a: float, inner=inner) -> float:
            s = 0.0
            for m, vte in enumerate(inner):
                vtr = np.concatenate([f for j, f in enumerate(inner) if j != m])
                s += ((y[vte] - fit_predict(X[vtr], y[vtr], X[vte], a)) ** 2).sum()
            return float(s)

        a = min(alphas, key=loss)
        picks.append(float(a))
        pred[te] = fit_predict(X[tr], y[tr], X[te], a)
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()), picks


def alpha_sweep(z: np.ndarray, y: np.ndarray, folds: list[np.ndarray],
                alphas=ALPHAS) -> tuple[dict[str, float], np.ndarray]:
    """Pooled out-of-fold R^2 per alpha, and the out-of-fold predictions at alphas[0]."""
    out, first = {}, None
    for a in alphas:
        pred = np.zeros_like(y)
        for i, te in enumerate(folds):
            tr = np.concatenate([f for j, f in enumerate(folds) if j != i])
            pred[te] = fit_predict(z[tr], y[tr], z[te], a)
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
    ap.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS),
                    help="ridge grid for the sweep and the nested readout (0 = min-norm)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    alphas = tuple(args.alphas)

    imgs, E, _ = load_world(args.data)
    y = E[:, FRICTION]
    folds = folds_for(len(y))
    report = {"data": args.data, "alphas": [f"{a:g}" for a in alphas], "runs": {}}
    for r in args.runs:
        z = encode_z(r, imgs)
        active, rank = std_rank(z)
        sweep, _ = alpha_sweep(z, y, folds, alphas)
        _, pred = alpha_sweep(z, y, folds, (PRIMARY_ALPHA,))
        nested, picks = nested_grid(z, y, alphas)
        row = {"active_dims": active, "std_rank": rank, "friction_r2": sweep,
               "friction_r2_nested": nested, "nested_alpha_picks": picks,
               "fold_detail_alpha_1e-3": worst_fold(y, pred, folds)}
        report["runs"][r] = row
        wf = row["fold_detail_alpha_1e-3"]
        print(f"{r}: active {active} rank {rank} | "
              + " ".join(f"a{a} {v:+.3f}" for a, v in sweep.items())
              + f" | nested {nested:+.3f} (picks {' '.join(f'{a:g}' for a in picks)})"
              + f" | worst fold {wf['per_fold'][wf['worst']]:+.2f}, top-3 scenes"
              f" {wf['top3_share']:.0%} of its error")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
