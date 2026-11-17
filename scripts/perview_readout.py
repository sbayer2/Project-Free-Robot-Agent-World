"""Which camera views carry the material, view by view, and how much do all 16 carry together?

The cross-view pilot reads features built from all 16 views at once. This
reads the photos one view at a time, the 16-view mean, and all 16 views
concatenated, with the same features the cloud session used on its MuJoCo 3.17
re-renders (cloud-notes 9ec0ce3, 19a367f), so the Mac's own photos can be set
beside them:

- per view: 16x16x3 block-mean pixels (768 dims);
- mean: the 16-view mean of those (768 dims);
- concat: all 16 side by side (12,288 dims).

Readout: ridge on column-standardized features with an unpenalized intercept
(the same estimator as oracle_ceiling.ridge_fit_predict), solved in dual form
so 12,288 dims cost no more than 768. Outer 5-fold R^2 per essence axis, alpha
picked per axis by inner 4-fold CV (primary), mean over outer-fold seeds;
best-of-grid R^2 (alpha chosen on the outer folds, optimistic) is co-reported
because the cloud's numbers are best of 4.

    python scripts/perview_readout.py --data data/pm_f31fix_loud data/pm_f31fix_loud_cam \
        data/pm_f31fix_ctrl data/pm_f31fix_ctrl_cam --out runs/postfix/perview.json
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

GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6)


def block_features(imgs: np.ndarray, g: int = 16) -> np.ndarray:
    """(B, V, H, W, C) -> (B, V, g*g*C): the mean of each (H/g x W/g) block, per view."""
    B, V, H, W, C = imgs.shape
    x = imgs.reshape(B, V, g, H // g, g, W // g, C).mean(axis=(3, 5), dtype=np.float64)
    return x.reshape(B, V, g * g * C)


class RidgePath:
    """Standardized ridge with an unpenalized intercept, for many alphas at once.

    Columns are standardized on the training rows, so they are centered there and the
    intercept is the training mean of Y. The penalized part is solved in dual form
    through one eigendecomposition of the training Gram matrix."""

    def __init__(self, Xtr: np.ndarray, Ytr: np.ndarray):
        self.mu = Xtr.mean(0)
        sd = Xtr.std(0)
        self.sd = np.where(sd < 1e-9, 1.0, sd)
        self.Xc = (Xtr - self.mu) / self.sd
        self.ybar = Ytr.mean(0)
        lam, U = np.linalg.eigh(self.Xc @ self.Xc.T)
        self.lam, self.U = lam, U
        self.UtY = U.T @ (Ytr - self.ybar)

    def predict(self, Xte: np.ndarray, alpha: float) -> np.ndarray:
        Kte = ((Xte - self.mu) / self.sd) @ self.Xc.T
        return self.ybar + (Kte @ self.U) @ (self.UtY / (self.lam + alpha)[:, None])


def _r2(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1.0 - ((Y - P) ** 2).sum(0) / np.where(tot > 1e-12, tot, 1.0)


def nested(X: np.ndarray, Y: np.ndarray, grid=GRID, k: int = 5, seed: int = 0) -> dict:
    """Outer k-fold R^2 per column with alpha picked per column by inner 4-fold CV, the
    best-of-grid outer R^2 per column, and the alphas picked (k x columns). The fold and
    inner-split draws follow crossview_pilot.nested_r2."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(X)), k)
    P = np.zeros_like(Y)
    Pgrid = np.zeros((len(grid),) + Y.shape)
    picks = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        inner = np.array_split(rng.permutation(tr), 4)
        loss = np.zeros((len(grid), Y.shape[1]))
        for m, vte in enumerate(inner):
            vtr = np.concatenate([f for j, f in enumerate(inner) if j != m])
            path = RidgePath(X[vtr], Y[vtr])
            for a, alpha in enumerate(grid):
                loss[a] += ((Y[vte] - path.predict(X[vte], alpha)) ** 2).sum(0)
        best = loss.argmin(0)
        path = RidgePath(X[tr], Y[tr])
        for a, alpha in enumerate(grid):
            Pgrid[a, te] = path.predict(X[te], alpha)
        P[te] = np.stack([Pgrid[best[c], te, c] for c in range(Y.shape[1])], 1)
        picks.append([float(grid[b]) for b in best])
    best_of_grid = np.max(np.stack([_r2(Y, Pgrid[a]) for a in range(len(grid))]), 0)
    return {"r2": _r2(Y, P), "best_of_grid": best_of_grid, "picks": picks}


def log_density(E: np.ndarray) -> np.ndarray:
    """Essence targets with density (column 0, kg/m^3 / 8000, sampled log-uniform) on a
    log scale; the other axes unchanged."""
    if (E[:, 0] <= 0).any():
        raise ValueError("density target must be positive to take its log")
    out = E.copy()
    out[:, 0] = np.log(E[:, 0])
    return out


def readout_sets(F: np.ndarray) -> dict[str, np.ndarray]:
    """(B, V, D) per-view features -> the sets read: each view, the mean, the concatenation."""
    sets = {f"view {i}": F[:, i] for i in range(F.shape[1])}
    sets["mean"] = F.mean(1)
    sets["concat"] = F.reshape(len(F), -1)
    return sets


def main() -> None:
    from f31_measure import AXES, load_world

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=3, help="outer-fold seeds")
    ap.add_argument("--block", type=int, default=16,
                    help="blocks per image side (16 = the cloud's 16x16x3 features)")
    ap.add_argument("--log-density", action="store_true",
                    help="read density on a log scale (the cloud's target) instead of linear")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {"grid": list(GRID), "axes": list(AXES), "seeds": args.seeds,
              "block": args.block, "density_scale": "log" if args.log_density else "linear",
              "worlds": {}}
    for data in args.data:
        imgs, E, _ = load_world(data)
        if args.log_density:
            E = log_density(E)
        F = block_features(imgs, g=args.block)
        del imgs
        world = {}
        print(f"[perview] {data}: {len(E)} scenes, {F.shape[1]} views, {F.shape[2]} dims per view"
              f"   nested R2 (best-of-grid), mean over {args.seeds} fold seeds")
        for name, X in readout_sets(F).items():
            runs = [nested(X, E, seed=s) for s in range(args.seeds)]
            r2 = np.mean([r["r2"] for r in runs], 0)
            bog = np.mean([r["best_of_grid"] for r in runs], 0)
            lo = np.min([r["r2"] for r in runs], 0)
            hi = np.max([r["r2"] for r in runs], 0)
            picks = [p for r in runs for p in r["picks"]]
            edge = sorted({p[c] for p in picks for c in range(len(AXES))} & {GRID[0], GRID[-1]})
            world[name] = {
                "r2": dict(zip(AXES, r2.tolist(), strict=True)),
                "r2_range": {a: [float(x), float(y)]
                             for a, x, y in zip(AXES, lo, hi, strict=True)},
                "best_of_grid": dict(zip(AXES, bog.tolist(), strict=True)), "picks": picks}
            cells = zip(AXES, r2, bog, strict=True)
            print(f"  {name:8s} " + "  ".join(f"{a} {v:+.3f} ({b:+.3f})" for a, v, b in cells)
                  + (f"   picks at grid edge {edge}" if edge else ""))
        report["worlds"][data] = world
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
