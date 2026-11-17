"""Cross-view pooling pilot: is friction in how the 16 views DIFFER?

The encoder averages its per-view latents (``mlx_net.py`` Encoder, last line),
so anything carried by differences between views is discarded before z. This
pilot measures, with no training, how much friction fixed per-view image
statistics carry under different ways of combining the views.

Per view: colour mean and std, luminance quantiles (0.5, 0.9, 0.99, 1.0) and
the fraction of pixels above three luminance thresholds (highlight area).
Feature sets combine those across views:

- A  mean-view pixel features (F19's ``pixel_features``, the "pixel ceiling")
- C  per-view stats, mean + std across views
- D  |DFT over the view index| (the FNet move applied to views)
- D' the same with the views shuffled by one fixed permutation first
- G  per-view stats sorted across views (order-free)
- H  mean + quantiles of |deviation from the view mean|

Readouts: nested 5-fold ridge R^2 (alpha chosen by an inner 4-fold CV on the
training part only), averaged over 3 fold seeds; plus ridge fit on the
world's train split, scored on its held-out (corner) scenes.

Exploratory and post hoc, NOT a preregistered test (docs/CROSSVIEW_POOLING.md
discloses its numbers as the baseline the preregistration was written after).

    python scripts/crossview_pilot.py --data data/pm_f31fix_loud data/pm_f31fix_ctrl
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
from f31_measure import AXES, load_world  # noqa: E402
from oracle_ceiling import ridge_fit_predict  # noqa: E402
from probe_appearance import pixel_features  # noqa: E402

ALPHAS = (1e-3, 1e-1, 1.0, 10.0, 100.0)
SHUFFLE_SEED = 7


def per_view(imgs: np.ndarray) -> np.ndarray:
    """(B, V, H, W, 3) images -> (B, V, 13) per-view colour and highlight stats."""
    B, V = imgs.shape[:2]
    x = imgs.reshape(B, V, -1, 3)
    lum = x.mean(-1)
    q = np.quantile(lum, [0.5, 0.9, 0.99, 1.0], axis=2).transpose(1, 2, 0)
    hi = np.stack([(lum > t).mean(-1) for t in (0.6, 0.8, 0.9)], -1)
    return np.concatenate([x.mean(2), x.std(2), q, hi], -1)


def feature_sets(imgs: np.ndarray) -> dict[str, np.ndarray]:
    f = per_view(imgs)
    n = len(f)
    perm = np.random.default_rng(SHUFFLE_SEED).permutation(f.shape[1])
    dev = np.abs(f - f.mean(1, keepdims=True))
    return {
        "A mean-view pixels": pixel_features(imgs),
        "C mean + std across views": np.concatenate([f.mean(1), f.std(1)], 1),
        "D |DFT over view index|": np.abs(np.fft.rfft(f, axis=1)).reshape(n, -1),
        "D' |DFT|, views shuffled": np.abs(np.fft.rfft(f[:, perm], axis=1)).reshape(n, -1),
        "G sorted across views": np.sort(f, axis=1).reshape(n, -1),
        "H mean + |dev| quantiles": np.concatenate(
            [f.mean(1)] + [np.quantile(dev, q, axis=1) for q in (0.25, 0.5, 0.75, 1.0)], 1),
    }


def _r2(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1.0 - ((Y - P) ** 2).sum(0) / np.where(tot > 1e-12, tot, 1.0)


def inner_cv_loss(X: np.ndarray, Y: np.ndarray, inner: list, alpha: float) -> float:
    """Summed squared error of ridge(alpha) over the inner folds."""
    s = 0.0
    for m in range(len(inner)):
        vte = inner[m]
        vtr = np.concatenate([inner[j] for j in range(len(inner)) if j != m])
        s += ((Y[vte] - ridge_fit_predict(X[vtr], Y[vtr], X[vte], alpha=alpha)) ** 2).sum()
    return float(s)


def nested_r2(X: np.ndarray, Y: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Outer k-fold R^2, alpha picked by inner 4-fold CV on each training part."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(X)), k)
    P = np.zeros_like(Y)
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        inner = np.array_split(rng.permutation(tr), 4)
        alpha = min(ALPHAS, key=lambda a, inner=inner: inner_cv_loss(X, Y, inner, a))
        P[te] = ridge_fit_predict(X[tr], Y[tr], X[te], alpha=alpha)
    return _r2(Y, P)


def heldout_r2(X: np.ndarray, Y: np.ndarray, splits: np.ndarray) -> np.ndarray | None:
    """Ridge on the train split, R^2 on the held-out split; best alpha by friction
    (an optimistic readout, reported only beside the nested one)."""
    tr, te = splits == "train", splits == "test"
    if te.sum() <= 2:
        return None
    return max((_r2(Y[te], ridge_fit_predict(X[tr], Y[tr], X[te], alpha=a)) for a in ALPHAS),
               key=lambda r: r[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=3, help="outer-fold seeds")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {}
    for d in args.data:
        imgs, E, splits = load_world(d)
        name = os.path.basename(os.path.normpath(d))
        print(f"\n{name}: {len(E)} scenes, {int((splits == 'test').sum())} held out"
              "   (nested 5-fold R2, mean over fold seeds)")
        rows = {}
        for label, X in feature_sets(imgs).items():
            r = np.array([nested_r2(X, E, seed=s) for s in range(args.seeds)])
            h = heldout_r2(X, E, splits)
            rows[label] = {"dim": int(X.shape[1]),
                           "nested": {a: float(r[:, i].mean()) for i, a in enumerate(AXES)},
                           "friction_seed_range": [float(r[:, 1].min()), float(r[:, 1].max())],
                           "heldout": None if h is None else
                           {a: float(v) for a, v in zip(AXES, h, strict=True)}}
            print(f"  {label:28s} dim {X.shape[1]:4d}  friction {r[:, 1].mean():+.3f}"
                  f" ({r[:, 1].min():+.3f}..{r[:, 1].max():+.3f})  density {r[:, 0].mean():+.3f}"
                  f"  restitution {r[:, 2].mean():+.3f}"
                  + ("" if h is None else f" | held-out friction {h[1]:+.3f}"))
        report[name] = rows
        del imgs

    if args.out:
        os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
        json.dump(report, open(os.path.expanduser(args.out), "w"), indent=2)
        print(f"\n[crossview-pilot] wrote {args.out}")


if __name__ == "__main__":
    main()
