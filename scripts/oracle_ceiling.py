"""The oracle ceiling (F18): how much held-out behavior does the TRUE essence predict?

Fits the generator's own inputs -- shape and/or the hidden essence
(density, friction, restitution) -- directly to the 21-dim behavior_vector(),
with no encoder and no training loop. This measures the benchmark's information
ceiling independent of any model, and answers the question F17 left open: is the
"1.6 bits" saturation a fact about the instrument or about the world as authored?

The decisive arm is `shape` (no essence at all). pm_big has 5 shapes = 2.3 bits;
F17 saturates at 1.6. If shape alone reaches the trained model's gain, the
model's measured advantage is shape recognition, not essence.

Metric matches the project's: gain = MSE(predict train-mean) / MSE(oracle).
Design and registered predictions: docs/ORACLE_CEILING.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections.abc import Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pseudomarble.probes import behavior_field_names, behavior_vector  # noqa: E402

ESSENCE_AXES = ("density", "friction", "restitution")


def _appearance_row(ap: dict) -> list[float]:
    """The visual channels the renderer actually used -- what a perfect encoder
    could read off the pixels. Essence -> appearance carries authored Gaussian
    noise (MaterialSampler appearance_noise=0.07) plus a random hue, so this is
    strictly less informative than the true essence. That gap is the point."""
    c = list(ap["base_color"])
    return c + [ap["roughness"], ap["metallic"], ap["transmission"], ap["ior"]]


def load(data_dir: str):
    """-> shape one-hot, essence (normalized), appearance params, behavior (21),
    shape names, split labels."""
    shapes, essences, appears, behaviors, splits = [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(data_dir), "*", "sample.json"))):
        d = json.load(open(f))
        shapes.append(d["input"]["shape"])
        essences.append([d["physics"]["normalized"][a] for a in ESSENCE_AXES])
        appears.append(_appearance_row(d["material_truth"]["appearance_params"]))
        behaviors.append(behavior_vector(d["behavior"]["probes"], normalize=True))
        splits.append(d["split"])
    names = sorted(set(shapes))
    onehot = np.zeros((len(shapes), len(names)))
    for i, s in enumerate(shapes):
        onehot[i, names.index(s)] = 1.0
    return (onehot, np.array(essences), np.array(appears), np.array(behaviors),
            names, np.array(splits))


MAX_QUAD_FEATURES = 2000  # Gram matrix stays under ~32 MB; see best_oracle().


def quadratic(X: np.ndarray) -> np.ndarray:
    """[X, pairwise products incl. squares] -- polynomial features that still
    extrapolate (unlike RBF/kNN), so the corner arm stays meaningful."""
    d = X.shape[1] * (X.shape[1] + 3) // 2
    if d > MAX_QUAD_FEATURES:
        raise ValueError(
            f"quadratic expansion of {X.shape[1]} cols -> {d} features; the ridge "
            f"Gram matrix would be {d*d*8/1e9:.1f} GB. Pass quad=False for one-hot "
            f"blocks (their quadratic expansion is information-free anyway)."
        )
    cols = [X]
    n = X.shape[1]
    for i in range(n):
        for j in range(i, n):
            cols.append((X[:, i] * X[:, j])[:, None])
    return np.concatenate(cols, axis=1)


def ridge_fit_predict(Xtr, Ytr, Xte, alpha: float = 1e-3):
    """Closed-form ridge with an intercept; column-standardized for conditioning."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    A = np.concatenate([(Xtr - mu) / sd, np.ones((len(Xtr), 1))], axis=1)
    B = np.concatenate([(Xte - mu) / sd, np.ones((len(Xte), 1))], axis=1)
    reg = alpha * np.eye(A.shape[1])
    reg[-1, -1] = 0.0  # never penalize the intercept
    W = np.linalg.solve(A.T @ A + reg, A.T @ Ytr)
    return B @ W


def knn_fit_predict(Xtr, Ytr, Xte, k: int = 5):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    d = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d, axis=1)[:, :min(k, len(A))]
    return Ytr[idx].mean(axis=1)


def gain(Ytr: np.ndarray, Yte: np.ndarray, Pte: np.ndarray) -> float:
    """MSE(predict train-mean) / MSE(oracle) on the held-out set."""
    base = float(((Yte - Ytr.mean(0)) ** 2).mean())
    mse = float(((Yte - Pte) ** 2).mean())
    return base / mse if mse > 0 else float("inf")


def best_oracle(Xtr, Ytr, Xte, Yte, quad: bool = True) -> tuple[float, str, np.ndarray]:
    """Best gain over the regressor family -- a ceiling estimate is a LOWER bound.

    ``quad=False`` for one-hot feature blocks: the quadratic expansion of a
    one-hot is information-free (x_i*x_j = 0 for i!=j, x_i^2 = x_i) and the
    normal equations go as O(d^4) -- 492 cluster columns expand to ~124k
    features and a 123 GB Gram matrix.
    """
    cands: list[tuple[str, np.ndarray]] = [
        ("ridge-linear", ridge_fit_predict(Xtr, Ytr, Xte)),
        ("knn-5", knn_fit_predict(Xtr, Ytr, Xte)),
    ]
    if quad:
        cands.append(("ridge-quad", ridge_fit_predict(quadratic(Xtr), Ytr, quadratic(Xte))))
    scored = [(gain(Ytr, Yte, P), name, P) for name, P in cands]
    return max(scored, key=lambda t: t[0])


def quantize(E: np.ndarray, Etr: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    """One-hot of a k-means (Lloyd, numpy) cluster id fit on the TRAIN essence.
    Caps the essence at log2(n_clusters) bits -- the oracle's analogue of F17's FSQ."""
    rng = np.random.default_rng(seed)
    C = Etr[rng.choice(len(Etr), size=min(n_clusters, len(Etr)), replace=False)]
    for _ in range(50):
        a = np.argmin(((Etr[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([Etr[a == i].mean(0) if (a == i).any() else C[i]
                         for i in range(len(C))])
        if np.allclose(newC, C):
            break
        C = newC
    ids = np.argmin(((E[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
    oh = np.zeros((len(E), len(C)))
    oh[np.arange(len(E)), ids] = 1.0
    return oh


def arms(S: np.ndarray, E: np.ndarray, A: np.ndarray) -> dict[str, np.ndarray]:
    """shape+appearance is the FAIR ceiling for any vision encoder (appearance is
    all the pixels carry); shape+essence is the UNFAIR one (true generator inputs,
    unreachable through the authored appearance noise)."""
    return {
        "shape": S,
        "essence": E,
        "shape+appearance": np.concatenate([S, A], 1),
        "shape+essence": np.concatenate([S, E], 1),
    }


def report(tag: str, S, E, A, Y, tr: np.ndarray, te: np.ndarray, fields: Sequence[str]) -> dict:
    print(f"\n=== {tag}: {tr.sum()} train / {te.sum()} test scenes ===")
    print(f"{'arm':16s} {'gain':>7s} {'train gain':>11s}  {'regressor':>12s}")
    out: dict[str, float] = {}
    preds: dict[str, np.ndarray] = {}
    for name, X in arms(S, E, A).items():
        g, which, P = best_oracle(X[tr], Y[tr], X[te], Y[te])
        gtr, _, _ = best_oracle(X[tr], Y[tr], X[tr], Y[tr])
        out[name] = g
        preds[name] = P
        print(f"{name:16s} {g:7.3f} {gtr:11.3f}  {which:>12s}")
    contrib = out["shape+essence"] - out["shape"]
    fair = out["shape+appearance"] - out["shape"]
    print(f"{'essence contribution (shape+essence - shape)':46s} {contrib:+.3f}")
    print(f"{'FAIR contribution (shape+appearance - shape)':46s} {fair:+.3f}"
          f"   <- reachable from pixels")

    print("\n  bit curve (shape free, essence quantized to 2^b clusters):")
    print(f"  {'clusters':>9s} {'bits':>6s} {'gain':>7s}")
    curve = []
    # A cluster needs training support to fit a mean; more clusters than
    # ntrain/4 is memorization, not a bit budget. Skips are reported, not hidden.
    max_clusters = max(1, int(tr.sum()) // 4)
    for n in (1, 3, 9, 81, 729):
        if n > max_clusters:
            print(f"  {n:9d} {np.log2(n):6.1f}  SKIPPED (needs {4*n} train scenes, "
                  f"have {int(tr.sum())})")
            continue
        X = S if n == 1 else np.concatenate([S, quantize(E, E[tr], n)], 1)
        g, _, _ = best_oracle(X[tr], Y[tr], X[te], Y[te], quad=(n == 1))
        bits = 0.0 if n == 1 else float(np.log2(n))
        curve.append({"clusters": n, "bits": bits, "gain": g})
        print(f"  {n:9d} {bits:6.1f} {g:7.3f}")
    Xc = np.concatenate([S, E], 1)
    gc, _, _ = best_oracle(Xc[tr], Y[tr], Xc[te], Y[te])
    print(f"  {'continuous':>9s} {'inf':>6s} {gc:7.3f}")

    print(f"\n  per-field essence contribution (top 6 of {len(fields)}):")
    ps, pse = preds["shape"], preds["shape+essence"]
    per = []
    for i, f in enumerate(fields):
        b = float(((Y[te][:, i] - Y[tr][:, i].mean()) ** 2).mean())
        gs = b / max(float(((Y[te][:, i] - ps[:, i]) ** 2).mean()), 1e-12)
        gse = b / max(float(((Y[te][:, i] - pse[:, i]) ** 2).mean()), 1e-12)
        per.append({"field": f, "shape": gs, "shape+essence": gse, "delta": gse - gs})
    for r in sorted(per, key=lambda r: -r["delta"])[:6]:
        print(f"  {r['field']:22s} shape {r['shape']:6.2f} -> +essence "
              f"{r['shape+essence']:6.2f}  ({r['delta']:+.2f})")
    return {"arms": out, "essence_contribution": contrib,
            "fair_contribution": fair, "bit_curve": curve,
            "continuous_gain": gc, "per_field": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pm_big")
    ap.add_argument("--out", default=None, help="write a JSON report here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    S, E, A, Y, names, splits = load(args.data)
    fields = behavior_field_names()
    print(f"[oracle] {len(Y)} scenes, shapes={names} ({np.log2(len(names)):.1f} bits), "
          f"behavior dim={Y.shape[1]}, appearance dim={A.shape[1]}")

    res = {}
    # corner: the preregistered extrapolation split -- apples-to-apples with F13/F17.
    res["corner"] = report("corner (extrapolation split)", S, E, A, Y,
                           splits == "train", splits == "test", fields)
    # iid: random 80/20 of the TRAIN-region scenes only; removes extrapolation.
    idx = np.where(splits == "train")[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(idx)
    cut = int(0.8 * len(perm))
    tr = np.zeros(len(Y), bool)
    tr[perm[:cut]] = True
    te = np.zeros(len(Y), bool)
    te[perm[cut:]] = True
    res["iid"] = report("iid (random split of train region)", S, E, A, Y, tr, te, fields)

    if args.out:
        os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
        json.dump(res, open(os.path.expanduser(args.out), "w"), indent=2)
        print(f"\n[oracle] wrote {args.out}")


if __name__ == "__main__":
    main()
