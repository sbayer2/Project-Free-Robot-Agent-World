"""F31 pilot: does the raw probe TRAJECTORY carry material content the 21 summaries discard?

The F18 move applied to the dynamics target, before any model exists. F31 (a
dynamics head, registered in F28 Amendment 2) is only worth designing if the
trajectory is a richer, still-learnable target than behavior_vector(). This
script answers that with oracles on the generator's own inputs -- no encoder,
no training loop -- on the shape-degenerate f27b-style worlds (one box, so all
behavior variation is material).

Three readouts per world:

1. FORWARD gain, the project's metric: gain = MSE(predict train-mean) /
   MSE(oracle), for {essence, appearance} -> {summary (21), trajectory}.
   ``appearance`` is the FAIR arm (all the pixels carry); ``essence`` is the
   unreachable-through-noise ceiling.
2. INVERSE R^2: how much of the hidden essence can be read back OUT of each
   target (ridge/kNN, held-out). If trajectory -> essence beats summary ->
   essence, the summaries were throwing material content away.
3. MOTION content: per probe, the fraction of the 2 s window before the object
   comes to rest -- a trajectory that is mostly stationary is a padded summary,
   not a dynamics target.

Trajectory features: per probe, pos (3) + up (3) at ``--frames`` evenly spaced
samples, relative to the frame-0 pose; every column standardized on TRAIN
statistics so no coordinate dominates the MSE.

Exploratory pilot, NOT a preregistered test: its job is to set F31's bars.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from oracle_ceiling import (  # noqa: E402
    ESSENCE_AXES,
    _appearance_row,
    best_oracle,
    knn_fit_predict,
    ridge_fit_predict,
)

from pseudomarble.probes import REST_SPEED, behavior_vector  # noqa: E402

PROBES = ("drop", "tilt", "push")


def resample(traj: list[dict], n: int) -> np.ndarray:
    """(n, 6) array of [pos - pos0, up] at n evenly spaced frame indices."""
    if len(traj) < 2:
        raise ValueError("trajectory needs at least two frames")
    idx = np.linspace(0, len(traj) - 1, n).round().astype(int)
    p0 = np.asarray(traj[0]["pos"], float)
    rows = [np.concatenate([np.asarray(traj[i]["pos"], float) - p0,
                            np.asarray(traj[i]["up"], float)]) for i in idx]
    return np.stack(rows)


def motion_fraction(traj: list[dict], rest_speed: float = REST_SPEED) -> float:
    """Fraction of the window up to the LAST frame moving faster than rest_speed."""
    t = np.array([f["t"] for f in traj], float)
    p = np.array([f["pos"] for f in traj], float)
    dt = np.diff(t)
    dt = np.where(dt <= 0, np.nan, dt)
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    moving = np.where(np.nan_to_num(v) > rest_speed)[0]
    if len(moving) == 0:
        return 0.0
    return float((moving[-1] + 1) / len(v))


def load(data_dir: str, frames: int):
    """-> essence (n,3), appearance (n,8), summary (n,21), trajectory (n,3*frames*6),
    motion (n,3), split labels."""
    E, A, Ys, Yt, M, sp = [], [], [], [], [], []
    files = sorted(glob.glob(os.path.join(os.path.expanduser(data_dir), "*", "sample.json")))
    if not files:
        raise FileNotFoundError(f"no */sample.json under {data_dir}")
    for f in files:
        d = json.load(open(f))
        probes = {p["spec"]["kind"]: p for p in d["behavior"]["probes"]}
        missing = [k for k in PROBES if "trajectory" not in probes.get(k, {})]
        if missing:
            raise ValueError(f"{f}: no trajectory for {missing}; regenerate with --keep-trajectory")
        E.append([d["physics"]["normalized"][a] for a in ESSENCE_AXES])
        A.append(_appearance_row(d["material_truth"]["appearance_params"]))
        Ys.append(behavior_vector(d["behavior"]["probes"], normalize=True))
        Yt.append(np.concatenate([resample(probes[k]["trajectory"], frames).ravel()
                                  for k in PROBES]))
        M.append([motion_fraction(probes[k]["trajectory"]) for k in PROBES])
        sp.append(d["split"])
    return (np.array(E), np.array(A), np.array(Ys), np.array(Yt), np.array(M),
            np.array(sp))


def standardize(Ytr: np.ndarray, *others: np.ndarray):
    """Scale every column by TRAIN mean/sd; constant columns are dropped (they
    carry no information and would divide by zero)."""
    mu, sd = Ytr.mean(0), Ytr.std(0)
    keep = sd > 1e-9
    f = lambda Y: (Y[:, keep] - mu[keep]) / sd[keep]  # noqa: E731
    return (f(Ytr),) + tuple(f(Y) for Y in others)


def inverse_r2(Xtr, Etr, Xte, Ete) -> dict[str, float]:
    """Held-out R^2 of reading each essence axis back out of a target block,
    best of ridge / kNN (a lower bound, like every oracle here)."""
    best = None
    for P in (ridge_fit_predict(Xtr, Etr, Xte, alpha=1.0), knn_fit_predict(Xtr, Etr, Xte)):
        ss = ((Ete - P) ** 2).sum(0)
        tot = ((Ete - Etr.mean(0)) ** 2).sum(0)
        r2 = 1.0 - ss / np.where(tot > 0, tot, 1.0)
        best = r2 if best is None else np.maximum(best, r2)
    return {a: float(v) for a, v in zip(ESSENCE_AXES, best, strict=True)}


def iid_split(splits: np.ndarray, seed: int):
    idx = np.where(splits == "train")[0]
    perm = np.random.default_rng(seed).permutation(idx)
    cut = int(0.8 * len(perm))
    tr = np.zeros(len(splits), bool)
    te = np.zeros(len(splits), bool)
    tr[perm[:cut]] = True
    te[perm[cut:]] = True
    return tr, te


def block_columns(frames: int) -> dict[str, list[int]]:
    """Column indices of each probe x {pos, up} block in the trajectory target."""
    out = {}
    for pi, k in enumerate(PROBES):
        for lo, name in ((0, "pos"), (3, "up")):
            out[f"{k}.{name}"] = [pi * frames * 6 + f * 6 + lo + c
                                  for f in range(frames) for c in range(3)]
    return out


def block_gains(E, Yt, tr, te, frames: int) -> dict[str, dict]:
    """essence -> each trajectory block, standardized WITHIN the block. Resolves
    which parts of the path the essence determines: the aggregate gain is
    dominated by whatever the per-column standardization promotes."""
    res = {}
    for name, cols in block_columns(frames).items():
        Y = Yt[:, cols]
        a, b = standardize(Y[tr], Y[te])
        if a.shape[1] == 0:
            res[name] = {"gain": None, "note": "constant"}
            continue
        g, which, _ = best_oracle(E[tr], a, E[te], b)
        res[name] = {"gain": g, "regressor": which, "raw_var": float(Y.var(0).sum())}
    return res


def analyze(E, A, Ys, Yt, M, tr, te) -> dict:
    Ys_tr, Ys_te = standardize(Ys[tr], Ys[te])
    Yt_tr, Yt_te = standardize(Yt[tr], Yt[te])
    fwd = {}
    for xname, X in (("essence", E), ("appearance", A)):
        for yname, (ytr, yte) in (("summary", (Ys_tr, Ys_te)), ("trajectory", (Yt_tr, Yt_te))):
            g, which, _ = best_oracle(X[tr], ytr, X[te], yte)
            fwd[f"{xname}->{yname}"] = {"gain": g, "regressor": which}
    inv = {"summary": inverse_r2(Ys_tr, E[tr], Ys_te, E[te]),
           "trajectory": inverse_r2(Yt_tr, E[tr], Yt_te, E[te])}
    return {"forward": fwd, "inverse_r2": inv, "n_train": int(tr.sum()), "n_test": int(te.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="one or more dataset dirs")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", action="store_true",
                    help="also report essence -> each probe x {pos, up} block (iid)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {"frames": args.frames, "worlds": {}}
    for d in args.data:
        E, A, Ys, Yt, M, sp = load(d, args.frames)
        name = os.path.basename(os.path.normpath(d))
        res = {"n": len(E), "summary_dim": Ys.shape[1], "trajectory_dim": Yt.shape[1],
               "motion_fraction": {k: {"mean": float(M[:, i].mean()),
                                       "p10": float(np.percentile(M[:, i], 10)),
                                       "p90": float(np.percentile(M[:, i], 90))}
                                   for i, k in enumerate(PROBES)}}
        res["corner"] = analyze(E, A, Ys, Yt, M, sp == "train", sp == "test")
        res["iid"] = analyze(E, A, Ys, Yt, M, *iid_split(sp, args.seed))
        if args.blocks:
            res["blocks_iid"] = block_gains(E, Yt, *iid_split(sp, args.seed), args.frames)
        report["worlds"][name] = res

        print(f"\n=== {name}: {len(E)} scenes | summary {Ys.shape[1]}d, "
              f"trajectory {Yt.shape[1]}d ===")
        print("  motion fraction (mean, p10-p90): " + ", ".join(
            f"{k} {v['mean']:.2f} ({v['p10']:.2f}-{v['p90']:.2f})"
            for k, v in res["motion_fraction"].items()))
        for split in ("corner", "iid"):
            r = res[split]
            print(f"  [{split}] {r['n_train']} train / {r['n_test']} test")
            for k, v in r["forward"].items():
                print(f"    gain {k:24s} {v['gain']:7.3f}  ({v['regressor']})")
            for tgt, r2 in r["inverse_r2"].items():
                print(f"    essence R2 from {tgt:10s} " +
                      "  ".join(f"{a} {x:+.3f}" for a, x in r2.items()))
        for blk, v in res.get("blocks_iid", {}).items():
            print(f"  [iid block] {blk:10s} " + ("constant" if v["gain"] is None else
                  f"gain {v['gain']:6.3f} ({v['regressor']}), raw var {v['raw_var']:.4f}"))

    if args.out:
        os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
        json.dump(report, open(os.path.expanduser(args.out), "w"), indent=2)
        print(f"\n[f31-oracle] wrote {args.out}")


if __name__ == "__main__":
    main()
