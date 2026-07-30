#!/usr/bin/env python
"""P5 gate: how much of ABO's behavior is predictable from GEOMETRY alone?

F18's shape-only oracle used a one-hot over 5 MuJoCo primitives. That control
does NOT transfer to ABO: every object carries ``input.shape == "mesh"``, so the
one-hot is a constant column and the oracle degenerates to predict-the-mean
(gain ~ 1.0) no matter how much shape actually explains.

So build the real thing from geometry recovered off the prepared meshes:
extents, aspect ratios, hull volume, area. If this comes in well above pm_big's
1.331x, then on real objects shape explains most of the behavior, essence
headroom shrinks, and an F22 stage-2 null would be uninformative -- which is a
reason to redesign stage 2 rather than run it.

Needs no model and no training. Usage:
    .venv/bin/python scripts/abo_shape_oracle.py [--data data/pm_abo]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from oracle_ceiling import best_oracle  # noqa: E402

PREPARED = os.path.expanduser("~/datasets/abo/prepared")


def geom_features(object_id: str):
    """(extent_x, y, z, aspect_max, aspect_min, hull_vol, area, centroid_z) or None."""
    import trimesh  # type: ignore

    path = os.path.join(PREPARED, object_id, "meshes", "model.obj")
    if not os.path.exists(path):
        return None
    try:
        m = trimesh.load(path, force="mesh")
        e = np.asarray(m.extents, float)
        if e.size != 3 or not np.all(np.isfinite(e)) or e.max() <= 0:
            return None
        srt = np.sort(e)
        hull = float(m.convex_hull.volume) if m.convex_hull is not None else 0.0
        cz = float(m.centroid[2] - m.bounds[0][2]) / max(e[2], 1e-9)
        return [e[0], e[1], e[2], srt[2] / max(srt[0], 1e-9),
                srt[1] / max(srt[0], 1e-9), hull, float(m.area), cz]
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pm_abo")
    args = ap.parse_args()

    from pseudomarble.probes import behavior_vector

    X, Y, split, missing = [], [], [], 0
    for f in sorted(glob.glob(f"{args.data}/*/sample.json")):
        s = json.load(open(f))
        oid = s["material_truth"]["object_id"]
        g = geom_features(oid)
        if g is None:
            missing += 1
            continue
        X.append(g)
        Y.append(behavior_vector(s["behavior"]["probes"], normalize=True))
        split.append(s.get("split"))

    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)
    split = np.asarray(split)
    tr, te = split == "train", split == "test"
    print(f"objects: {len(X)} usable ({missing} meshes missing)  "
          f"train {tr.sum()} / test {te.sum()}  features {X.shape[1]}")

    # log-scale the size-like columns; they span orders of magnitude
    Xl = X.copy()
    for c in (0, 1, 2, 5, 6):
        Xl[:, c] = np.log10(np.maximum(Xl[:, c], 1e-9))
    mu, sd = Xl[tr].mean(0), Xl[tr].std(0) + 1e-9
    Xz = (Xl - mu) / sd

    gain, name, _ = best_oracle(Xz[tr], Y[tr], Xz[te], Y[te])
    print(f"\nABO geometry-only oracle gain = {gain:.3f}   (regressor: {name})")
    print("pm_big shape-only oracle      = 1.331   (5 primitives, one-hot)")

    # A gain BELOW ~1 is not "lots of headroom" -- it means the target is not
    # predictable from these features at all, and on ABO the cause is measurable:
    # the push outcomes are heavy-tailed (median path_length 0.18 m, p99 42 m,
    # max 510 m -- MuJoCo blowing up on concave decomposed meshes). MSE-based
    # gain is then dominated by a handful of physically nonsensical scenes and
    # pins near 1.0 whatever the features are. That is a BLOCKER, not headroom.
    if gain < 1.10:
        verdict = ("TARGET UNPREDICTABLE (gain ~ 1) -- not headroom. The ABO probe "
                   "outcomes are outlier-dominated, so MSE gain cannot resolve any "
                   "effect. Robustify the outcomes before stage 2 is meaningful.")
    elif gain > 1.331:
        verdict = ("HEADROOM GONE -- geometry already explains most of the behavior; "
                   "an F22 stage-2 null would be uninformative. Redesign first.")
    else:
        verdict = ("HEADROOM INTACT -- geometry explains less than pm_big's shape "
                   "did, so essence has room to show up. Stage 2 is worth running.")
    print(f"\n[P5 gate] {verdict}")

    out = "runs/f22/abo_shape_oracle.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"n": int(len(X)), "missing": missing, "gain": float(gain),
               "regressor": name, "pm_big_shape_gain": 1.331,
               "verdict": verdict}, open(out, "w"), indent=2)
    print(f"[P5] wrote {out}")


if __name__ == "__main__":
    main()
