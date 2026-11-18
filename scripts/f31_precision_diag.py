"""F31a encode-precision diagnostics for torch checkpoints (diagnostic, not registered).

Three readings of how much friction a trained z holds, and how robust it is:

* ``readout``: friction R^2 at the frozen fixed alpha 1e-3 (``probe_appearance.
  kfold_r2``, the F31a readout), nested alpha on {0, 1e-5..1e2}
  (``specular_view_diag.cv_readout``), held-out corner R^2
  (``f31_measure.heldout_r2``), and ``z_readout_alpha.std_rank``.
* ``bands``: the out-of-fold gain from alpha 1e-3 to 1e-5, split by singular
  component of the standardized training design. Bands use the shrinkage factor
  f = s^2 / (s^2 + 1e-3): B1 f > 0.9, B2 0.5-0.9, B3 0.1-0.5, B4 < 0.1, with B4
  split at the std_rank threshold (s > 1e-6 * s_max). Also the R^2 from B1 alone.
* ``precision``: re-encode with the inputs and weights of chosen layers rounded
  to TF32 (10 stored mantissa bits), emulating a reduced-precision matmul:
  ``proj`` (the Metal-matched site) or ``all`` (every encoder conv + proj).

    python scripts/f31_precision_diag.py --data data/pm_f31fix_loud \
        --runs runs/f31fix/loud_joint_s{0,1,2,3,4} runs/f31fix/loud_traj_s{0,1,2,3,4} \
        --out runs/f31fix/precision_diag.json

Needs numpy; torch for encoding (pure-numpy helpers are importable without it).
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

FRICTION = 1
A_FIX, A_LO = 1e-3, 1e-5
NESTED_GRID = (0.0,) + tuple(10.0 ** k for k in range(-5, 3))
BAND_EDGES = (0.9, 0.5, 0.1)
RANK_TOL = 1e-6


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def bands_of(s: np.ndarray) -> np.ndarray:
    """Band index 0..4 per singular value: B1..B4 by shrinkage at A_FIX, with B4
    split into 3 (above the std_rank threshold) and 4 (below it)."""
    f = s ** 2 / (s ** 2 + A_FIX)
    b = np.where(f > BAND_EDGES[0], 0, np.where(f > BAND_EDGES[1], 1,
                                               np.where(f > BAND_EDGES[2], 2, 3)))
    return np.where((b == 3) & (s <= RANK_TOL * s[0]), 4, b)


def band_decompose(z: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0) -> dict:
    """Per-band out-of-fold gain from A_FIX to A_LO on kfold_r2's folds.

    The ridge prediction on standardized active columns (intercept unpenalized)
    is a sum over singular components with weights s / (s^2 + alpha); a band's
    gain swaps only its components to the A_LO weight."""
    n = len(y)
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    keys = ("fix", "lo", "b0", "b1", "b2", "b3", "b4", "b0only")
    pred = {key: np.zeros(n) for key in keys}
    counts = np.zeros(5)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        mu, sd = z[tr].mean(0), z[tr].std(0)
        act = sd > 1e-9
        xtr = (z[tr][:, act] - mu[act]) / sd[act]
        xte = (z[te][:, act] - mu[act]) / sd[act]
        ym = y[tr].mean()
        u, s, vt = np.linalg.svd(xtr, full_matrices=False)
        proj = (xte @ vt.T) * (u.T @ (y[tr] - ym))
        w_fix, w_lo = s / (s ** 2 + A_FIX), s / (s ** 2 + A_LO)
        b = bands_of(s)
        counts += np.bincount(b, minlength=5)
        pred["fix"][te] = proj @ w_fix + ym
        pred["lo"][te] = proj @ w_lo + ym
        for j in range(5):
            pred[f"b{j}"][te] = proj @ np.where(b == j, w_lo, w_fix) + ym
        pred["b0only"][te] = proj @ np.where(b == 0, w_fix, 0.0) + ym
    r = {key: _r2(y, p) for key, p in pred.items()}
    gain = [r[f"b{j}"] - r["fix"] for j in range(5)]
    return {"fixed": r["fix"], "lo": r["lo"], "total_gain": r["lo"] - r["fix"],
            "gain": {"B1": gain[0], "B2": gain[1], "B3": gain[2],
                     "B4_above_rank": gain[3], "B4_below_rank": gain[4]},
            "components_per_fold": dict(zip(("B1", "B2", "B3", "B4_above_rank",
                                             "B4_below_rank"), (counts / k).tolist(),
                                            strict=True)),
            "B1_only_r2": r["b0only"]}


def tf32(t):
    """Round a float32 torch tensor to 10 stored mantissa bits (nearest)."""
    import torch
    i = t.contiguous().view(torch.int32)
    return ((i + 0x1000) & ~0x1FFF).view(torch.float32)


def round_layers(model, sites: str):
    """Round weights and input activations of the encoder's matmul layers."""
    import torch
    if sites == "none":
        return model
    mods = {"proj": [model.proj], "all": list(model.convs) + [model.proj]}[sites]
    with torch.no_grad():
        for m in mods:
            m.weight.copy_(tf32(m.weight))
            m.register_forward_pre_hook(lambda mod, args: (tf32(args[0]),))
    return model


def encode(run_dir: str, imgs: np.ndarray, sites: str = "none", chunk: int = 32) -> np.ndarray:
    import torch

    from pseudomarble.config import ModelConfig
    from pseudomarble.models.torch_net import TorchModel

    meta = json.load(open(os.path.join(run_dir, "metrics.json")))
    known = ModelConfig.__dataclass_fields__.keys()
    c = {k: v for k, v in meta["config"].items() if k in known}
    for k in ("conv_channels", "trajectory_probes"):
        if k in c:
            c[k] = tuple(c[k])
    model = TorchModel(ModelConfig(**c))
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), weights_only=True))
    model.eval()
    round_layers(model, sites)
    with torch.no_grad():
        return np.concatenate([model.encode(torch.from_numpy(imgs[i:i + chunk])).numpy()
                               for i in range(0, len(imgs), chunk)])


def readout(z: np.ndarray, E: np.ndarray, splits: np.ndarray) -> dict:
    from f31_measure import heldout_r2
    from probe_appearance import kfold_r2
    from specular_view_diag import cv_readout
    from z_readout_alpha import std_rank
    active, rank = std_rank(z)
    return {"fixed": float(kfold_r2(z, E)[FRICTION]),
            "nested": cv_readout(z, E[:, FRICTION], alphas=NESTED_GRID)["nested"],
            "heldout": heldout_r2(z, E, splits)["friction"],
            "active": active, "rank": rank}


def main(argv=None) -> None:
    from f31_measure import load_world
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--sites", nargs="+", default=["none", "proj", "all"],
                    choices=["none", "proj", "all"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    imgs, E, splits = load_world(args.data)
    report = {"data": args.data, "nested_grid": NESTED_GRID, "runs": {}}
    for r in args.runs:
        row = {}
        z0 = None
        for sites in args.sites:
            z = encode(r, imgs, sites)
            row[sites] = readout(z, E, splits)
            if sites == "none":
                z0 = z
                row["bands"] = band_decompose(z, E[:, FRICTION])
            elif z0 is not None:
                sd0 = z0.std(0)
                act = sd0 > 1e-9
                row[sites]["median_col_sd_ratio"] = float(
                    np.median((z - z0)[:, act].std(0) / sd0[act]))
        report["runs"][r] = row
        print(f"[precision] {r}: " + "  ".join(
            f"{s} fixed {row[s]['fixed']:+.3f} nested {row[s]['nested']:+.3f} "
            f"held-out {row[s]['heldout']:+.3f}" for s in args.sites)
            + (f"  B1-only {row['bands']['B1_only_r2']:+.3f}" if "bands" in row else ""),
            flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
