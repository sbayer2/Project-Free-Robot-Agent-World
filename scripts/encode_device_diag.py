"""Where the MLX encode device changes friction in z (diagnostic, not registered).

crossview_measure --device cpu vs gpu moves the Step 9 trajectory arm's
friction readout a lot and the joint arm's little. Per run, this script:
- encodes z four ways, splitting the encoder at the global average pool:
  convs on CPU or GPU x projection on CPU or GPU (cc, cg, gc, gg), and reads
  friction nested (FROZEN_GRID) and at the fixed alpha 1e-3 for each;
- checks the GPU repeats itself and does not depend on the chunk size;
- compares the GPU's z and GAP features with the CPU's per column: the
  standardized discrepancy sd(gpu - cpu) / sd(cpu) and the coefficient of
  variation sd / |mean| of the CPU column, with their Spearman correlation.

    python scripts/encode_device_diag.py --data data/pm_f31fix_loud \
        --runs runs/f31fix_mlx/loud_{joint,traj}_s{0..4} \
        --out runs/f31fix_mlx/device_check/diag.json

Needs MLX.
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
from crossview_measure import FROZEN_GRID, git_head, load_run  # noqa: E402
from f31_measure import load_world, r2_dict  # noqa: E402
from z_readout_alpha import FRICTION, nested_grid  # noqa: E402


def encode_split(model, imgs: np.ndarray, conv_dev: str, proj_dev: str,
                 chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """z (B, L) and per-view GAP features (B * N, C), with the conv stack and
    the projection + view pool each on its own device."""
    import mlx.core as mx
    import mlx.nn as nn

    dev = {"cpu": mx.cpu, "gpu": mx.gpu}
    enc = model.encoder
    zs, gs = [], []
    for i in range(0, len(imgs), chunk):
        x = mx.array(imgs[i:i + chunk])
        B, N = x.shape[0], x.shape[1]
        with mx.stream(dev[conv_dev]):
            h = x.reshape(B * N, *x.shape[2:])
            for conv in enc.convs:
                h = nn.relu(conv(h))
            g = h.mean(axis=(1, 2))
            mx.eval(g)
        with mx.stream(dev[proj_dev]):
            v = nn.relu(enc.proj(g)).reshape(B, N, -1)
            z = enc.pool_views(v)
            mx.eval(z)
        zs.append(np.array(z))
        gs.append(np.array(g))
    return np.concatenate(zs), np.concatenate(gs)


def column_compare(ref: np.ndarray, other: np.ndarray) -> dict:
    """Per active column of ref: sd(other - ref) / sd(ref) against sd / |mean|."""
    from scipy.stats import spearmanr
    sd = ref.std(0)
    act = sd > 1e-9
    disc = (other - ref)[:, act].std(0) / sd[act]
    cv = sd[act] / np.maximum(np.abs(ref[:, act].mean(0)), 1e-30)
    rho = spearmanr(disc, cv).statistic if act.sum() > 2 else float("nan")
    return {"active": int(act.sum()), "disc_median": float(np.median(disc)),
            "disc_max": float(disc.max()), "frac_disc_gt_0.1": float((disc > 0.1).mean()),
            "cv_median": float(np.median(cv)), "spearman_disc_cv": float(rho)}


def friction(z: np.ndarray, E: np.ndarray) -> dict:
    nested, _ = nested_grid(z, E[:, FRICTION], FROZEN_GRID)
    return {"nested": nested, "fixed": r2_dict(z, E)["friction"]}


def main() -> None:
    from pseudomarble.models.mlx_net import build_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    imgs, E, _ = load_world(args.data)
    report = {"data": args.data, "grid": list(FROZEN_GRID), "code": git_head(), "runs": {}}
    for r in args.runs:
        cfg, _ = load_run(r)
        model = build_model(cfg)
        model.load_weights(os.path.join(r, "model.safetensors"))
        dev = {"c": "cpu", "g": "gpu"}
        enc = {k: encode_split(model, imgs, dev[k[0]], dev[k[1]])
               for k in ("cc", "cg", "gc", "gg")}
        z_again, _ = encode_split(model, imgs, "gpu", "gpu")
        z_chunk8, _ = encode_split(model, imgs, "gpu", "gpu", chunk=8)
        row = {"friction": {k: friction(z, E) for k, (z, _) in enc.items()},
               "gpu_repeat_identical": bool(np.array_equal(z_again, enc["gg"][0])),
               "gpu_chunk8_max_abs": float(np.abs(z_chunk8 - enc["gg"][0]).max()),
               "gpu_chunk8_friction": friction(z_chunk8, E),
               "z_columns": column_compare(enc["cc"][0], enc["gg"][0]),
               "gap_columns": column_compare(enc["cc"][1], enc["gg"][1])}
        report["runs"][r] = row
        f = row["friction"]
        print(f"{r}: nested " + " ".join(f"{k} {f[k]['nested']:+.3f}" for k in f)
              + " | fixed " + " ".join(f"{k} {f[k]['fixed']:+.3f}" for k in f)
              + f" | repeat {row['gpu_repeat_identical']} chunk8 "
              f"{row['gpu_chunk8_friction']['nested']:+.3f}"
              f" | z disc med {row['z_columns']['disc_median']:.3g}"
              f" >0.1 {row['z_columns']['frac_disc_gt_0.1']:.2f}"
              f" rho {row['z_columns']['spearman_disc_cv']:+.2f}"
              f" | GAP disc med {row['gap_columns']['disc_median']:.3g}"
              f" cv med {row['gap_columns']['cv_median']:.3g}"
              f" rho {row['gap_columns']['spearman_disc_cv']:+.2f}")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
