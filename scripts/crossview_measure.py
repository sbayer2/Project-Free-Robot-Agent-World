"""Per-checkpoint readouts for the cross-view pooling preregistration
(docs/CROSSVIEW_POOLING.md §4). scripts/crossview_verdicts.py reads its reports.

Per run, on the world it was trained on (all scenes):
- friction z R^2, nested: alpha picked per outer fold by inner 4-fold CV on the
  frozen grid (primary; z_readout_alpha.nested_grid), with the picks;
- z R^2 per essence axis at the fixed alpha 1e-3 on the same 5 folds
  (f31_measure.r2_dict; friction co-reported, density and restitution
  descriptive);
- active dims and the rank of standardized z (z_readout_alpha.std_rank);
- held-out z R^2: ridge (alpha 1e-3) on train-split z, scored on the test
  split (f31_measure.heldout_r2);
- the final latent PR and whether any logged value is non-finite;
- held-out behavior gain, MSE(predict the train mean) / MSE(model) on the test
  split (descriptive);
- with --d1 (the P0 runs): friction nested R^2 of the five per-view
  constructions on v_1..v_16 (D1), in dual form (perview_readout.nested,
  tested equal to nested_grid), so 4096 columns cost no more than 256.

    python scripts/crossview_measure.py --data data/pm_f31fix_loud_cam \
        --runs runs/xv/P0_loud_s{0..19} --d1 --out runs/xv/P0_loud.json

Needs MLX (encodes the checkpoints).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from f31_measure import AXES, heldout_r2, load_world, r2_dict  # noqa: E402
from perview_readout import nested as nested_dual  # noqa: E402
from z_readout_alpha import FRICTION, nested_grid, std_rank  # noqa: E402

FROZEN_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
D1_QUANTILES = (0.25, 0.5, 0.75, 1.0)


def load_run(run: str):
    from pseudomarble.config import model_config_from_dict
    meta = json.load(open(os.path.join(run, "metrics.json")))
    return model_config_from_dict(meta["config"]), meta


def nonfinite(history: list[dict]) -> bool:
    """True if any logged number (loss, MSEs, PR) is NaN or infinite."""
    return any(isinstance(v, float) and not math.isfinite(v)
               for row in history for v in row.values())


def heldout_gain(Y: np.ndarray, pred: np.ndarray, splits: np.ndarray) -> float:
    """MSE(predict the train mean) / MSE(model) on the test split."""
    tr, te = splits == "train", splits == "test"
    base = float(((Y[te] - Y[tr].mean(0)) ** 2).mean())
    mse = float(((Y[te] - pred[te]) ** 2).mean())
    return base / mse if mse > 0 else float("inf")


def d1_sets(v: np.ndarray) -> dict[str, np.ndarray]:
    """Per-view latents (B, N, L) -> D1's five constructions (§4): the plain
    concatenation and the pilot's C, D, G, H (crossview_pilot.feature_sets),
    computed on v instead of on image statistics."""
    n = len(v)
    dev = np.abs(v - v.mean(1, keepdims=True))
    return {
        "concat": v.reshape(n, -1),
        "C mean + std": np.concatenate([v.mean(1), v.std(1)], 1),
        "D |DFT over view index|": np.abs(np.fft.rfft(v, axis=1)).reshape(n, -1),
        "G sorted across views": np.sort(v, axis=1).reshape(n, -1),
        "H mean + |dev| quantiles": np.concatenate(
            [v.mean(1)] + [np.quantile(dev, q, axis=1) for q in D1_QUANTILES], 1),
    }


def measure(z: np.ndarray, E: np.ndarray, splits: np.ndarray, Y: np.ndarray,
            beh: np.ndarray, history: list[dict], v: np.ndarray | None = None) -> dict:
    """One run's readouts from its encodings (no MLX)."""
    y = E[:, FRICTION]
    nested, picks = nested_grid(z, y, FROZEN_GRID)
    active, rank = std_rank(z)
    row = {"friction_nested": nested, "nested_picks": picks,
           "z_r2_fixed": r2_dict(z, E), "z_r2_heldout": heldout_r2(z, E, splits),
           "active_dims": active, "std_rank": rank,
           "latent_pr": history[-1].get("latent_pr") if history else None,
           "nonfinite": nonfinite(history),
           "heldout_gain": heldout_gain(Y, beh, splits)}
    if v is not None:
        row["d1"] = {name: float(nested_dual(X, y[:, None], grid=FROZEN_GRID)["r2"][0])
                     for name, X in d1_sets(v).items()}
    return row


def encode(run: str, imgs: np.ndarray, views: bool = False, chunk: int = 32) -> dict:
    """Encode every scene with a run's checkpoint: z, behavior predictions and,
    when ``views``, the per-view latents."""
    import mlx.core as mx

    from pseudomarble.models.mlx_net import build_model

    cfg, meta = load_run(run)
    model = build_model(cfg)
    model.load_weights(os.path.join(run, "model.safetensors"))
    zs, beh, vs = [], [], []
    for i in range(0, len(imgs), chunk):
        x = mx.array(imgs[i:i + chunk])
        out = model(x)
        zs.append(np.array(out["z"]))
        beh.append(np.array(out["behavior"]))
        if views:
            vs.append(np.array(model.encode_views(x)))
    return {"z": np.concatenate(zs), "behavior": np.concatenate(beh),
            "v": np.concatenate(vs) if views else None, "cfg": cfg, "meta": meta}


def check_cameras(cfg, data: str, tol: float = 1e-6) -> None:
    """An attn run must be measured on a world whose cameras match its own."""
    from pseudomarble.config import ATTN_POOLS
    from pseudomarble.data.dataset import PseudoMarbleDataset
    if cfg.view_pool not in ATTN_POOLS:
        return
    world = PseudoMarbleDataset(data).camera_directions()
    if len(world) != len(cfg.view_dirs) or any(
            abs(a - b) > tol for u, w in zip(world, cfg.view_dirs, strict=True)
            for a, b in zip(u, w, strict=True)):
        raise SystemExit(f"{data}: cameras differ from the run's view_dirs")


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def main() -> None:
    from pseudomarble.data.dataset import PseudoMarbleDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--d1", action="store_true", help="also read D1 (the P0 runs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = PseudoMarbleDataset(args.data)
    if args.d1:
        # D1's concat and DFT constructions index views by position, so every
        # scene must store the same cameras in the same order (raises if not).
        ds.camera_directions()
    imgs, E, splits = load_world(args.data)
    Y = np.array([s.behavior_target(True) for s in ds.scenes])
    report = {"data": args.data, "grid": list(FROZEN_GRID), "axes": list(AXES),
              "code": git_head(), "runs": {}}
    print(f"[xv-measure] {args.data}: {len(E)} scenes")
    for r in args.runs:
        enc = encode(r, imgs, views=args.d1)
        check_cameras(enc["cfg"], args.data)
        row = measure(enc["z"], E, splits, Y, enc["behavior"], enc["meta"]["history"],
                      enc["v"])
        row["view_pool"] = enc["cfg"].view_pool
        report["runs"][r] = row
        print(f"  {r} [{row['view_pool']}]: friction nested {row['friction_nested']:+.3f}"
              f" fixed {row['z_r2_fixed']['friction']:+.3f}"
              f" held-out {row['z_r2_heldout']['friction']:+.3f}"
              f" | rank {row['std_rank']}/{row['active_dims']} PR {row['latent_pr']:.1f}"
              + (" NONFINITE" if row["nonfinite"] else "")
              + ("".join(f" | D1 {k.split()[0]} {x:+.3f}" for k, x in row["d1"].items())
                 if "d1" in row else ""))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
