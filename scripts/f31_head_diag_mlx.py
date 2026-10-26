"""F31a trajectory-head collapse diagnostic, for MLX checkpoints.

The MLX twin of the torch diagnostic in scripts/f31_head_diag.py (PR #46),
with the same per-split, per-block schema: trajectory gain =
MSE(predict train-mean) / MSE(model), and the across-scene std of the head's
predictions against the target's, each averaged over the block's columns. A
collapsed head shows train gain ~1 and prediction std << target std. It adds
one MLX-only readout: the share of each head's hidden ReLUs that are dead
(never positive) on every scene of the world.

This is how the collapse was found on the Mac (2026-10-24): all 5 loud-traj
seeds at train gain 0.98-0.99, prediction std ~0.003 against a target std of
0.25, and the drop block (the legible 0.6 % of the target) at held-out gain
0.40-0.47.

    python scripts/f31_head_diag_mlx.py --runs runs/f31fix_mlx/loud_traj_s{0,1,2,3,4} \
        --data data/pm_f31fix_loud --out runs/f31fix_mlx/head_diag_loud_traj.json
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


def probe_blocks(probes, frames: int, pos_only: bool) -> dict[str, slice]:
    """Column slice of each probe in the flattened trajectory target
    (probes.trajectory_vector's layout), plus "all"."""
    per = frames * (3 if pos_only else 6)
    blocks = {p: slice(i * per, (i + 1) * per) for i, p in enumerate(probes)}
    blocks["all"] = slice(0, per * len(probes))
    return blocks


def block_report(target: np.ndarray, pred: np.ndarray, train_mean: np.ndarray,
                 blocks: dict[str, slice]) -> dict[str, float]:
    """gain / pred_std / target_std per block for one split."""
    out = {}
    for b, sl in blocks.items():
        t, p = target[:, sl], pred[:, sl]
        base = float(((t - train_mean[sl]) ** 2).mean())
        mse = float(((t - p) ** 2).mean())
        out[f"{b}_gain"] = base / mse if mse > 0 else float("inf")
        out[f"{b}_pred_std"] = float(p.std(0).mean())
        out[f"{b}_target_std"] = float(t.std(0).mean())
    return out


def dead_fraction(hidden: np.ndarray) -> float:
    """Share of hidden units (columns) that are never positive on any row."""
    return float((hidden.max(0) <= 0).mean())


def diagnose(run: str, data: str) -> dict:
    import mlx.nn as nn
    from f31_measure_mlx import load_config

    from pseudomarble.data.dataset import PseudoMarbleDataset
    from pseudomarble.models.mlx_net import build_model
    from pseudomarble.models.train import trajectory_kwargs

    cfg, _ = load_config(run)
    if cfg.trajectory_weight <= 0:
        raise SystemExit(f"{run}: no trajectory head (trajectory_weight <= 0)")
    model = build_model(cfg)
    model.load_weights(os.path.join(run, "model.safetensors"))
    blocks = probe_blocks(cfg.trajectory_probes, cfg.trajectory_frames,
                          cfg.trajectory_pos_only)
    res: dict = {}
    hidden = {"behavior": [], "trajectory": []}
    for split in ("train", "test"):
        P, T = [], []
        for b in PseudoMarbleDataset(data, split=split).iter_batches(
                32, shuffle=False, with_images=True, as_mlx=True, **trajectory_kwargs(cfg)):
            out = model(b["images"])
            P.append(np.array(out["trajectory"]))
            T.append(np.array(b["trajectory"]))
            for k in hidden:
                hidden[k].append(np.array(nn.relu(getattr(model, k).fc1(out["z"]))))
        res[split] = (np.concatenate(T), np.concatenate(P))
    tr = res["train"][0]
    row: dict = {}
    for split, (T, P) in res.items():
        row.update({f"{split}_{k}": v
                    for k, v in block_report(T, P, tr.mean(0), blocks).items()})
    var = tr.var(0)
    row["block_share_of_target_var"] = {b: float(var[sl].sum() / var.sum())
                                        for b, sl in blocks.items() if b != "all"}
    row["dead_relu_fraction"] = {k: dead_fraction(np.concatenate(v))
                                 for k, v in hidden.items()}
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True,
                    help="world the runs trained on (MLX metrics.json does not record it)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {}
    for run in args.runs:
        out[os.path.basename(run.rstrip("/"))] = row = diagnose(run, args.data)
        flat = {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}
        print(f"[f31-head-mlx] {run} {json.dumps(flat)} dead "
              f"{json.dumps({k: round(v, 3) for k, v in row['dead_relu_fraction'].items()})}",
              flush=True)
    json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
