"""F31a trajectory-head collapse diagnostic, for torch checkpoints.

Did the trajectory head learn the paths, or predict one path for every scene?
Per run and split: trajectory gain = MSE(predict train-mean) / MSE(model),
overall and per probe block, plus the across-scene std of the head's
predictions against the target's (each averaged over the block's columns).
A collapsed head shows train gain ~1 and prediction std << target std.

Written after the MLX replication found the head collapsed on the Mac; on the
torch F31a fixed-world runs it found the same in all 8 traj seeds (train gain
~0.98, prediction std ~0.002 vs target 0.249, held-out gain never > 1.014).

    python scripts/f31_head_diag.py --runs runs/f31fix/loud_traj_s0 ... \
        --out runs/f31fix/head_diag_loud_traj.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))


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


def diagnose(run: str, cache: dict) -> dict:
    import torch

    from pseudomarble.config import ModelConfig
    from pseudomarble.data.dataset import PseudoMarbleDataset
    from pseudomarble.models.torch_net import build_model
    from pseudomarble.models.train_torch import load_split, predict

    m = json.load(open(os.path.join(run, "metrics.json")))
    cfg = ModelConfig(**{k: tuple(v) if isinstance(v, list) else v
                         for k, v in m["config"].items()})
    if cfg.trajectory_weight <= 0:
        raise SystemExit(f"{run}: no trajectory head (trajectory_weight <= 0)")
    tf = dict(trajectory_frames=cfg.trajectory_frames,
              trajectory_probes=cfg.trajectory_probes,
              trajectory_pos_only=cfg.trajectory_pos_only)
    key = (m["data"], tuple(sorted(tf.items())))
    if key not in cache:
        cache[key] = {s: load_split(PseudoMarbleDataset(m["data"], split=s), None, tf)
                      for s in ("train", "test")}
    data = cache[key]
    model = build_model(cfg)
    model.load_state_dict(torch.load(os.path.join(run, "model.pt"), weights_only=True))
    model.eval()
    blocks = probe_blocks(cfg.trajectory_probes, cfg.trajectory_frames,
                          cfg.trajectory_pos_only)
    tr = data["train"]["trajectory"].numpy()
    hist = [r.get("heldout_trajectory_gain") for r in m["history"]]
    row: dict = {"final_heldout_trajectory_gain_logged": m["final"].get("heldout_trajectory_gain"),
                 "max_heldout_trajectory_gain_any_epoch": max(h for h in hist if h is not None)}
    for split in ("train", "test"):
        pred = predict(model, data[split], 32)["trajectory"].numpy()
        rep = block_report(data[split]["trajectory"].numpy(), pred, tr.mean(0), blocks)
        row.update({f"{split}_{k}": v for k, v in rep.items()})
    var = tr.var(0)
    row["block_share_of_target_var"] = {b: float(var[sl].sum() / var.sum())
                                        for b, sl in blocks.items() if b != "all"}
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cache: dict = {}
    out = {}
    for run in args.runs:
        out[os.path.basename(run.rstrip("/"))] = row = diagnose(run, cache)
        flat = {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}
        print(f"[f31-head] {run} {json.dumps(flat)}", flush=True)
    json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
