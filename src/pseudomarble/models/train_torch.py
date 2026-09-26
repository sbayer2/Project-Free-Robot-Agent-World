"""Train the shared-latent model on CPU with PyTorch -- the cloud-sandbox trainer.

A mirror of ``train.py`` (MLX, canonical) for sessions without Apple silicon.
It reuses train.py's argument parser, ``make_config`` and behavior warmup, so
the two trainers cannot drift in what a flag means; only the numeric backend
differs (``torch_net`` instead of ``mlx_net``). NOT canonical: torch and MLX
initialize differently, so torch-trained numbers are comparable to MLX ones
only after a backend-parity check (docs/CLOUD_COMPUTE.md). An experiment whose
every arm trains here is internally valid on its own terms.

    python -m pseudomarble.models.train_torch --data data/pm_f31_g2 \
        --epochs 50 --lr 2e-4 --seed 0 --out runs/torch/g2_s0

Images are decoded once and held in memory (a 512-scene, 16-view, 128 px world
is ~1.5 GB as float32), so an epoch costs compute, not PNG decoding.
Writes ``model.pt`` + ``metrics.json`` (per-epoch history plus the final
held-out behavior gain = MSE(predict train-mean) / MSE(model), the F27/F29
metric).
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

from pseudomarble.data.dataset import PseudoMarbleDataset
from pseudomarble.models.train import (
    behavior_warmup_scale,
    make_config,
    parse_args,
    trajectory_kwargs,
)


def load_split(ds: PseudoMarbleDataset, max_views, traj_kwargs: dict | None = None) -> dict:
    """All of a split as torch tensors: images (S,N,H,W,3), behavior, essence
    (+ the F31 trajectory target when ``traj_kwargs`` is non-empty)."""
    import numpy as np
    import torch

    imgs, beh, ess, traj = [], [], [], []
    traj_kwargs = traj_kwargs or {}
    for b in ds.iter_batches(64, shuffle=False, with_images=True, max_views=max_views,
                             **traj_kwargs):
        imgs.append(np.asarray(b["images"], dtype=np.float32))
        beh.extend(b["behavior"])
        ess.extend(b["essence"])
        traj.extend(b.get("trajectory", []))
    out = {"images": torch.from_numpy(np.concatenate(imgs)),
           "behavior": torch.tensor(beh, dtype=torch.float32),
           "essence": torch.tensor(ess, dtype=torch.float32)}
    if traj_kwargs:
        out["trajectory"] = torch.tensor(traj, dtype=torch.float32)
    return out


def batches(n: int, batch_size: int, shuffle: bool, seed: int) -> list[list[int]]:
    """Index batches; the shuffle is seeded per epoch like train.py's."""
    import random
    order = list(range(n))
    if shuffle:
        random.Random(seed).shuffle(order)
    return [order[i:i + batch_size] for i in range(0, n, batch_size)]


def heldout_gain(train_beh, test_beh, pred) -> float:
    """MSE(predict train-mean) / MSE(model) on the held-out split."""
    base = float(((test_beh - train_beh.mean(0)) ** 2).mean())
    mse = float(((test_beh - pred) ** 2).mean())
    return base / mse if mse > 0 else float("inf")


def participation_ratio(z) -> float:
    """(sum var)^2 / sum(var^2) across latent dims (train.py's latent_pr)."""
    v = z.var(dim=0, unbiased=False)
    return float(v.sum() ** 2 / ((v ** 2).sum() + 1e-12))


def predict(model, data: dict, batch_size: int) -> dict:
    import torch
    outs: dict[str, list] = {"behavior": [], "essence": [], "z": [], "render_mse": []}
    with torch.no_grad():
        for idx in batches(len(data["behavior"]), batch_size, False, 0):
            x = data["images"][idx]
            o = model(x)
            outs["behavior"].append(o["behavior"])
            outs["essence"].append(o["essence"])
            outs["z"].append(o["z"])
            if "trajectory" in o:
                outs.setdefault("trajectory", []).append(o["trajectory"])
            outs["render_mse"].append(((o["render"] - x.mean(dim=1)) ** 2).mean(
                dim=(1, 2, 3)))
    return {k: torch.cat(v) for k, v in outs.items()}


def main(argv: list[str]) -> None:
    import torch

    from pseudomarble.models.torch_net import build_model, loss_fn

    args = parse_args(argv)
    cfg = make_config(args)
    torch.manual_seed(args.seed)

    train_ds = PseudoMarbleDataset(args.data, split="train")
    test_ds = PseudoMarbleDataset(args.data, split="test")
    if len(train_ds) == 0:
        raise SystemExit("no training scenes; generate a dataset first")
    res = train_ds[0].record.get("appearance", {}).get("resolution")
    if res is not None and res != cfg.image_size:
        raise SystemExit(f"dataset rendered at {res}px but model image_size="
                         f"{cfg.image_size}; pass --image-size {res}")
    tf = trajectory_kwargs(cfg)
    train = load_split(train_ds, args.max_views, tf)
    test = load_split(test_ds, args.max_views, tf) if len(test_ds) else None
    print(f"[train-torch] {len(train['behavior'])} train / "
          f"{0 if test is None else len(test['behavior'])} test scenes, "
          f"{torch.get_num_threads()} threads")

    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    os.makedirs(args.out, exist_ok=True)
    history: list[dict] = []

    for epoch in range(args.epochs):
        scale = behavior_warmup_scale(epoch, args.behavior_warmup_epochs)
        epoch_cfg = replace(cfg, behavior_weight=cfg.behavior_weight * scale)
        model.train()
        running, steps = 0.0, 0
        for idx in batches(len(train["behavior"]), args.batch_size, True, epoch):
            x = train["images"][idx]
            opt.zero_grad()
            loss = loss_fn(model(x), train["behavior"][idx], train["essence"][idx],
                           epoch_cfg, render_t=x.mean(dim=1), model=model,
                           trajectory_t=train["trajectory"][idx] if tf else None)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            steps += 1
        model.eval()
        row: dict = {"epoch": epoch, "train_loss": running / max(1, steps)}
        tr_pred = predict(model, {k: v[:128] for k, v in train.items()}, args.batch_size)
        row["latent_pr"] = participation_ratio(tr_pred["z"])
        if test is not None:
            te = predict(model, test, args.batch_size)
            row["behavior_mse"] = float(((te["behavior"] - test["behavior"]) ** 2).mean())
            row["essence_mse"] = float(((te["essence"] - test["essence"]) ** 2).mean())
            row["render_mse"] = float(te["render_mse"].mean())
            row["heldout_gain"] = heldout_gain(train["behavior"], test["behavior"],
                                               te["behavior"])
            if tf:
                row["heldout_trajectory_gain"] = heldout_gain(
                    train["trajectory"], test["trajectory"], te["trajectory"])
        if scale < 1.0:
            row["behavior_weight_scale"] = scale
        history.append(row)
        print(f"[train-torch] epoch {epoch:3d}  " + "  ".join(
            f"{k} {v:.4f}" for k, v in row.items() if k != "epoch"), flush=True)

    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    final = history[-1] if history else {}
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump({"backend": "torch", "seed": args.seed, "lr": args.lr,
                   "epochs": args.epochs, "data": args.data,
                   "config": cfg.__dict__, "final": final, "history": history},
                  fh, indent=2, default=list)
    print(f"[train-torch] saved model.pt + metrics.json -> {args.out}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
