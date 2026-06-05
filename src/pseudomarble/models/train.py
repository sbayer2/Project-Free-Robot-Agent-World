"""Train the encoder + behavior head on the Mac (MLX).

Predicts the flattened drop/tilt/push behavior vector from multi-view images,
with a light auxiliary essence head. Evaluation is on the **held-out essence
region** (the test split), which is the honest generalization signal — behavior
predicted for material essences never seen in training.

    python -m pseudomarble.models.train --data data/pseudo_marble \
        --epochs 30 --batch-size 16 --lr 1e-3 --out runs/exp1

Arg parsing is pure-Python (tested in any session); everything numeric requires
MLX and runs on Apple silicon. For an in-session forward check without MLX, see
``models/numpy_net.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from pseudomarble.config import ModelConfig
from pseudomarble.data.dataset import PseudoMarbleDataset


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="train pseudo-marble encoder + behavior head")
    p.add_argument("--data", default="data/pseudo_marble", help="dataset root")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=None, help="override ModelConfig")
    p.add_argument("--image-size", type=int, default=None,
                   help="model render size; must match the dataset's resolution")
    p.add_argument("--max-views", type=int, default=None, help="cap views per scene")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/exp", help="checkpoint/metrics dir")
    return p.parse_args(argv)


def make_config(args: argparse.Namespace) -> ModelConfig:
    from dataclasses import replace
    cfg = ModelConfig()
    if args.latent_dim is not None:
        cfg = replace(cfg, latent_dim=args.latent_dim)
    if args.image_size is not None:
        cfg = replace(cfg, image_size=args.image_size)
    return cfg


def evaluate(model, dataset: PseudoMarbleDataset, cfg: ModelConfig,
             batch_size: int, max_views) -> Dict[str, float]:
    """Mean behavior / essence MSE over a split (no shuffling)."""
    import mlx.core as mx  # type: ignore

    nb = ne = nr = n = 0.0
    for batch in dataset.iter_batches(batch_size, shuffle=False,
                                      with_images=True, max_views=max_views, as_mlx=True):
        out = model(batch["images"])
        bs = batch["behavior"].shape[0]
        nb += float(mx.mean((out["behavior"] - batch["behavior"]) ** 2)) * bs
        ne += float(mx.mean((out["essence"] - batch["essence"]) ** 2)) * bs
        nr += float(mx.mean((out["render"] - mx.mean(batch["images"], axis=1)) ** 2)) * bs
        n += bs
    n = max(1.0, n)
    return {"behavior_mse": nb / n, "essence_mse": ne / n, "render_mse": nr / n}


def main(argv: List[str]) -> None:
    import mlx.core as mx  # type: ignore
    import mlx.nn as nn  # type: ignore
    import mlx.optimizers as optim  # type: ignore

    from pseudomarble.models.mlx_net import build_model, loss_fn

    args = parse_args(argv)
    cfg = make_config(args)
    mx.random.seed(args.seed)

    train = PseudoMarbleDataset(args.data, split="train")
    test = PseudoMarbleDataset(args.data, split="test")
    print(f"[train] {len(train)} train / {len(test)} test scenes "
          f"(test = held-out essence region)")
    if len(train) == 0:
        raise SystemExit("no training scenes; generate a dataset first")

    # The render head reconstructs at cfg.image_size, so the dataset must be
    # rendered at that resolution (the recon target is the mean input view).
    res = train[0].record.get("appearance", {}).get("resolution")
    if res is not None and res != cfg.image_size:
        raise SystemExit(
            f"dataset rendered at {res}px but model image_size={cfg.image_size}. "
            f"Regenerate with --resolution {cfg.image_size} (a power-of-two multiple "
            f"of render_seed={cfg.render_seed}), or pass --image-size {res}."
        )

    model = build_model(cfg)
    optimizer = optim.AdamW(learning_rate=args.lr)
    os.makedirs(args.out, exist_ok=True)
    history: List[Dict] = []

    def batch_loss(m):
        return loss_fn(m, batch, cfg)

    for epoch in range(args.epochs):
        running, steps = 0.0, 0
        for batch in train.iter_batches(args.batch_size, shuffle=True, seed=epoch,
                                        with_images=True, max_views=args.max_views,
                                        as_mlx=True):
            loss, grads = nn.value_and_grad(model, batch_loss)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            running += float(loss)
            steps += 1
        train_loss = running / max(1, steps)
        metrics = evaluate(model, test, cfg, args.batch_size, args.max_views) if len(test) else {}
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        print(f"[train] epoch {epoch:3d}  loss {train_loss:.4f}  "
              + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

    model.save_weights(os.path.join(args.out, "model.safetensors"))
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump({"config": cfg.__dict__, "history": history}, fh, indent=2)
    print(f"[train] saved checkpoint + metrics -> {args.out}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
