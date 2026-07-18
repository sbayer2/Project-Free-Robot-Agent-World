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
    p.add_argument("--lr", type=float, default=5e-4,
                   help="AdamW learning rate (default 5e-4 per F12/F13: 1e-3 "
                        "collapses ~1/3 of inits; 5e-4 trains 20/20 healthy)")
    p.add_argument("--latent-dim", type=int, default=None, help="override ModelConfig")
    p.add_argument("--latent-trits", type=int, default=None,
                   help="FSQ bottleneck width in ternary dims (F17 rate-distortion "
                        "lever); each trit carries log2(3)~1.585 bits; 0/absent = off")
    p.add_argument("--image-size", type=int, default=None,
                   help="model render size; must match the dataset's resolution")
    p.add_argument("--max-views", type=int, default=None, help="cap views per scene")
    p.add_argument("--behavior-weight", type=float, default=None,
                   help="loss weight on the behavior head (default 1.0; set 0 for a "
                        "render-only model)")
    p.add_argument("--essence-weight", type=float, default=None,
                   help="loss weight on the auxiliary essence head (default 0.3)")
    p.add_argument("--appearance-weight", type=float, default=None,
                   help="loss weight on the F20 auxiliary appearance head (z -> the "
                        "8 clean appearance_params; default 0/off). >0 forces the "
                        "encoder to retain the physics-material channels the render "
                        "loss otherwise discards")
    p.add_argument("--render-weight", type=float, default=None,
                   help="loss weight on the render head (default 1.0; set 0 for a "
                        "behavior-only model)")
    p.add_argument("--behavior-warmup-epochs", type=int, default=0,
                   help="ramp the behavior-head loss weight linearly from 0 to its "
                        "full value over the first K epochs (basin-selection lever: "
                        "let render+essence shape the latent before behavior "
                        "gradients act; 0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/exp", help="checkpoint/metrics dir")
    return p.parse_args(argv)


def behavior_warmup_scale(epoch: int, warmup_epochs: int) -> float:
    """Linear 0->1 ramp for the behavior weight: 0 at epoch 0, 1 from epoch
    ``warmup_epochs`` on. ``warmup_epochs <= 0`` disables the ramp (always 1).
    Pure-Python (tested without MLX)."""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / float(warmup_epochs))


def make_config(args: argparse.Namespace) -> ModelConfig:
    from dataclasses import replace
    cfg = ModelConfig()
    if args.latent_dim is not None:
        cfg = replace(cfg, latent_dim=args.latent_dim)
    if args.latent_trits is not None:
        cfg = replace(cfg, latent_trits=args.latent_trits)
    if args.image_size is not None:
        cfg = replace(cfg, image_size=args.image_size)
    if args.behavior_weight is not None:
        cfg = replace(cfg, behavior_weight=args.behavior_weight)
    if args.essence_weight is not None:
        cfg = replace(cfg, essence_weight=args.essence_weight)
    if args.render_weight is not None:
        cfg = replace(cfg, render_weight=args.render_weight)
    if args.appearance_weight is not None:
        cfg = replace(cfg, appearance_weight=args.appearance_weight)
    return cfg


def latent_pr(model, dataset: PseudoMarbleDataset, batch_size: int, max_views,
              cap: int = 128) -> float:
    """Participation ratio of the latent over the first ``cap`` scenes:
    (sum var)^2 / sum(var^2) across dims, variance taken across scenes. PR ~ 0
    means the encoder maps every scene to the same z — the collapse-basin
    signature (FINDINGS F10); healthy escaped models sit at PR ~ 8-84."""
    import mlx.core as mx  # type: ignore
    import numpy as np

    zs, n = [], 0
    for batch in dataset.iter_batches(batch_size, shuffle=False, with_images=True,
                                      max_views=max_views, as_mlx=True):
        z = model(batch["images"])["z"]
        mx.eval(z)
        zs.append(np.asarray(z))
        n += z.shape[0]
        if n >= cap:
            break
    v = np.concatenate(zs, axis=0)[:cap].var(axis=0)
    return float((v.sum() ** 2) / (np.sum(v ** 2) + 1e-12))


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

    from dataclasses import replace

    def batch_loss(m):
        return loss_fn(m, batch, epoch_cfg)

    for epoch in range(args.epochs):
        scale = behavior_warmup_scale(epoch, args.behavior_warmup_epochs)
        epoch_cfg = replace(cfg, behavior_weight=cfg.behavior_weight * scale)
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
        metrics["latent_pr"] = latent_pr(model, train, args.batch_size, args.max_views)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        if scale < 1.0:
            row["behavior_weight_scale"] = scale
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
