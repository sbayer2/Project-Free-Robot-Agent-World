"""Run the coherence experiment on a trained MLX model — the project's payoff.

Measures, on the **held-out essence region** (test split), whether training raises
render↔physics coherence above the *architectural* baseline that an untrained
shared model already shows (FINDINGS F6): the honest signal is

    learned_coherence = trained_shared - mean_over_seeds(untrained_shared)

evaluated for BOTH physics-side targets (compare_both_targets logic):
  * behavior — the drop/tilt/push outcome head (headline, but partly non-smooth;
    `push.toppled` is a threshold — see F8, treat it with care);
  * essence  — the (density, friction, restitution) head, a smooth function of z,
    robust to the topple-chaos bias. A high essence-coherence beside a low
    behavior-coherence is the chaos artifact, not a true absence of coupling.

Coherence is forward-only finite differences (`models/coherence.py`), so we drive
`mean_coherence` directly with MLX decoders rather than `coherence_bench.compare`
(which assumes a `.encode` method the MLX model doesn't expose). The independent
render-only/behavior-only ~0 control needs masked-loss training and is a follow-up;
`learned_coherence` is the scientifically load-bearing number here.

Mac only (MLX). Train the shared model first::

    python -m pseudomarble.models.train --data data/pm_mid --epochs 40 \
        --image-size 128 --out runs/mid_shared
    python scripts/run_coherence_experiment.py --data data/pm_mid \
        --checkpoint runs/mid_shared/model.safetensors --out runs/mid_coherence
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import replace
from typing import Dict, List, Sequence


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="coherence experiment on a trained MLX model")
    p.add_argument("--data", required=True, help="dataset root (must have a test split)")
    p.add_argument("--checkpoint", required=True, help="trained shared model .safetensors")
    p.add_argument("--out", default="runs/coherence", help="report dir")
    p.add_argument("--split", default="test", help="which split to measure on (held-out=test)")
    p.add_argument("--image-size", type=int, default=128, help="must match the checkpoint")
    p.add_argument("--latent-dim", type=int, default=None, help="override ModelConfig if needed")
    p.add_argument("--max-views", type=int, default=None, help="cap views/scene (match training)")
    p.add_argument("--n-dirs", type=int, default=48, help="latent perturbation directions")
    p.add_argument("--eps", type=float, default=1e-3, help="finite-difference step")
    p.add_argument("--untrained-seeds", type=int, default=5,
                   help="number of fresh untrained inits for the architectural baseline band")
    p.add_argument("--seed", type=int, default=0, help="direction-sampling seed")
    return p.parse_args(argv)


def make_config(args: argparse.Namespace):
    from pseudomarble.config import ModelConfig
    cfg = ModelConfig()
    if args.image_size is not None:
        cfg = replace(cfg, image_size=args.image_size)
    if args.latent_dim is not None:
        cfg = replace(cfg, latent_dim=args.latent_dim)
    return cfg


def load_split_images(data: str, split: str, max_views, image_size: int):
    """Return (images mx.array (B,N,H,W,C), behavior (B,21), essence (B,3))."""
    import mlx.core as mx  # type: ignore

    from pseudomarble.data.dataset import PseudoMarbleDataset

    ds = PseudoMarbleDataset(data, split=split)
    if len(ds) == 0:
        raise SystemExit(f"no scenes in split={split!r}; generate a dataset with a {split} region")
    imgs, beh, ess = [], [], []
    for batch in ds.iter_batches(batch_size=len(ds), shuffle=False, with_images=True,
                                 max_views=max_views, as_mlx=True):
        imgs.append(batch["images"])
        beh.append(batch["behavior"])
        ess.append(batch["essence"])
    return (mx.concatenate(imgs, axis=0), mx.concatenate(beh, axis=0),
            mx.concatenate(ess, axis=0), len(ds))


def encode_to_lists(model, imgs) -> List[List[float]]:
    import numpy as np  # noqa: F401
    z = model(imgs)["z"]
    import mlx.core as mx  # type: ignore
    mx.eval(z)
    return [[float(v) for v in row] for row in z.tolist()]


def make_decoders(model):
    """z(list) -> flat list, for render / behavior / essence heads (MLX forward)."""
    import mlx.core as mx  # type: ignore

    def _flat(arr) -> List[float]:
        mx.eval(arr)
        return [float(v) for v in arr.reshape(-1).tolist()]

    def f_render(z: Sequence[float]) -> List[float]:
        return _flat(model.decode(mx.array([list(z)], dtype=mx.float32))[0])

    def f_behavior(z: Sequence[float]) -> List[float]:
        return _flat(model.behavior_from_z(mx.array([list(z)], dtype=mx.float32))[0])

    def f_essence(z: Sequence[float]) -> List[float]:
        return _flat(model.essence_from_z(mx.array([list(z)], dtype=mx.float32))[0])

    return f_render, f_behavior, f_essence


def test_mse(model, imgs, beh, ess, beh_mean, ess_mean) -> Dict[str, float]:
    """Held-out MSE for each head, plus a predict-the-train-mean baseline and the
    ratio (>1 ⇒ the model generalizes to the held-out region; ≤1 ⇒ no better than
    guessing the training mean — the honest read on a hard extrapolation split)."""
    import mlx.core as mx  # type: ignore
    out = model(imgs)
    render_target = mx.mean(imgs, axis=1)
    b = float(mx.mean((out["behavior"] - beh) ** 2))
    e = float(mx.mean((out["essence"] - ess) ** 2))
    base_b = float(mx.mean((beh - beh_mean) ** 2))
    base_e = float(mx.mean((ess - ess_mean) ** 2))
    return {
        "behavior_mse": b, "essence_mse": e,
        "render_mse": float(mx.mean((out["render"] - render_target) ** 2)),
        "behavior_mse_baseline": base_b, "essence_mse_baseline": base_e,
        "behavior_gain_over_mean": (base_b / b) if b > 0 else None,
        "essence_gain_over_mean": (base_e / e) if e > 0 else None,
    }


def participation_ratio(model, imgs) -> Dict[str, float]:
    """Effective number of active latent dims = (Σvar)²/Σvar² over the encodings.
    Diagnostic against the latent-rank-collapse confound: if a TRAINED model's
    coherence were high merely because z collapsed to a few dims (both heads then
    trivially co-respond), its PR would be LOW. We expect the opposite."""
    import numpy as np
    z = np.asarray(model(imgs)["z"])
    v = z.var(axis=0)
    pr = float((v.sum() ** 2) / (np.sum(v ** 2) + 1e-12))
    return {"participation_ratio": pr, "n_dims": int(z.shape[1])}


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    try:
        import mlx.core as mx  # type: ignore # noqa: F401
    except Exception:
        raise SystemExit("this experiment needs MLX (Apple silicon): pip install -e \".[mlx]\"")
    import mlx.core as mx  # type: ignore

    from pseudomarble.models.coherence_bench import mean_coherence
    from pseudomarble.models.mlx_net import build_model

    cfg = make_config(args)
    imgs, beh, ess, n = load_split_images(args.data, args.split, args.max_views, cfg.image_size)
    tr_imgs, tr_beh, tr_ess, n_tr = load_split_images(args.data, "train", args.max_views,
                                                      cfg.image_size)
    beh_mean = mx.mean(tr_beh, axis=0, keepdims=True)
    ess_mean = mx.mean(tr_ess, axis=0, keepdims=True)
    print(f"[coherence] split={args.split}: {n} held-out scenes ({n_tr} train), "
          f"images {tuple(imgs.shape)}, latent_dim={cfg.latent_dim}, n_dirs={args.n_dirs}")

    # --- trained shared model -------------------------------------------------
    trained = build_model(cfg)
    trained.load_weights(args.checkpoint)
    zs = encode_to_lists(trained, imgs)
    fr, fb, fe = make_decoders(trained)
    shared = {
        "behavior": mean_coherence(zs, fr, fb, n_dirs=args.n_dirs, eps=args.eps, seed=args.seed),
        "essence": mean_coherence(zs, fr, fe, n_dirs=args.n_dirs, eps=args.eps, seed=args.seed),
    }
    mse = test_mse(trained, imgs, beh, ess, beh_mean, ess_mean)
    pr_trained = participation_ratio(trained, tr_imgs)
    print(f"[coherence] trained shared  behavior={shared['behavior']:.4f}  "
          f"essence={shared['essence']:.4f}")
    print(f"[coherence] held-out MSE    behavior={mse['behavior_mse']:.4f} "
          f"(mean {mse['behavior_mse_baseline']:.4f}, {mse['behavior_gain_over_mean']:.2f}x)  "
          f"essence={mse['essence_mse']:.4f} "
          f"(mean {mse['essence_mse_baseline']:.4f}, {mse['essence_gain_over_mean']:.2f}x)  "
          f"render={mse['render_mse']:.4f}")

    # --- architectural baseline over fresh untrained inits --------------------
    arch = {"behavior": [], "essence": []}
    pr_untrained: List[float] = []
    for k in range(args.untrained_seeds):
        mx.random.seed(1000 + k)
        um = build_model(cfg)
        zsu = encode_to_lists(um, imgs)
        ufr, ufb, ufe = make_decoders(um)
        arch["behavior"].append(mean_coherence(zsu, ufr, ufb, n_dirs=args.n_dirs,
                                                eps=args.eps, seed=args.seed))
        arch["essence"].append(mean_coherence(zsu, ufr, ufe, n_dirs=args.n_dirs,
                                               eps=args.eps, seed=args.seed))
        pr_untrained.append(participation_ratio(um, tr_imgs)["participation_ratio"])
        print(f"[coherence] untrained seed {k}: behavior={arch['behavior'][-1]:.4f}  "
              f"essence={arch['essence'][-1]:.4f}")

    def agg(vals: List[float]) -> Dict[str, float]:
        return {"mean": statistics.mean(vals),
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals), "max": max(vals)}

    pr_un_mean = statistics.mean(pr_untrained) if pr_untrained else None
    report = {"split": args.split, "n_scenes": n, "n_train": n_tr, "n_dirs": args.n_dirs,
              "untrained_seeds": args.untrained_seeds, "config_image_size": cfg.image_size,
              "latent_dim": cfg.latent_dim, "held_out_mse": mse,
              "latent_participation": {"trained": pr_trained["participation_ratio"],
                                       "untrained_mean": pr_un_mean, "n_dims": cfg.latent_dim},
              "targets": {}}
    print(f"[coherence] latent participation ratio: trained={pr_trained['participation_ratio']:.1f}"
          f"  untrained={pr_un_mean:.1f} (of {cfg.latent_dim})  "
          f"— rules out rank-collapse if trained > untrained")
    print("\n[coherence] === learned_coherence = trained_shared - mean(untrained) ===")
    for tgt in ("behavior", "essence"):
        a = agg(arch[tgt])
        learned = shared[tgt] - a["mean"]
        report["targets"][tgt] = {
            "trained_shared": shared[tgt],
            "architectural_baseline": a,
            "learned_coherence": learned,
            "clears_band": learned > a["std"],   # crude: above one std of the baseline noise
        }
        print(f"  {tgt:9s} trained={shared[tgt]:.4f}  "
              f"arch={a['mean']:.4f}±{a['std']:.4f} [{a['min']:.3f},{a['max']:.3f}]  "
              f"=> learned={learned:+.4f}  {'(clears band)' if learned > a['std'] else '(within noise)'}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "coherence_report.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[coherence] wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
