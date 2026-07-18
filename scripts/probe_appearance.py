"""The z -> appearance_params probe (F19): does the latent SEE the essence, or
does the behavior head fail to USE it?

Linear-probes a frozen trained latent z against the ground-truth visual channels
(material_truth.appearance_params) and the hidden essence (density, friction,
restitution). Compares the trained checkpoints to untrained inits (the F6/B4
baseline a random projection gets for free) and to a pixel-feature ceiling (what
is linearly available in the image at all). The preservation fraction

    preserved = (R2[z_trained] - R2[z_untrained]) / (R2[pixels] - R2[z_untrained])

says what share of the pixel-available appearance signal the trained latent
keeps. High => the info is in z and the behavior head ignores it (loss/arch);
low => the encoder discarded it (render/capacity). Design + registered
predictions: docs/PROBE_APPEARANCE.md.

Mac/MLX step (encodes .safetensors checkpoints); the probe itself is numpy.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections.abc import Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from oracle_ceiling import ridge_fit_predict  # noqa: E402

ESSENCE_AXES = ("density", "friction", "restitution")
APPEARANCE_CHANNELS = ("color_r", "color_g", "color_b", "color_a",
                       "roughness", "metallic", "transmission", "ior")


def appearance_row(mt: dict) -> list[float]:
    ap = mt["appearance_params"]
    return list(ap["base_color"]) + [ap["roughness"], ap["metallic"],
                                     ap["transmission"], ap["ior"]]


def kfold_r2(X: np.ndarray, Y: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Per-column out-of-fold R^2 of a ridge X -> Y (Y may be multi-column)."""
    n = len(X)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    pred = np.zeros_like(Y, dtype=float)
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        pred[te] = ridge_fit_predict(X[tr], Y[tr], X[te])
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.where(ss_tot < 1e-12, 1.0, ss_tot)


def pixel_features(imgs_np: np.ndarray) -> np.ndarray:
    """Mean-view appearance summary: per-channel mean+std over the frame, plus an
    8x8 downsample. The ceiling for what is linearly in the pixels."""
    mean_view = imgs_np.mean(axis=1)  # (B, H, W, C)
    b, h, w, c = mean_view.shape
    chan_mean = mean_view.reshape(b, -1, c).mean(1)
    chan_std = mean_view.reshape(b, -1, c).std(1)
    g = 8
    ys = (np.arange(g) * h // g)
    xs = (np.arange(g) * w // g)
    small = mean_view[:, ys][:, :, xs].reshape(b, -1)  # (B, 8*8*C)
    return np.concatenate([chan_mean, chan_std, small], axis=1)


def encode_z(model, imgs, chunk: int = 64) -> np.ndarray:
    import mlx.core as mx  # type: ignore
    rows = []
    for i in range(0, imgs.shape[0], chunk):
        z = model(imgs[i:i + chunk])["z"]
        mx.eval(z)
        rows.append(np.array(z.tolist(), dtype=np.float32))
    return np.concatenate(rows, axis=0)


def summarize(name: str, r2_by_seed: list[np.ndarray], channels: Sequence[str]) -> dict:
    M = np.stack(r2_by_seed)  # (seeds, channels)
    return {"name": name, "channels": list(channels),
            "mean": M.mean(0).tolist(), "std": M.std(0).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pm_big")
    ap.add_argument("--checkpoints", default="runs/basin/lrlo_s*/model.safetensors",
                    help="glob for trained .safetensors")
    ap.add_argument("--untrained-seeds", type=int, default=10)
    ap.add_argument("--max-views", type=int, default=16)
    ap.add_argument("--out", default="runs/probe_appearance/report.json")
    args = ap.parse_args()

    import mlx.core as mx  # type: ignore

    from pseudomarble.config import ModelConfig
    from pseudomarble.data.dataset import PseudoMarbleDataset
    from pseudomarble.models.mlx_net import build_model

    cfg = ModelConfig()
    ds = PseudoMarbleDataset(args.data, split=None)  # ALL scenes; probe is per-scene
    batch = next(ds.iter_batches(batch_size=len(ds), shuffle=False, with_images=True,
                                 max_views=args.max_views, as_mlx=True))
    imgs = batch["images"]
    scene_ids = batch["scene_ids"]
    essence = np.array([[e for e in row] for row in batch["essence"]], dtype=np.float32)

    appear = np.array([appearance_row(json.load(
        open(os.path.join(args.data, sid, "sample.json")))["material_truth"])
        for sid in scene_ids], dtype=np.float32)
    imgs_np = np.array(imgs.tolist(), dtype=np.float32)
    print(f"[probe] {len(scene_ids)} scenes, images {tuple(imgs.shape)}, "
          f"latent_dim={cfg.latent_dim}", flush=True)

    ckpts = sorted(glob.glob(args.checkpoints))
    if not ckpts:
        raise SystemExit(f"no checkpoints match {args.checkpoints!r}")
    print(f"[probe] {len(ckpts)} trained checkpoints, {args.untrained_seeds} untrained", flush=True)

    targets = {"appearance": (appear, APPEARANCE_CHANNELS),
               "essence": (essence, ESSENCE_AXES)}

    # pixel ceiling (one fit; no seed variation).
    px = pixel_features(imgs_np)
    pixels_r2 = {t: kfold_r2(px, Y) for t, (Y, _) in targets.items()}

    trained_r2 = {t: [] for t in targets}
    untrained_r2 = {t: [] for t in targets}

    for ck in ckpts:
        m = build_model(cfg)
        m.load_weights(ck)
        z = encode_z(m, imgs)
        for t, (Y, _) in targets.items():
            trained_r2[t].append(kfold_r2(z, Y))
        print(f"[probe] trained {os.path.basename(os.path.dirname(ck))}", flush=True)

    for k in range(args.untrained_seeds):
        mx.random.seed(2000 + k)
        um = build_model(cfg)
        z = encode_z(um, imgs)
        for t, (Y, _) in targets.items():
            untrained_r2[t].append(kfold_r2(z, Y))
        print(f"[probe] untrained {k}", flush=True)

    # Metric note (see docs/PROBE_APPEARANCE.md amendment 2026-07-16): the
    # preregistered "preservation fraction" (tr - un) / (px - un) is mis-specified.
    # A random-projection encoder is near-LOSSLESS linearly (Johnson-Lindenstrauss),
    # so z_untrained R^2 >= z_trained R^2 for visible channels and the denominator
    # collapses. Training can only DISCARD linear appearance-decodability, never add
    # it. The robust quantities are absolute R^2 and RETENTION = z_trained /
    # z_untrained: the fraction of the random-encoder near-ceiling that survives
    # training. retention ~ 1 => appearance kept in z (behavior head's failure to
    # USE it is the binding constraint); retention << 1 => training discarded it.
    report: dict = {"n_scenes": len(scene_ids), "n_trained": len(ckpts),
                    "n_untrained": args.untrained_seeds, "targets": {}}
    for t, (_, ch) in targets.items():
        tr = summarize("z_trained", trained_r2[t], ch)
        un = summarize("z_untrained", untrained_r2[t], ch)
        px_r2 = pixels_r2[t]
        tr_m = np.array(tr["mean"])
        un_m = np.array(un["mean"])
        # retention is meaningful only where the random encoder itself decodes the
        # channel (untrained R^2 > 0.1) -- otherwise nothing linear is there to keep.
        present = un_m > 0.1
        retention = np.where(present, tr_m / np.where(present, un_m, 1.0), np.nan)
        report["targets"][t] = {
            "channels": list(ch), "pixels_r2": px_r2.tolist(),
            "z_trained_r2": tr["mean"], "z_trained_std": tr["std"],
            "z_untrained_r2": un["mean"], "retention": retention.tolist(),
            "present_in_untrained": present.tolist(),
        }
        print(f"\n=== {t} ===")
        print(f"{'channel':12s} {'pixels':>7s} {'z_untr':>7s} {'z_train':>8s} "
              f"{'retention':>10s}")
        for i, c in enumerate(ch):
            rv = f"{retention[i]*100:6.0f}%" if not np.isnan(retention[i]) else "   n/a"
            mark = "" if present[i] else "  (not linearly in z at all)"
            print(f"{c:12s} {px_r2[i]:7.3f} {un_m[i]:7.3f} {tr_m[i]:8.3f} {rv:>10s}{mark}")
        keep = present & ~np.isnan(retention)
        if keep.any():
            agg = float(np.mean(retention[keep]))
            report["targets"][t]["retention_agg"] = agg
            print(f"{'AGG (present)':12s} {'':7s} {'':7s} {'':8s} {agg*100:9.0f}%")

    os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
    json.dump(report, open(os.path.expanduser(args.out), "w"), indent=2)
    print(f"\n[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
