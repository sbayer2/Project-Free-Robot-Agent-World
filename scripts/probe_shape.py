"""Shape legibility vs material use beyond shape (step 4c of the camera fix).

Two rival explanations for a gain drop between two renders of the same world:
H1, the photos show SHAPE less clearly; H2, the model lost the small MATERIAL
extra it had above the shape-only oracle. This script measures both.

(A) shape accuracy: 5-fold CV ridge (one-hot, argmax) over ALL scenes, from pixel
    features, untrained z and trained z. Ridge alpha is swept and the best
    reported (optimistic for every row alike).
(B) on the test split, per trained seed:
    B1 = fraction of scenes whose predicted behavior is nearest the TRUE shape's
         train-mean behavior (behavior-level shape recognition);
    B2 = MSE(shape-only) / MSE(model), > 1 means the model beats shape-only;
    B3 = Pearson of (prediction - shape mean) with (truth - shape mean), pooled
         over scenes x fields: material use beyond shape.

Mac/MLX step (encodes .safetensors checkpoints); the probes are numpy.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from oracle_ceiling import ridge_fit_predict  # noqa: E402

ALPHAS = (1e-3, 1.0, 100.0)


def load(data: str, split: str | None, max_views: int):
    from pseudomarble.data.dataset import PseudoMarbleDataset
    ds = PseudoMarbleDataset(data, split=split)
    batch = next(ds.iter_batches(batch_size=len(ds), shuffle=False, with_images=True,
                                 max_views=max_views, as_mlx=True))
    shapes = [json.load(open(os.path.join(data, sid, "sample.json")))["input"]["shape"]
              for sid in batch["scene_ids"]]
    beh = np.array(batch["behavior"].tolist(), dtype=np.float64)
    return batch["images"], beh, shapes


def pixel_features(imgs_np: np.ndarray, per_view: bool) -> np.ndarray:
    """8x8 downsample (+ channel mean/std) of the mean view, or of every view."""
    g = 8
    _, n, h, w, c = imgs_np.shape
    ys, xs = np.arange(g) * h // g, np.arange(g) * w // g
    if per_view:
        small = imgs_np[:, :, ys][:, :, :, xs]  # (B, N, g, g, C)
        return small.reshape(len(imgs_np), -1)
    mv = imgs_np.mean(axis=1)
    flat = mv.reshape(len(mv), -1, c)
    return np.concatenate([flat.mean(1), flat.std(1),
                           mv[:, ys][:, :, xs].reshape(len(mv), -1)], axis=1)


def shape_accuracy(X: np.ndarray, labels: np.ndarray, k: int = 5, seed: int = 0) -> float:
    names = sorted(set(labels.tolist()))
    Y = (labels[:, None] == np.array(names)[None, :]).astype(np.float64)
    folds = np.array_split(np.random.default_rng(seed).permutation(len(X)), k)
    best = 0.0
    for alpha in ALPHAS:
        correct = 0
        for i in range(k):
            te = folds[i]
            tr = np.concatenate([folds[j] for j in range(k) if j != i])
            P = ridge_fit_predict(X[tr], Y[tr], X[te], alpha=alpha)
            correct += int((P.argmax(1) == Y[te].argmax(1)).sum())
        best = max(best, correct / len(X))
    return best


def encode(model, imgs, chunk: int = 64):
    import mlx.core as mx  # type: ignore
    zs, bs = [], []
    for i in range(0, imgs.shape[0], chunk):
        out = model(imgs[i:i + chunk])
        mx.eval(out["z"], out["behavior"])
        zs.append(np.array(out["z"].tolist(), dtype=np.float64))
        bs.append(np.array(out["behavior"].tolist(), dtype=np.float64))
    return np.concatenate(zs), np.concatenate(bs)


def behavior_readout(pred: np.ndarray, truth: np.ndarray, shapes: list[str],
                     shape_means: dict[str, np.ndarray]) -> dict:
    names = sorted(shape_means)
    M = np.stack([shape_means[s] for s in names])
    mu_true = np.stack([shape_means[s] for s in shapes])
    nearest = [names[j] for j in ((pred[:, None, :] - M[None]) ** 2).sum(-1).argmin(1)]
    b1 = float(np.mean([a == b for a, b in zip(nearest, shapes, strict=True)]))
    b2 = float(((truth - mu_true) ** 2).mean() / ((truth - pred) ** 2).mean())
    dp, dt = (pred - mu_true).ravel(), (truth - mu_true).ravel()
    b3 = float(np.corrcoef(dp, dt)[0, 1])
    return {"B1_nearest_true_shape": b1, "B2_gain_over_shape_only": b2,
            "B3_within_shape_pearson": b3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoints", required=True, help="glob for trained .safetensors")
    ap.add_argument("--untrained-seeds", type=int, default=10)
    ap.add_argument("--max-views", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import mlx.core as mx  # type: ignore

    from pseudomarble.config import ModelConfig
    from pseudomarble.models.mlx_net import build_model

    cfg = ModelConfig()
    imgs, _, shapes_all = load(args.data, None, args.max_views)
    _, beh_tr, shapes_tr = load(args.data, "train", args.max_views)
    imgs_te, beh_te, shapes_te = load(args.data, "test", args.max_views)
    labels = np.array(shapes_all)
    shape_means = {s: beh_tr[np.array(shapes_tr) == s].mean(0) for s in sorted(set(shapes_tr))}
    imgs_np = np.array(imgs.tolist(), dtype=np.float32)
    print(f"[shape] {len(labels)} scenes ({len(shapes_te)} test), "
          f"shapes={sorted(set(shapes_all))}", flush=True)

    rep: dict = {"data": args.data, "n_scenes": len(labels), "n_test": len(shapes_te),
                 "alphas": list(ALPHAS), "A_shape_accuracy": {}, "B_per_seed": []}
    A = rep["A_shape_accuracy"]
    A["pixels_mean_view"] = shape_accuracy(pixel_features(imgs_np, False), labels)
    A["pixels_per_view"] = shape_accuracy(pixel_features(imgs_np, True), labels)
    print(f"[shape] pixels mean-view {A['pixels_mean_view']:.3f}  "
          f"per-view {A['pixels_per_view']:.3f}", flush=True)

    un = []
    for k in range(args.untrained_seeds):
        mx.random.seed(2000 + k)
        z, _ = encode(build_model(cfg), imgs)
        un.append(shape_accuracy(z, labels))
    A["z_untrained"] = {"mean": float(np.mean(un)), "std": float(np.std(un)), "per_seed": un}

    tr_acc = []
    for ck in sorted(glob.glob(args.checkpoints)):
        m = build_model(cfg)
        m.load_weights(ck)
        z, _ = encode(m, imgs)
        _, pred_te = encode(m, imgs_te)
        acc = shape_accuracy(z, labels)
        tr_acc.append(acc)
        row = {"checkpoint": ck, "shape_accuracy": acc,
               **behavior_readout(pred_te, beh_te, shapes_te, shape_means)}
        rep["B_per_seed"].append(row)
        print(f"[shape] {os.path.basename(os.path.dirname(ck)):10s} acc {acc:.3f}  "
              f"B1 {row['B1_nearest_true_shape']:.2f}  B2 {row['B2_gain_over_shape_only']:.3f}  "
              f"B3 {row['B3_within_shape_pearson']:+.3f}", flush=True)
    A["z_trained"] = {"mean": float(np.mean(tr_acc)), "std": float(np.std(tr_acc))}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"[shape] untrained z {A['z_untrained']['mean']:.3f} +- {A['z_untrained']['std']:.3f}  "
          f"trained z {A['z_trained']['mean']:.3f} +- {A['z_trained']['std']:.3f}")
    print(f"[shape] wrote {args.out}")


if __name__ == "__main__":
    main()
