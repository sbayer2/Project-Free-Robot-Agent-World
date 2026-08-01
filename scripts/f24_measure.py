#!/usr/bin/env python
"""F24 measurement driver: representational convergence (docs/CONVERGENCE.md).

Encodes every preregistered checkpoint set, computes all alignment numbers
(mutual kNN primary, linear CKA secondary, trit agreement for the FSQ sets),
and writes runs/f24/f24_report.json. NO verdicts here -- the frozen decision
rules live in scripts/f24_verdicts.py, which reads the report.

Mac only (MLX). Usage:
    .venv/bin/python scripts/f24_measure.py            # full run
    .venv/bin/python scripts/f24_measure.py --dry-run  # list work, touch nothing
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

from pseudomarble.models.alignment import (  # noqa: E402
    knn_indices,
    mutual_information,
    participation_ratio,
    terciles,
    trit_agreement,
    trit_agreement_null,
)

# (label, r, dataset, checkpoint prefix) -- preregistration section 4.
ARMS = [
    ("ctrl", 0.000, "data/pm_f22_ctrl", "runs/f22/ctrl_cont"),
    ("base", 0.263, "data/pm_f22_base", "runs/f22/base_cont"),
    ("g15", 0.636, "data/pm_f22_g15", "runs/f22/g15_cont"),
    ("loud", 0.992, "data/pm_f22_loud", "runs/f22/loud_cont"),
]
# Cross-arm pairs (section 5): PRIMARY is base<->loud; the rest descriptive.
CROSS = [("base", "loud"), ("ctrl", "loud"), ("base", "g15"), ("g15", "loud")]
# (label, r, dataset, checkpoint prefix) for the k=1 FSQ sets (section 6).
FSQ_SETS = [
    ("fsq_base", 0.263, "data/pm_big", "runs/fsq/k1"),
    ("fsq_loud", 0.992, "data/pm_f22_loud", "runs/f22/loud_fsq1"),
]
K = 10
N_UNTRAINED = 8
UNTRAINED_SEED_BASE = 1000  # mx.random.seed(1000+i), section 3
PERM_SEED = 2026
PR_FLOOR = 8.0
FACTORS = ("heaviness", "grip", "hardness")


def build_from_metrics(ck_path: str):
    """Rebuild the exact ModelConfig a checkpoint was trained with."""
    from pseudomarble.config import ModelConfig
    from pseudomarble.models.mlx_net import build_model

    with open(os.path.join(os.path.dirname(ck_path), "metrics.json")) as f:
        cfg_dict = json.load(f)["config"]
    cfg_dict["conv_channels"] = tuple(cfg_dict["conv_channels"])
    cfg = ModelConfig(**cfg_dict)
    m = build_model(cfg)
    m.load_weights(ck_path)
    return m, cfg


def encode_arm(data: str):
    """Load one arm's images once. Returns (imgs mlx, test mask, ds)."""
    from render_fidelity_eval import load_arrays

    imgs, _yb, _ya, _tr, te, ds = load_arrays(data)
    return imgs, te, ds


def encode_codes(model, imgs, chunk: int = 64) -> np.ndarray:
    """Chunked FSQ code extraction -> (B,) ints in {-1,0,1}."""
    import mlx.core as mx  # type: ignore

    rows = []
    for i in range(0, imgs.shape[0], chunk):
        code = model(imgs[i:i + chunk])["code"]
        mx.eval(code)
        rows.append(np.array(code.tolist(), np.float32))
    flat = np.concatenate(rows, axis=0).reshape(-1)
    return np.rint(flat).astype(np.int64)


def overlap(ix: np.ndarray, iy: np.ndarray) -> float:
    """Mean kNN-set overlap given precomputed (n,k) neighbor index arrays."""
    return float(np.mean([len(set(a.tolist()) & set(b.tolist())) / ix.shape[1]
                          for a, b in zip(ix, iy, strict=True)]))


def pair_stats(zs: list, knns: list, grams: list, pairs: list) -> dict:
    """Alignment over the given index pairs; per-unit mean kNN for the stats."""
    n_units = len(zs)
    knn_vals, cka_vals = {}, {}
    for i, j in pairs:
        knn_vals[(i, j)] = overlap(knns[i], knns[j])
        num = float(np.trace(grams[i] @ grams[j]))
        den = float(np.sqrt(np.trace(grams[i] @ grams[i])
                            * np.trace(grams[j] @ grams[j])))
        cka_vals[(i, j)] = num / den if den > 0 else 0.0
    per_unit = []
    for u in range(n_units):
        vals = [v for (i, j), v in knn_vals.items() if u in (i, j)]
        per_unit.append(float(np.mean(vals)) if vals else float("nan"))
    return {
        "pairs": [[i, j] for i, j in pairs],
        "knn": [knn_vals[p] for p in pairs],
        "cka": [cka_vals[p] for p in pairs],
        "knn_mean": float(np.mean(list(knn_vals.values()))),
        "cka_mean": float(np.mean(list(cka_vals.values()))),
        "per_unit_mean_knn": per_unit,
    }


def centered_gram(z: np.ndarray) -> np.ndarray:
    zc = z.astype(np.float64)
    Km = zc @ zc.T
    n = Km.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ Km @ H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/f24/f24_report.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = []
    for label, r, data, prefix in ARMS + FSQ_SETS:
        cks = sorted(glob.glob(f"{prefix}_s*/model.safetensors"))
        plan.append((label, r, data, prefix, cks))
        print(f"{label:9s} r={r:.3f}  {data:18s}  {len(cks):2d} checkpoints")
    if args.dry_run:
        print(f"untrained per arm: {N_UNTRAINED}  k={K}  perm seed={PERM_SEED}")
        return

    import mlx.core as mx  # type: ignore
    from probe_appearance import encode_z

    from pseudomarble.config import ModelConfig
    from pseudomarble.models.mlx_net import build_model

    report: dict = {"k": K, "n_untrained": N_UNTRAINED,
                    "untrained_seed_base": UNTRAINED_SEED_BASE,
                    "perm_seed": PERM_SEED, "pr_floor": PR_FLOOR,
                    "arms": {}, "cross": {}, "fsq": {}, "gates": {}}
    all_pairs = lambda n: [(i, j) for i in range(n) for j in range(i + 1, n)]  # noqa: E731

    # Cache of per-arm artifacts kept after images are freed (z is tiny).
    Z: dict = {}          # label -> {"trained": [z...], "untrained": [z...]}
    KNN: dict = {}        # label -> same structure of (n,k) index arrays
    KNN_TE: dict = {}
    GRAM: dict = {}
    scene_ids: dict = {}

    for label, r, data, _prefix, cks in plan[:len(ARMS)]:
        print(f"=== arm {label} (r={r:.3f}) ===")
        imgs, te, ds = encode_arm(data)
        scene_ids[label] = [s.record["scene_id"] for s in ds.scenes]
        seeds = [os.path.basename(os.path.dirname(c)).split("_s")[-1] for c in cks]

        trained_z, prs, dropped = [], [], []
        for ck in cks:
            m, _cfg = build_from_metrics(ck)
            z = encode_z(m, imgs)
            pr = participation_ratio(z)
            prs.append(pr)
            if pr < PR_FLOOR:
                dropped.append(ck)
            else:
                trained_z.append(z)
            del m
        untrained_z = []
        for i in range(N_UNTRAINED):
            mx.random.seed(UNTRAINED_SEED_BASE + i)
            m = build_model(ModelConfig(image_size=128))
            untrained_z.append(encode_z(m, imgs))
            del m
        del imgs
        mx.clear_cache()

        knn_t = [knn_indices(z, K) for z in trained_z]
        knn_u = [knn_indices(z, K) for z in untrained_z]
        gram_t = [centered_gram(z) for z in trained_z]
        gram_u = [centered_gram(z) for z in untrained_z]
        te_idx = np.where(te)[0]
        knn_t_te = [knn_indices(z[te_idx], K) for z in trained_z]
        knn_u_te = [knn_indices(z[te_idx], K) for z in untrained_z]

        # Gates (section 8): self == 1.0, scene-shuffle ~ chance.
        rng = np.random.default_rng(PERM_SEED)
        z0 = trained_z[0]
        self_a = overlap(knn_indices(z0, K), knn_indices(z0, K))
        shuf_a = overlap(knn_indices(z0, K), knn_indices(z0, K)[rng.permutation(len(z0))])
        report["gates"][label] = {"self": self_a, "shuffle": shuf_a,
                                  "chance": K / (len(z0) - 1)}

        tstats = pair_stats(trained_z, knn_t, gram_t, all_pairs(len(trained_z)))
        ustats = pair_stats(untrained_z, knn_u, gram_u, all_pairs(len(untrained_z)))
        t_te = pair_stats(trained_z, knn_t_te, gram_t, all_pairs(len(trained_z)))
        u_te = pair_stats(untrained_z, knn_u_te, gram_u, all_pairs(len(untrained_z)))
        report["arms"][label] = {
            "r": r, "data": data, "seeds": seeds,
            "n_trained_used": len(trained_z), "dropped_below_pr": dropped,
            "pr_per_seed": prs,
            "trained": tstats, "untrained": ustats,
            "trained_test_only": {"knn_mean": t_te["knn_mean"],
                                   "per_unit_mean_knn": t_te["per_unit_mean_knn"]},
            "untrained_test_only": {"knn_mean": u_te["knn_mean"]},
            "learned_knn": tstats["knn_mean"] - ustats["knn_mean"],
        }
        print(f"  trained knn {tstats['knn_mean']:.4f}  untrained {ustats['knn_mean']:.4f}"
              f"  learned {tstats['knn_mean'] - ustats['knn_mean']:+.4f}"
              f"  (cka {tstats['cka_mean']:.3f}/{ustats['cka_mean']:.3f})")

        Z[label] = {"trained": trained_z, "untrained": untrained_z}
        KNN[label] = {"trained": knn_t, "untrained": knn_u}
        KNN_TE[label] = {"trained": knn_t_te, "untrained": knn_u_te}
        GRAM[label] = {"trained": gram_t, "untrained": gram_u}

    # Cross-arm (section 5). Scene rows must correspond across arms.
    for a, b in CROSS:
        if a not in Z or b not in Z:
            continue
        assert scene_ids[a] == scene_ids[b], f"scene order differs: {a} vs {b}"
        na, nb = len(Z[a]["trained"]), len(Z[b]["trained"])
        seeds_a = report["arms"][a]["seeds"]
        seeds_b = report["arms"][b]["seeds"]
        knn_vals = {}
        for i in range(na):
            for j in range(nb):
                if seeds_a[i] == seeds_b[j]:
                    continue  # same seed = same init (train.py:143); excluded
                knn_vals[(i, j)] = overlap(KNN[a]["trained"][i], KNN[b]["trained"][j])
        u_vals = [overlap(KNN[a]["untrained"][i], KNN[b]["untrained"][j])
                  for i in range(N_UNTRAINED) for j in range(N_UNTRAINED) if i != j]
        # Per-seed mean cross-alignment, both sides (H2's paired statistic).
        per_seed_b = [float(np.mean([v for (i, j), v in knn_vals.items() if j == jj]))
                      for jj in range(nb)]
        per_seed_a = [float(np.mean([v for (i, j), v in knn_vals.items() if i == ii]))
                      for ii in range(na)]
        report["cross"][f"{a}|{b}"] = {
            "knn_mean": float(np.mean(list(knn_vals.values()))),
            "untrained_knn_mean": float(np.mean(u_vals)),
            "per_seed_mean_knn_a": per_seed_a,
            "per_seed_mean_knn_b": per_seed_b,
            "n_pairs": len(knn_vals),
        }
        print(f"cross {a}<->{b}: trained {np.mean(list(knn_vals.values())):.4f} "
              f"untrained {np.mean(u_vals):.4f}")

    # H3: trit agreement (section 6).
    for label, r, data, _prefix, cks in plan[len(ARMS):]:
        print(f"=== fsq set {label} (r={r:.3f}) ===")
        imgs, _te, ds = encode_arm(data)
        codes = []
        for ck in cks:
            m, cfg = build_from_metrics(ck)
            assert cfg.latent_trits == 1, f"{ck} is not a k=1 FSQ checkpoint"
            codes.append(encode_codes(m, imgs))
            del m
        del imgs
        mx.clear_cache()

        shapes = [s.shape for s in ds.scenes]
        shape_labels = np.array([sorted(set(shapes)).index(s) for s in shapes])
        factor_vals = {f: [s.record["material_truth"]["factors"][f]
                           for s in ds.scenes] for f in FACTORS}
        pairs_out = []
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                agree, perm = trit_agreement(codes[i], codes[j])
                null = trit_agreement_null(codes[i], codes[j], seed=PERM_SEED)
                pairs_out.append({
                    "pair": [i, j], "agreement": agree,
                    "perm": {str(k): v for k, v in perm.items()},
                    "null_mean": float(null.mean()),
                    "null_p95": float(np.quantile(null, 0.95)),
                })
        mi = {"shape": [float(mutual_information(c, shape_labels)) for c in codes]}
        for f in FACTORS:
            lab = terciles(factor_vals[f])
            mi[f] = [float(mutual_information(c, lab)) for c in codes]
        code_marginals = [np.bincount(c + 1, minlength=3).tolist() for c in codes]
        report["fsq"][label] = {"r": r, "data": data, "n_seeds": len(codes),
                                "pairs": pairs_out, "mi": mi,
                                "code_marginals": code_marginals}
        for p in pairs_out:
            print(f"  pair {p['pair']}: agree {p['agreement']:.3f} "
                  f"(null {p['null_mean']:.3f}, p95 {p['null_p95']:.3f})")
        print(f"  MI shape {np.mean(mi['shape']):.3f} | " +
              " ".join(f"{f} {np.mean(mi[f]):.3f}" for f in FACTORS))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
