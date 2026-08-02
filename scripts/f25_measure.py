#!/usr/bin/env python
"""F25 measurement driver: architected unity in the strong world
(docs/ARCHITECTED_UNITY.md). Runs the full triad on every arm --

    Delta      held-out gain minus the shape-only oracle (F22 helpers)
    coherence  learned render<->behavior coherence (F13/F23 harness, subprocess)
    consensus  kNN alignment for continuous arms (F24 harness) or adjusted
               Rand index over code partitions for FSQ arms

-- and writes runs/f25/f25_report.json. NO verdicts here; the frozen rules
live in scripts/f25_verdicts.py.

Mac only (MLX). Usage:
    .venv/bin/python scripts/f25_measure.py            # full run
    .venv/bin/python scripts/f25_measure.py --dry-run  # list work, touch nothing
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from pseudomarble.models.alignment import (  # noqa: E402
    adjusted_rand_index,
    knn_indices,
    participation_ratio,
    trit_agreement,
    trit_agreement_null,
)

DATA = "data/pm_f22_g2"  # r = 0.809, the strong world (prereg section 2)
# (label, kind, checkpoint glob prefix)
ARMS = [
    ("cont", "cont", "runs/f22/g2_cont"),
    ("fsq1", "fsq", "runs/f25/g2_fsq1"),
    ("fsq2", "fsq", "runs/f25/g2_fsq2"),
    ("fsq4", "fsq", "runs/f25/g2_fsq4"),
    ("fsq8", "fsq", "runs/f25/g2_fsq8"),
    ("cohw03", "cohw", "runs/f25/g2_cohw03"),
    ("cohw10", "cohw", "runs/f25/g2_cohw10"),
    ("cohw30", "cohw", "runs/f25/g2_cohw30"),
]
RENDER_ONLY = "runs/f25/g2_renderonly/model.safetensors"
BEHAV_ONLY = "runs/f25/g2_behavonly/model.safetensors"
K = 10
N_UNTRAINED_ALIGN = 8
UNTRAINED_SEED_BASE = 1000
PERM_SEED = 2026
PY = ".venv/bin/python"


def build_from_metrics(ck_path: str):
    from pseudomarble.config import ModelConfig
    from pseudomarble.models.mlx_net import build_model

    with open(os.path.join(os.path.dirname(ck_path), "metrics.json")) as f:
        cfg_dict = json.load(f)["config"]
    cfg_dict["conv_channels"] = tuple(cfg_dict["conv_channels"])
    known = ModelConfig.__dataclass_fields__.keys()
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in known})
    m = build_model(cfg)
    m.load_weights(ck_path)
    return m, cfg


def encode_codes(model, imgs, chunk: int = 64) -> np.ndarray:
    """Chunked FSQ code extraction -> (B, k) ints in {-1,0,1}."""
    import mlx.core as mx  # type: ignore

    rows = []
    for i in range(0, imgs.shape[0], chunk):
        code = model(imgs[i:i + chunk])["code"]
        mx.eval(code)
        rows.append(np.array(code.tolist(), np.float32))
    return np.rint(np.concatenate(rows, axis=0)).astype(np.int64)


def overlap(ix: np.ndarray, iy: np.ndarray) -> float:
    return float(np.mean([len(set(a.tolist()) & set(b.tolist())) / ix.shape[1]
                          for a, b in zip(ix, iy, strict=True)]))


def run_coherence(label: str, cks: list, trits: int, dry: bool):
    """Drive scripts/run_coherence_experiment.py exactly as F23 did."""
    out_dir = f"runs/f25/coh_{label}"
    cmd = [PY, "scripts/run_coherence_experiment.py", "--data", DATA,
           "--checkpoints", ",".join(cks), "--untrained-seeds", "5",
           "--image-size", "128", "--out", out_dir]
    if trits:
        cmd += ["--latent-trits", str(trits)]
    if os.path.exists(RENDER_ONLY) and os.path.exists(BEHAV_ONLY):
        cmd += ["--render-only", RENDER_ONLY, "--physics-only", BEHAV_ONLY]
    else:
        print(f"  ! {label}: masked-loss controls missing -- CONTROL GATE UNMET")
    if dry:
        print("  would run:", " ".join(cmd[:6]), "... --out", out_dir)
        return None
    subprocess.run(cmd, check=True, capture_output=True)
    return json.load(open(os.path.join(out_dir, "coherence_report.json")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/f25/f25_report.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = []
    for label, kind, prefix in ARMS:
        cks = sorted(glob.glob(f"{prefix}_s*/model.safetensors"))
        plan.append((label, kind, cks))
        print(f"{label:8s} {kind:5s} {len(cks):2d} checkpoints")
    if args.dry_run:
        for label, kind, cks in plan:
            trits = int(label[3:]) if kind == "fsq" else 0
            run_coherence(label, cks, trits, dry=True)
        return

    import mlx.core as mx  # type: ignore
    from oracle_ceiling import best_oracle
    from oracle_ceiling import gain as gain_ratio
    from probe_appearance import encode_z
    from render_fidelity_eval import behavior_from_z, load_arrays, shape_onehot

    from pseudomarble.config import ModelConfig
    from pseudomarble.models.mlx_net import build_model

    report: dict = {"data": DATA, "k": K, "arms": {}}
    imgs, Yb, _ya, tr, te, ds = load_arrays(DATA)
    S = shape_onehot(ds)
    shape_gain = best_oracle(S[tr], Yb[tr], S[te], Yb[te], quad=False)[0]
    report["shape_oracle_gain"] = float(shape_gain)
    print(f"shape-only oracle: {shape_gain:.3f}")

    # Untrained alignment baseline on this world (F24 convention), once.
    untrained_z = []
    for i in range(N_UNTRAINED_ALIGN):
        mx.random.seed(UNTRAINED_SEED_BASE + i)
        m = build_model(ModelConfig(image_size=128))
        untrained_z.append(encode_z(m, imgs))
        del m
    knn_u = [knn_indices(z, K) for z in untrained_z]
    u_align = float(np.mean([overlap(knn_u[i], knn_u[j])
                             for i in range(len(knn_u))
                             for j in range(i + 1, len(knn_u))]))
    report["untrained_align_knn"] = u_align
    print(f"untrained alignment floor: {u_align:.4f}")

    for label, kind, cks in plan:
        if not cks:
            print(f"=== {label}: NO CHECKPOINTS, skipped ===")
            continue
        print(f"=== arm {label} ({kind}) ===")
        trits = int(label[3:]) if kind == "fsq" else 0

        gains, prs, zs, codes = [], [], [], []
        for ck in cks:
            m, cfg = build_from_metrics(ck)
            z = encode_z(m, imgs)
            prs.append(participation_ratio(z))
            gains.append(gain_ratio(Yb[tr], Yb[te], behavior_from_z(m, z)[te]))
            if kind == "fsq":
                codes.append(encode_codes(m, imgs))
            else:
                zs.append(z)
            del m
        mx.clear_cache()

        arm: dict = {
            "kind": kind, "n_seeds": len(cks),
            "delta_per_seed": [float(g - shape_gain) for g in gains],
            "delta_mean": float(np.mean(gains) - shape_gain),
            "gain_mean": float(np.mean(gains)),
            "pr_per_seed": prs,
        }

        if kind in ("cont", "cohw"):
            knn_t = [knn_indices(z, K) for z in zs]
            pair_vals = {(i, j): overlap(knn_t[i], knn_t[j])
                         for i in range(len(zs)) for j in range(i + 1, len(zs))}
            per_seed = [float(np.mean([v for (i, j), v in pair_vals.items()
                                       if u in (i, j)])) for u in range(len(zs))]
            arm["align_knn_trained_mean"] = float(np.mean(list(pair_vals.values())))
            arm["align_per_seed_mean"] = per_seed
            arm["align_learned"] = arm["align_knn_trained_mean"] - u_align
            print(f"  delta {arm['delta_mean']:+.3f}  align learned "
                  f"{arm['align_learned']:+.4f}  PR {min(prs):.1f}-{max(prs):.1f}")
        else:
            aris = [float(adjusted_rand_index(codes[i], codes[j]))
                    for i in range(len(codes)) for j in range(i + 1, len(codes))]
            arm["ari_per_pair"] = aris
            arm["ari_mean"] = float(np.mean(aris))
            if trits == 1:  # F24-comparable statistic
                pairs = []
                for i in range(len(codes)):
                    for j in range(i + 1, len(codes)):
                        agree, _ = trit_agreement(codes[i][:, 0], codes[j][:, 0])
                        null = trit_agreement_null(codes[i][:, 0], codes[j][:, 0],
                                                   seed=PERM_SEED)
                        pairs.append({"agreement": agree,
                                      "null_mean": float(null.mean()),
                                      "null_p95": float(np.quantile(null, 0.95))})
                arm["trit_agreement"] = pairs
            print(f"  delta {arm['delta_mean']:+.3f}  ARI {arm['ari_mean']:.3f}  "
                  f"PR {min(prs):.1f}-{max(prs):.1f}")

        coh = run_coherence(label, cks, trits, dry=False)
        if coh is not None:
            t = coh["targets"]["behavior"]
            arm["coherence"] = {
                "learned": t["learned_coherence"],
                "trained_shared": t["trained_shared"],
                "architectural": t["architectural_baseline"],
                "independent_baseline": t.get("independent_baseline"),
                "essence_learned": coh["targets"]["essence"]["learned_coherence"],
                "latent_pr_trained": coh["latent_participation"]["trained_mean"],
            }
            print(f"  coherence learned {t['learned_coherence']:+.4f}  "
                  f"essence {arm['coherence']['essence_learned']:+.4f}")
        report["arms"][label] = arm

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
