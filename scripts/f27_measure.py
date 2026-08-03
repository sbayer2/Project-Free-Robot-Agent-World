#!/usr/bin/env python
"""F27 measurement driver: the shape-degenerate world (docs/SHAPE_DEGENERATE.md).

Per arm (ctrl/base/g2/loud on data/pm_f27_*): apparatus checks (all-box,
matched physics across arms, measured r), behavior gain vs train-mean +
appearance-oracle ceiling, learned coherence (F23/F25/F26 harness), and
cross-seed learned kNN alignment (F24 harness). Writes
runs/f27/f27_report.json. NO verdicts here; frozen rules live in
scripts/f27_verdicts.py.

Mac only (MLX). Usage:
    .venv/bin/python scripts/f27_measure.py            # full run
    .venv/bin/python scripts/f27_measure.py --dry-run  # list work, touch nothing
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
    knn_indices,
    participation_ratio,
)

ARMS = [("ctrl", "data/pm_f27_ctrl"), ("base", "data/pm_f27_base"),
        ("g2", "data/pm_f27_g2"), ("loud", "data/pm_f27_loud")]
K = 10
N_UNTRAINED_ALIGN = 8
UNTRAINED_SEED_BASE = 1000
PY = ".venv/bin/python"


def sample_paths(data: str):
    return sorted(glob.glob(f"{data}/train_*/sample.json")
                  + glob.glob(f"{data}/test_*/sample.json"))


def apparatus_checks() -> dict:
    """All-box, matched physics across arms, measured r per arm."""
    out: dict = {"all_box": {}, "r": {}, "physics_matched": None}
    phys: dict = {}
    for tag, data in ARMS:
        shapes, rough, fric = set(), [], []
        rows = {}
        for p in sample_paths(data):
            s = json.load(open(p))
            shapes.add(s["input"]["shape"])
            rough.append(s["material_truth"]["appearance_params"]["roughness"])
            fric.append(s["physics"]["raw"]["friction"])
            rows[s["scene_id"]] = (s["physics"]["raw"]["density"],
                                   s["physics"]["raw"]["friction"],
                                   s["physics"]["raw"]["restitution"])
        out["all_box"][tag] = sorted(shapes)
        out["r"][tag] = float(np.corrcoef(rough, fric)[0, 1])
        phys[tag] = rows
    ref = phys[ARMS[0][0]]
    matched = all(phys[tag] == ref for tag, _ in ARMS[1:])
    out["physics_matched"] = matched
    return out


def run_coherence(root: str, tag: str, cks: list, dry: bool):
    out_dir = f"{root}/coh_{tag}"
    cached = os.path.join(out_dir, "coherence_report.json")
    if not dry and os.path.exists(cached):
        print(f"  coherence: reusing {cached}")
        return json.load(open(cached))
    cmd = [PY, "scripts/run_coherence_experiment.py", "--data", f"data/pm_f27_{tag}",
           "--checkpoints", ",".join(cks), "--untrained-seeds", "5",
           "--image-size", "128", "--out", out_dir]
    ro = f"{root}/{tag}_renderonly/model.safetensors"
    bo = f"{root}/{tag}_behavonly/model.safetensors"
    if os.path.exists(ro) and os.path.exists(bo):
        cmd += ["--render-only", ro, "--physics-only", bo]
    else:
        print(f"  ! {tag}: masked-loss controls missing -- CONTROL GATE UNMET")
    if dry:
        print(f"  would run harness -> {out_dir}")
        return None
    subprocess.run(cmd, check=True, capture_output=True)
    return json.load(open(cached))


def overlap(ix: np.ndarray, iy: np.ndarray) -> float:
    return float(np.mean([len(set(a.tolist()) & set(b.tolist())) / ix.shape[1]
                          for a, b in zip(ix, iy, strict=True)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/f27",
                    help="checkpoint/control/report root (F27b uses runs/f27b)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = args.root
    out_path = args.out or f"{root}/f27_report.json"

    plan = []
    for tag, data in ARMS:
        cks = sorted(glob.glob(f"{root}/{tag}_s*/model.safetensors"))
        plan.append((tag, data, cks))
        print(f"{tag:5s} {data:18s} {len(cks)} checkpoints")
    if args.dry_run:
        for tag, _data, cks in plan:
            run_coherence(root, tag, cks, dry=True)
        return

    import mlx.core as mx  # type: ignore
    from oracle_ceiling import best_oracle
    from oracle_ceiling import gain as gain_ratio
    from probe_appearance import encode_z
    from render_fidelity_eval import behavior_from_z, load_arrays

    from pseudomarble.config import ModelConfig
    from pseudomarble.models.mlx_net import build_model

    print("apparatus checks...")
    report: dict = {"apparatus": apparatus_checks(), "arms": {}}
    for tag, checks in report["apparatus"]["all_box"].items():
        print(f"  {tag}: shapes={checks} r={report['apparatus']['r'][tag]:+.3f}")
    print(f"  physics matched across arms: {report['apparatus']['physics_matched']}")

    for tag, data, cks in plan:
        if not cks:
            print(f"=== {tag}: NO CHECKPOINTS, skipped ===")
            continue
        print(f"=== arm {tag} ===")
        imgs, Yb, Ya, tr, te, _ds = load_arrays(data)
        app_ceiling = best_oracle(Ya[tr], Yb[tr], Ya[te], Yb[te])[0]

        gains, prs, zs = [], [], []
        for ck in cks:
            with open(os.path.join(os.path.dirname(ck), "metrics.json")) as f:
                cfg_d = json.load(f)["config"]
            cfg_d["conv_channels"] = tuple(cfg_d["conv_channels"])
            known = ModelConfig.__dataclass_fields__.keys()
            m = build_model(ModelConfig(**{k: v for k, v in cfg_d.items() if k in known}))
            m.load_weights(ck)
            z = encode_z(m, imgs)
            prs.append(participation_ratio(z))
            gains.append(gain_ratio(Yb[tr], Yb[te], behavior_from_z(m, z)[te]))
            zs.append(z)
            del m
        untrained_z = []
        for i in range(N_UNTRAINED_ALIGN):
            mx.random.seed(UNTRAINED_SEED_BASE + i)
            m = build_model(ModelConfig(image_size=128))
            untrained_z.append(encode_z(m, imgs))
            del m
        del imgs
        mx.clear_cache()

        knn_t = [knn_indices(z, K) for z in zs]
        knn_u = [knn_indices(z, K) for z in untrained_z]
        t_align = float(np.mean([overlap(knn_t[i], knn_t[j])
                                 for i in range(len(zs)) for j in range(i + 1, len(zs))]))
        u_align = float(np.mean([overlap(knn_u[i], knn_u[j])
                                 for i in range(len(knn_u))
                                 for j in range(i + 1, len(knn_u))]))

        arm = {
            "r": report["apparatus"]["r"][tag], "n_seeds": len(cks),
            "gain_per_seed": [float(g) for g in gains],
            "gain_mean": float(np.mean(gains)),
            "appearance_oracle_gain": float(app_ceiling),
            "pr_per_seed": prs,
            "align_knn_trained": t_align, "align_knn_untrained": u_align,
            "align_learned": t_align - u_align,
        }
        print(f"  gain {arm['gain_mean']:.3f} (ceiling {app_ceiling:.3f})  "
              f"align learned {arm['align_learned']:+.4f}  "
              f"PR {min(prs):.1f}-{max(prs):.1f}")

        coh = run_coherence(root, tag, cks, dry=False)
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
                  f"(untrained {t['architectural_baseline']['mean']:+.4f})  "
                  f"essence {arm['coherence']['essence_learned']:+.4f}")
        report["arms"][tag] = arm

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
