#!/usr/bin/env python
"""F26 measurement driver: the coherence-vs-coupling curve
(docs/COHERENCE_VS_R.md). Assembles learned coherence across the nine
continuous worlds (reusing the F23/F25 reports where the identical harness
already produced them) plus the three k=1 FSQ points, and writes
runs/f26/f26_report.json. NO verdicts here; the frozen rules live in
scripts/f26_verdicts.py.

Mac only (MLX). Usage:
    .venv/bin/python scripts/f26_measure.py            # full run
    .venv/bin/python scripts/f26_measure.py --dry-run  # list work, touch nothing
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess

PY = ".venv/bin/python"

# (tag, r) for the nine continuous arms; datasets are data/pm_f22_<tag> except
# base uses the pm_f22_base regeneration of pm_big. Checkpoints live at
# runs/f22/<tag>_cont_s*. Reused reports (identical harness) and pre-existing
# masked-loss control pairs are looked up in the dicts below; every other arm
# gets fresh controls under runs/f26/.
CONT_ARMS = [
    ("ctrl", 0.000), ("base", 0.263), ("g11", 0.357), ("g13", 0.517),
    ("g15", 0.636), ("g2", 0.809), ("g3", 0.927), ("g5", 0.978),
    ("loud", 0.992),
]
REUSE = {"base": "runs/f23/coh_base_cont", "g15": "runs/f23/coh_g15_cont",
         "g2": "runs/f25/coh_cont"}
CONTROLS = {"base": "runs/f23/base", "g15": "runs/f23/g15", "g2": "runs/f25/g2"}


def cont_paths(tag: str):
    data = f"data/pm_f22_{tag}"
    prefix = f"runs/f22/{tag}_cont"
    ctrl_base = CONTROLS.get(tag, f"runs/f26/{tag}")
    controls = (f"{ctrl_base}_renderonly", f"{ctrl_base}_behavonly")
    return data, prefix, REUSE.get(tag), controls
# k=1 FSQ points (H2, descriptive). Inherit the world's continuous control
# (the F25 disclosed repair): controls entry is None.
FSQ_ARMS = [
    ("fsq_base", 0.263, "data/pm_big", "runs/fsq/k1", None),
    ("fsq_g2", 0.809, "data/pm_f22_g2", "runs/f25/g2_fsq1", "runs/f25/coh_fsq1"),
    ("fsq_loud", 0.992, "data/pm_f22_loud", "runs/f22/loud_fsq1", None),
]


def coherence_for(tag: str, data: str, cks: list, reuse: str | None,
                  controls, trits: int, dry: bool):
    if reuse:
        path = os.path.join(reuse, "coherence_report.json")
        if os.path.exists(path):
            if not dry:
                print(f"  reusing {path}")
            return json.load(open(path)) if not dry else None
        print(f"  ! reuse dir {reuse} missing; measuring fresh")
    out_dir = f"runs/f26/coh_{tag}"
    cached = os.path.join(out_dir, "coherence_report.json")
    if not dry and os.path.exists(cached):
        print(f"  reusing {cached}")
        return json.load(open(cached))
    cmd = [PY, "scripts/run_coherence_experiment.py", "--data", data,
           "--checkpoints", ",".join(cks), "--untrained-seeds", "5",
           "--image-size", "128", "--out", out_dir]
    if trits:
        cmd += ["--latent-trits", str(trits)]
    elif controls:
        ro = os.path.join(controls[0], "model.safetensors")
        bo = os.path.join(controls[1], "model.safetensors")
        if os.path.exists(ro) and os.path.exists(bo):
            cmd += ["--render-only", ro, "--physics-only", bo]
        else:
            print(f"  ! {tag}: masked-loss controls missing -- CONTROL GATE UNMET")
    if dry:
        print(f"  would run harness -> {out_dir}")
        return None
    subprocess.run(cmd, check=True, capture_output=True)
    return json.load(open(cached))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/f26/f26_report.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    report: dict = {"cont": {}, "fsq_k1": {}}
    for tag, r in CONT_ARMS:
        data, prefix, reuse, controls = cont_paths(tag)
        cks = sorted(glob.glob(f"{prefix}_s*/model.safetensors"))
        print(f"arm {tag:5s} r={r:.3f}  {len(cks):2d} seeds")
        res = coherence_for(tag, data, cks, reuse, controls, 0, args.dry_run)
        if res is None:
            continue
        t = res["targets"]["behavior"]
        report["cont"][tag] = {
            "r": r, "n": t["trained_shared"]["n"],
            "learned": t["learned_coherence"],
            "trained_shared": t["trained_shared"],
            "architectural": t["architectural_baseline"],
            "independent_baseline": t.get("independent_baseline"),
            "essence_learned": res["targets"]["essence"]["learned_coherence"],
            "latent_pr_trained": res["latent_participation"]["trained_mean"],
        }
        print(f"  learned {t['learned_coherence']:+.4f}  "
              f"PR {res['latent_participation']['trained_mean']:.1f}")

    for tag, r, data, prefix, reuse in FSQ_ARMS:
        cks = sorted(glob.glob(f"{prefix}_s*/model.safetensors"))
        print(f"fsq {tag:9s} r={r:.3f}  {len(cks):2d} seeds")
        res = coherence_for(tag, data, cks, reuse, None, 1, args.dry_run)
        if res is None:
            continue
        t = res["targets"]["behavior"]
        report["fsq_k1"][tag] = {
            "r": r, "n": t["trained_shared"]["n"],
            "learned": t["learned_coherence"],
            "trained_shared": t["trained_shared"],
            "latent_pr_trained": res["latent_participation"]["trained_mean"],
        }
        print(f"  learned {t['learned_coherence']:+.4f}")

    if args.dry_run:
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
