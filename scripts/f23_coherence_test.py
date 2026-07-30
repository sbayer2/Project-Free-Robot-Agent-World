#!/usr/bin/env python
"""H1: does coupling strength buy COHERENCE, not just gain? (docs/COHERENCE_STRENGTH.md)

PRIMARY = learned_coherence(r=0.636) - learned_coherence(r=0.263)

Both arms are measured IN THE SAME RUN with the untrained baseline recomputed per
arm. F13's +0.146 is a VALIDITY CHECK on the r=0.263 arm, not the comparison
target -- section 3 of the preregistration explains why the cross-world
comparison was rejected (it is the error corrected in ABO_COUPLING.md A5).

This script DRIVES scripts/run_coherence_experiment.py rather than calling
coherence_bench.compare directly: that helper assumes a `.encode` method the MLX
model does not expose (see run_coherence_experiment.py:16-19). Reimplementing the
metric here would measure something subtly different from every historical
number, which is precisely what the validity gate exists to catch.

DECISION RULE, fixed before any run:

    COHERENCE RISES  PRIMARY >= +0.10 AND Welch t >= 2.5
                     (a third condition, ">=75% of seeds above reference", was
                     registered but proved UNCOMPUTABLE -- the driver emits only
                     summary stats. Dropped, not approximated; see main(). This
                     loosens RISES, so it cannot have produced the NULL.)
    NULL             |PRIMARY| < 0.05
    INCONCLUSIVE     otherwise

GATES -- all must pass before the verdict is read:
    collapse  latent participation ratio >= 8 (F12 healthy band) on both arms.
              F10: a COLLAPSED encoder scores HIGHER raw coherence, so this
              guard is what stops the degenerate solution being rewarded.
    control   |independent disjoint-latent baseline| < 0.05 on both arms. Needs
              render-only and behavior-only checkpoints (masked-loss training);
              if absent the control gate is UNMET and the verdict is withheld.
    validity  r=0.263 learned_coherence in [0.08, 0.22], bracketing F13's +0.146.
              Outside that the harness has drifted and the run is VOID.

Usage:
    .venv/bin/python scripts/f23_coherence_test.py            # after the runs exist
    .venv/bin/python scripts/f23_coherence_test.py --dry-run  # show what it will do
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess

# (label, r, dataset, trained tag, render-only tag, behavior-only tag)
ARMS = [
    ("r=0.263 (reference)", 0.263, "data/pm_f22_base", "base_cont",
     "base_renderonly", "base_behavonly"),
    ("r=0.636 (strong)", 0.636, "data/pm_f22_g15", "g15_cont",
     "g15_renderonly", "g15_behavonly"),
]
TARGET = "behavior"
N_UNTRAINED = 5
PR_FLOOR = 8.0
CONTROL_MAX = 0.05
VALIDITY_BAND = (0.08, 0.22)
RISE_MIN, RISE_T, NULL_MAX = 0.10, 2.5, 0.05
PY = ".venv/bin/python"


def welch_t_summary(a, b):
    """Welch t from summary stats {mean, sd, n} -- the driver does not emit
    per-seed values, so this is the only computable form (see the note in main)."""
    if a["n"] < 2 or b["n"] < 2:
        return float("nan")
    se = math.sqrt(a["sd"] ** 2 / a["n"] + b["sd"] ** 2 / b["n"])
    return (a["mean"] - b["mean"]) / se if se > 0 else float("nan")


def run_arm(label, data, tag, ro_tag, bo_tag, out_dir, dry=False):
    cks = sorted(glob.glob(f"runs/f22/{tag}_s*/model.safetensors"))
    if not cks:
        print(f"{label}: no trained checkpoints for tag {tag}")
        return None
    cmd = [PY, "scripts/run_coherence_experiment.py", "--data", data,
           "--checkpoints", ",".join(cks), "--untrained-seeds", str(N_UNTRAINED),
           "--image-size", "128", "--out", out_dir]
    ro = f"runs/f23/{ro_tag}/model.safetensors"
    bo = f"runs/f23/{bo_tag}/model.safetensors"
    if os.path.exists(ro) and os.path.exists(bo):
        cmd += ["--render-only", ro, "--physics-only", bo]
    else:
        print(f"  ! {label}: independent-baseline checkpoints missing "
              f"({ro_tag}/{bo_tag}) -- CONTROL GATE WILL BE UNMET")
    if dry:
        print("  would run:", " ".join(cmd[:6]), "... --out", out_dir)
        return None
    subprocess.run(cmd, check=True, capture_output=True)
    return json.load(open(os.path.join(out_dir, "coherence_report.json")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs("runs/f23", exist_ok=True)
    report: dict = {"rule": {"primary": "learned(r=0.636) - learned(r=0.263)",
                             "rise_min": RISE_MIN, "rise_t": RISE_T,
                             "null_max": NULL_MAX, "pr_floor": PR_FLOOR,
                             "control_max": CONTROL_MAX,
                             "validity_band": list(VALIDITY_BAND)},
                    "target": TARGET, "arms": {}}
    learned, gates_ok, gate_notes = {}, True, []

    for label, r, data, tag, ro, bo in ARMS:
        res = run_arm(label, data, tag, ro, bo, f"runs/f23/coh_{tag}", args.dry_run)
        if res is None:
            if args.dry_run:
                continue
            return
        t = res["targets"][TARGET]
        arch_mean = t["architectural_baseline"]["mean"]
        # run_coherence_experiment's agg() emits mean/std/min/max/n and NOT the
        # per-seed values, so the original per-seed Welch t and seed-consistency
        # conditions were UNCOMPUTABLE (they returned nan and 0/0). The NULL
        # branch never depended on them, so the 2026-07-30 verdict stands; but a
        # rise could not have been graded. Summary statistics are sufficient for
        # Welch, so derive it from (mean, std, n). `learned` shifts every seed by
        # the same per-arm constant, so its variance equals trained_shared's.
        ts = t["trained_shared"]
        learned[label] = {"mean": ts["mean"] - arch_mean, "sd": ts["std"],
                          "n": ts["n"]}
        pr = res["latent_participation"]["trained_mean"]
        indep = t.get("independent_baseline")

        report["arms"][label] = {
            "r": r, "learned_coherence": t["learned_coherence"],
            "trained_shared": t["trained_shared"], "architectural": t["architectural_baseline"],
            "independent_baseline": indep, "latent_pr_trained": pr,
            "learned_summary": learned[label]}
        print(f"{label:22s} learned {t['learned_coherence']:+.4f}  "
              f"arch {arch_mean:+.4f}  PR {pr:.1f}  "
              f"indep {indep if indep is None else round(indep, 4)}")

        if pr < PR_FLOOR:
            gate_notes.append(f"COLLAPSE gate failed on {label}: PR {pr:.1f} < {PR_FLOOR}")
            gates_ok = False
        if indep is None:
            gate_notes.append(f"CONTROL gate UNMET on {label}: no independent baseline "
                              f"(train render-only + behavior-only checkpoints)")
            gates_ok = False
        elif abs(indep) >= CONTROL_MAX:
            gate_notes.append(f"CONTROL gate failed on {label}: |indep| {abs(indep):.3f}")
            gates_ok = False

    if args.dry_run:
        print("\n[dry-run] no verdict computed.")
        return

    ref, strong = ARMS[0][0], ARMS[1][0]
    lo, hi = VALIDITY_BAND
    ref_learned = report["arms"][ref]["learned_coherence"]
    if not (lo <= ref_learned <= hi):
        gate_notes.append(f"VALIDITY gate failed: r=0.263 learned {ref_learned:+.4f} "
                          f"outside [{lo}, {hi}] -- harness drifted from F13")
        gates_ok = False

    primary = report["arms"][strong]["learned_coherence"] - ref_learned
    a, b = learned[strong], learned[ref]
    t_stat = welch_t_summary(a, b)
    n = a["n"]
    print(f"\nPRIMARY = {primary:+.4f}   Welch t {t_stat:.2f}   "
          f"(n={a['n']} vs {b['n']}, sd {a['sd']:.4f} / {b['sd']:.4f})")
    for g in gate_notes:
        print(f"  !! {g}")

    if not gates_ok:
        verdict = "VOID -- a gate failed or was unmet; no verdict may be read."
    # The frozen rule's third condition (">=75% of seeds above reference") is not
    # computable from the driver's summary output and is therefore DROPPED rather
    # than silently approximated. Both surviving conditions are unchanged, and
    # dropping a condition can only make RISES EASIER -- so this cannot have
    # manufactured the 2026-07-30 NULL, which came from |PRIMARY| < 0.05 anyway.
    elif primary >= RISE_MIN and t_stat >= RISE_T:
        verdict = ("COHERENCE RISES -- a world with real signal produces a MORE "
                   "UNIFIED latent, not merely a better predictor. The founding "
                   "thesis survives its sharpest test.")
    elif abs(primary) < NULL_MAX:
        verdict = ("NULL -- prediction improves through pathways that do NOT unify. "
                   "A direct negative on 'one latent, two projections', and sharper "
                   "than F18-F21 because the signal is proven present.")
    else:
        verdict = "INCONCLUSIVE -- below the prefixed bar. Report; do not claim."

    print(f"\n[H1] {verdict}")
    report.update({"primary": primary, "welch_t": t_stat, "n_seeds": n,
                   "gates_ok": gates_ok, "gate_notes": gate_notes, "verdict": verdict,
                   "note": ("H2 is registered separately in f23_extraction_test.py; "
                            "neither result may be cited as support for the other.")})
    json.dump(report, open("runs/f23/f23_coherence.json", "w"), indent=2)
    print("[H1] wrote runs/f23/f23_coherence.json")


if __name__ == "__main__":
    main()
