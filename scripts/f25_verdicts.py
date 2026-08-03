#!/usr/bin/env python
"""F25 frozen decision rules (docs/ARCHITECTED_UNITY.md section 5).

Reads runs/f25/f25_report.json (from scripts/f25_measure.py) and emits the
preregistered verdicts. Committed in the SAME commit as the preregistration,
before any F25 arm was trained -- editing after the report exists voids the
freeze.

    H1a  UNIFIED-WITHOUT-COST per FSQ k; SUCCESS if any k qualifies
    H1b  ENGINEERING-SUCCESS per coherence weight w (circular readout -- the
         verdict line carries the label)
    H2   OBJECTIVE (consensus from the coherence objective) + BUDGET (ARI k=1)
    H3   dose-response monotonicity across k (descriptive)

Usage:  .venv/bin/python scripts/f25_verdicts.py [--report runs/f25/f25_report.json]
"""
from __future__ import annotations

import argparse
import json
import math

# Frozen constants (do not edit after the freeze commit).
COH_RISE, COH_T = 0.10, 2.5
DELTA_TOL = 0.15
PR_FLOOR = 8.0
CONTROL_MAX = 0.05
H2_MIN, H2_T = 0.03, 2.5
ARI_K1_MIN = 0.5
FSQ_KS = ["fsq1", "fsq2", "fsq4", "fsq8"]
COHWS = ["cohw03", "cohw10", "cohw30"]


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0


def welch_t_summary(a, b):
    """(mean, sd, n) dicts -> Welch t. Coherence agg emits summary stats only;
    `learned` shifts every seed by a per-arm constant so trained_shared's sd is
    the right spread (the F23 repair, reused)."""
    if a["n"] < 2 or b["n"] < 2:
        return float("nan")
    se = math.sqrt(a["sd"] ** 2 / a["n"] + b["sd"] ** 2 / b["n"])
    return (a["mean"] - b["mean"]) / se if se > 0 else float("nan")


def welch_t(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = math.sqrt(sd(a) ** 2 / len(a) + sd(b) ** 2 / len(b))
    return (mean(a) - mean(b)) / se if se > 0 else float("nan")


def coh_summary(arm):
    c = arm["coherence"]
    ts = c["trained_shared"]
    return {"mean": c["learned"], "sd": ts["std"], "n": ts["n"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="runs/f25/f25_report.json")
    args = ap.parse_args()
    with open(args.report) as f:
        rep = json.load(f)
    arms = rep["arms"]

    # ---- gates ---------------------------------------------------------- #
    gates_ok = True
    for label, arm in arms.items():
        notes = []
        if min(arm["pr_per_seed"]) < PR_FLOOR:
            notes.append(f"PR {min(arm['pr_per_seed']):.1f} < {PR_FLOOR}")
        c = arm.get("coherence")
        if c is None:
            notes.append("coherence not measured")
        else:
            indep = c.get("independent_baseline")
            if indep is None:
                notes.append("CONTROL gate UNMET (no masked-loss baselines)")
            elif abs(indep) >= CONTROL_MAX:
                notes.append(f"CONTROL gate failed |indep| {abs(indep):.3f}")
        status = "PASS" if not notes else "FAIL: " + "; ".join(notes)
        print(f"gate {label:8s} {status}")
        if notes:
            gates_ok = False
    print(f"GATES: {'ALL PASS' if gates_ok else 'FAILED -- affected verdicts withheld'}")

    cont = arms["cont"]
    coh_cont = coh_summary(cont)
    d_cont = cont["delta_mean"]
    print(f"\nreferences (cont, n={cont['n_seeds']}): learned coherence "
          f"{coh_cont['mean']:+.4f}, delta {d_cont:+.3f}, "
          f"align learned {cont['align_learned']:+.4f}")

    # ---- H1a: the FSQ frontier ------------------------------------------ #
    print("\nH1a -- UNIFIED-WITHOUT-COST per FSQ k "
          f"(coh rise >= +{COH_RISE} t >= {COH_T}; delta >= cont - {DELTA_TOL}):")
    any_k, all_measured = False, True
    for label in FSQ_KS:
        arm = arms.get(label)
        if arm is None or "coherence" not in arm:
            print(f"  {label}: not measured")
            all_measured = False
            continue
        cs = coh_summary(arm)
        rise = cs["mean"] - coh_cont["mean"]
        t = welch_t_summary(cs, coh_cont)
        ok_coh = rise >= COH_RISE and t >= COH_T
        ok_delta = arm["delta_mean"] >= d_cont - DELTA_TOL
        q = ok_coh and ok_delta
        any_k = any_k or q
        print(f"  {label}: coh {cs['mean']:+.4f} (rise {rise:+.4f}, t {t:.2f}) "
              f"delta {arm['delta_mean']:+.3f} -> "
              f"{'UNIFIED-WITHOUT-COST' if q else 'no'}"
              f"{'' if ok_coh else ' [coherence bar missed]'}"
              f"{'' if ok_delta else ' [delta cost too high]'}")
    h1a = "SUCCESS" if any_k else ("FAILURE" if all_measured else "INCONCLUSIVE")
    print(f"H1a -> {h1a}")

    # ---- H1b: the coherence objective (circular; engineering) ----------- #
    print("\nH1b -- ENGINEERING-SUCCESS per w (readout is trained-on; "
          "engineering evidence, not discovery):")
    for label in COHWS:
        arm = arms.get(label)
        if arm is None or "coherence" not in arm:
            print(f"  {label}: not measured")
            continue
        cs = coh_summary(arm)
        rise = cs["mean"] - coh_cont["mean"]
        t = welch_t_summary(cs, coh_cont)
        ok = (rise >= COH_RISE and t >= COH_T
              and arm["delta_mean"] >= d_cont - DELTA_TOL
              and min(arm["pr_per_seed"]) >= PR_FLOOR)
        print(f"  {label}: coh {cs['mean']:+.4f} (rise {rise:+.4f}, t {t:.2f}) "
              f"delta {arm['delta_mean']:+.3f} PRmin {min(arm['pr_per_seed']):.1f} "
              f"essence {arm['coherence']['essence_learned']:+.4f} -> "
              f"{'ENGINEERING-SUCCESS' if ok else 'no'}")

    # ---- H2 -------------------------------------------------------------- #
    print(f"\nH2-OBJECTIVE (align rise >= +{H2_MIN}, t >= {H2_T}; "
          "3 arms tested -- multiplicity noted):")
    h2_obj = False
    for label in COHWS:
        arm = arms.get(label)
        if arm is None or "align_per_seed_mean" not in arm:
            continue
        rise = arm["align_learned"] - cont["align_learned"]
        t = welch_t(arm["align_per_seed_mean"], cont["align_per_seed_mean"])
        ok = rise >= H2_MIN and t >= H2_T
        h2_obj = h2_obj or ok
        print(f"  {label}: align learned {arm['align_learned']:+.4f} "
              f"(rise {rise:+.4f}, t {t:.2f}, n={arm['n_seeds']}) -> "
              f"{'PASS' if ok else 'no'}")
    print(f"H2-OBJECTIVE -> {'PASS' if h2_obj else 'FAIL'}")

    f1 = arms.get("fsq1", {})
    if "ari_mean" in f1:
        ok = f1["ari_mean"] > ARI_K1_MIN
        print(f"H2-BUDGET: ARI(k=1) {f1['ari_mean']:.3f} "
              f"{'>' if ok else '<='} {ARI_K1_MIN} -> {'PASS' if ok else 'FAIL'}")

    # ---- H3: dose-response (descriptive) --------------------------------- #
    series = {"coherence": [], "delta": [], "ari": []}
    for label in FSQ_KS:
        arm = arms.get(label)
        if arm is None:
            continue
        series["coherence"].append(arm.get("coherence", {}).get("learned"))
        series["delta"].append(arm.get("delta_mean"))
        series["ari"].append(arm.get("ari_mean"))
    print("\nH3 dose-response across k = 1,2,4,8 (descriptive):")
    for name, vals in series.items():
        if any(v is None for v in vals) or len(vals) < 4:
            print(f"  {name}: incomplete")
            continue
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        direction = ("monotone down" if all(d < 0 for d in diffs)
                     else "monotone up" if all(d > 0 for d in diffs) else "mixed")
        print(f"  {name}: " + " ".join(f"{v:+.3f}" for v in vals) + f" -> {direction}")


if __name__ == "__main__":
    main()
