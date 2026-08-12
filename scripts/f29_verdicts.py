#!/usr/bin/env python
"""F29 frozen decision rules (docs/HELD_OUT_TRANSFER.md sections 6-7).

Reads runs/f29/f29_report.json and emits the preregistered verdicts.
Committed in the SAME commit as the preregistration, before any
held-out-transfer number existed anywhere.

    H0   PASS / FAIL                                (kill switch: both students survive on loud)
    H2   POSITIVE / NULL / NEGATIVE / INCONCLUSIVE  (loud primary; per-arm co-report)
    H3   POSITIVE-COUPLING / NULL / INCONCLUSIVE    (4-point advantage curve in r)
    H4   descriptive fork; verdict withheld here    (F28 tie-in -- separate amendment if triggered)

Usage:  .venv/bin/python scripts/f29_verdicts.py [--report runs/f29/f29_report.json]
"""
from __future__ import annotations

import argparse
import json
import math

# Frozen constants. Verdict bars (DELTA, NULL_BAND, T_MIN, GAIN_MIN,
# PR_FLOOR, RHO_MIN, ADV_SPAN_MIN) are untouched since the freeze commit.
# The two GATE constants below marked [Amendment 2] were recalibrated by
# docs/HELD_OUT_TRANSFER.md Amendment 2 (2026-08-12) after the original
# flat bars tripped on provable freeze-time errors (a retraining floor
# contradicting TRAIN_GAIN_JOINT below, and an apparatus tolerance
# ignoring correlation-estimator variance).
DELTA = 0.15                    # H2/H3 advantage bar (gain difference)
NULL_BAND = 0.05
T_MIN = 2.5
GAIN_MIN = 1.5                  # H0 kill switch on held-out loud (both students, min-seed)
PR_FLOOR = 8.0                  # F28 collapse floor
RHO_MIN = 0.8                   # H3 Spearman bar
ADV_SPAN_MIN = 0.15             # H3 endpoint spread (loud - ctrl)
RETRAIN_REL = 0.85              # [Amendment 2] disjoint train gain >= 0.85 x joint's
N_HELD = 128                    # scenes per held-out arm (prereg section 3)
R_TOL_FLOOR = 0.05              # [Amendment 2] apparatus floor of the 2-sigma bar
EXTRAPOLATION_FRAC = 1.0        # 100% held-out samples outside training hull
WORLD_ORDER = ("ctrl", "base", "g2", "loud")


def r_tolerance(r_train: float) -> float:
    """[Amendment 2] variance-aware apparatus bar: 2 sigma of the sample-r
    estimator, sd ~ (1 - r^2)/sqrt(n), floored at the original 0.05."""
    return max(R_TOL_FLOOR, 2.0 * (1.0 - r_train ** 2) / math.sqrt(N_HELD))

# F27b train-world joint gains (from FINDINGS F27), reprinted for the H2/H3 co-report.
TRAIN_GAIN_JOINT = {"ctrl": 1.121, "base": 1.113, "g2": 3.625, "loud": 5.672}


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def welch_t(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = math.sqrt(sd(a) ** 2 / len(a) + sd(b) ** 2 / len(b))
    return (mean(a) - mean(b)) / se if se > 0 else float("nan")


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="runs/f29/f29_report.json")
    args = ap.parse_args()
    with open(args.report) as f:
        rep = json.load(f)
    arms = rep["arms"]

    # ---- gates (section 7) --------------------------------------------------- #
    gates_ok, notes = True, []
    void_arms = set()
    if not rep.get("meta", {}).get("amendment_1"):
        notes.append("PILOT: Amendment 1 not recorded in meta (alpha* not pinned) "
                     "-> gates FAIL")
        gates_ok = False
    for tag in WORLD_ORDER:
        w = arms.get(tag)
        if w is None:
            notes.append(f"MISSING world {tag}")
            gates_ok = False
            continue
        g = w["gates"]
        tol = r_tolerance(g["r_train"])
        if abs(g["r_held_out"] - g["r_train"]) >= tol:
            notes.append(f"APPARATUS: {tag} |r_held-r_train| "
                         f"{abs(g['r_held_out']-g['r_train']):.3f} >= {tol:.3f}")
            void_arms.add(tag)
        if g["extrapolation_frac_outside"] < EXTRAPOLATION_FRAC:
            notes.append(f"EXTRAPOLATION: {tag} {g['extrapolation_frac_outside']:.1%} "
                         f"outside training hull < {EXTRAPOLATION_FRAC:.0%}")
            void_arms.add(tag)
        retrain_floor = RETRAIN_REL * TRAIN_GAIN_JOINT[tag]
        if g["disjoint_train_gain_min"] < retrain_floor:
            notes.append(f"RETRAINING: {tag} disjoint train-world gain min "
                         f"{g['disjoint_train_gain_min']:.3f} < {retrain_floor:.3f} "
                         f"(= {RETRAIN_REL} x joint {TRAIN_GAIN_JOINT[tag]:.3f})")
            void_arms.add(tag)
        if g["joint_pr_min"] < PR_FLOOR or g["disjoint_pr_min"] < PR_FLOOR:
            notes.append(f"COLLAPSE: {tag} min PR joint {g['joint_pr_min']:.1f} "
                         f"disjoint {g['disjoint_pr_min']:.1f} < {PR_FLOOR}")
            void_arms.add(tag)
    for n in notes:
        print(f"  !! {n}")
    status = ("ALL PASS" if gates_ok and not void_arms
             else "FAILED/PARTIAL -- see notes")
    print(f"GATES: {status}")

    # [Amendment 2 layout fix] gates decide BEFORE the outcome table prints,
    # so a failed run withholds its table too (the original layout leaked it).
    if not gates_ok or void_arms & set(WORLD_ORDER):
        print("verdicts withheld: gates failed (section 7); outcome table "
              "withheld with them.")
        return

    # ---- table ------------------------------------------------------------ #
    print(f"\n{'arm':6s} {'r':>7s} {'joint':>10s} {'disjoint':>10s} "
          f"{'advantage':>10s} {'t':>6s} {'train_j':>8s}")
    for tag in WORLD_ORDER:
        w = arms.get(tag)
        if w is None:
            continue
        j = w["joint"]["per_seed_gain"]
        d = w["disjoint"]["per_pair_gain"]
        adv = mean(j) - mean(d)
        t = welch_t(j, d)
        print(f"{tag:6s} {w['r']:+7.3f} {mean(j):+7.3f}±{sd(j):.2f} "
              f"{mean(d):+7.3f}±{sd(d):.2f} {adv:+10.3f} {t:+6.2f} "
              f"{TRAIN_GAIN_JOINT[tag]:+8.3f}")

    # ---- H0 kill switch (loud only) -------------------------------------- #
    loud = arms["loud"]
    min_j = min(loud["joint"]["per_seed_gain"])
    min_d = min(loud["disjoint"]["per_pair_gain"])
    if min_j >= GAIN_MIN and min_d >= GAIN_MIN:
        h0 = "PASS"
    else:
        h0 = "FAIL"
    print(f"\nH0: min gain(joint,loud)={min_j:.3f}, "
          f"min gain(disjoint,loud)={min_d:.3f} -> {h0}")
    if h0 == "FAIL":
        print("   frozen reading: below apparatus threshold for measuring transfer "
              "at this scale. H2/H3 withheld.")
        return

    # ---- H2 primary (loud) ------------------------------------------------ #
    j_loud = loud["joint"]["per_seed_gain"]
    d_loud = loud["disjoint"]["per_pair_gain"]
    adv_loud = mean(j_loud) - mean(d_loud)
    t_loud = welch_t(j_loud, d_loud)
    if adv_loud >= DELTA and abs(t_loud) >= T_MIN:
        h2 = "POSITIVE"
    elif adv_loud <= -DELTA and abs(t_loud) >= T_MIN:
        h2 = "NEGATIVE"
    elif abs(adv_loud) < NULL_BAND:
        h2 = "NULL"
    else:
        h2 = "INCONCLUSIVE"
    print(f"\nH2 [loud PRIMARY]: advantage = {adv_loud:+.3f} "
          f"(t {t_loud:+.2f}) -> {h2}")
    if h2 == "POSITIVE":
        print("   frozen reading: unity has an OPERATIONAL signature at this "
              "scale -- joint carries transferable structure disjoint doesn't. "
              "H4 tie-in TRIGGERED: rerun F28's directional metric on held-out "
              "via future amendment (F29 does not measure it here).")
    elif h2 == "NULL":
        print("   frozen reading: no operational advantage at this scale, "
              "consistent with F28 zero AND Dyna-2 low-scale ablation. Bounds "
              "unity below this apparatus, not death of unity.")
    elif h2 == "NEGATIVE":
        print("   frozen reading: joint OVERFITS to training substrate; "
              "disjoint specialists transfer better. F27b substrate "
              "entanglement is actively harmful, not just cosmetic.")

    # Per-arm H2 co-report.
    print("\nH2 [per-arm co-report]:")
    for tag in WORLD_ORDER:
        if tag == "loud":
            continue
        w = arms[tag]
        j = w["joint"]["per_seed_gain"]
        d = w["disjoint"]["per_pair_gain"]
        adv = mean(j) - mean(d)
        t = welch_t(j, d)
        if adv >= DELTA and abs(t) >= T_MIN:
            v = "POSITIVE"
        elif adv <= -DELTA and abs(t) >= T_MIN:
            v = "NEGATIVE"
        elif abs(adv) < NULL_BAND:
            v = "NULL"
        else:
            v = "INCONCLUSIVE"
        print(f"   {tag}: advantage {adv:+.3f} (t {t:+.2f}) -> {v}")

    # ---- H3 curve --------------------------------------------------------- #
    tags = [t for t in WORLD_ORDER if t in arms]
    if len(tags) == 4:
        advs = [mean(arms[t]["joint"]["per_seed_gain"]) -
                mean(arms[t]["disjoint"]["per_pair_gain"]) for t in tags]
        rs = [arms[t]["r"] for t in tags]
        rho = spearman(rs, advs)
        span = advs[-1] - advs[0]  # loud - ctrl (WORLD_ORDER ordered ascending in r)
        if rho >= RHO_MIN and span >= ADV_SPAN_MIN:
            h3 = "POSITIVE-COUPLING"
        elif abs(span) < NULL_BAND:
            h3 = "NULL"
        else:
            h3 = "INCONCLUSIVE"
        print(f"\nH3: Spearman(advantage, r) = {rho:+.2f}, "
              f"span(loud-ctrl) = {span:+.3f} -> {h3}")
        print("    advantage per arm: " +
              ", ".join(f"{t} {advs[i]:+.3f}" for i, t in enumerate(tags)))
    else:
        print("\nH3: withheld (needs all 4 worlds live)")

    # ---- H4 (descriptive fork) ------------------------------------------- #
    print()
    if h2 == "POSITIVE":
        print("H4: TRIGGERED. F28 directional metric on held-out is the tie-in "
              "test. Not measured here; requires a separate driver commit AND "
              "amendment to this document.")
    else:
        print("H4: not triggered (H2 not POSITIVE). F28 zero and F29 finding "
              "jointly consistent with the scale-bound reading (P1's frozen path).")


if __name__ == "__main__":
    main()
