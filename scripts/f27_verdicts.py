#!/usr/bin/env python
"""F27 frozen decision rules (docs/SHAPE_DEGENERATE.md sections 5-6).

Reads runs/f27/f27_report.json and emits the preregistered verdicts.
Committed in the SAME commit as the preregistration, before any F27 dataset
existed.

    H1  SHAPE-CARRIED CONFIRMED / REFUTED / INCONCLUSIVE   (coh at r=0)
    H3  MATERIAL LEARNED / NOT LEARNED                      (gain gate for H2)
    H2  RISES / NULL / INCONCLUSIVE  (readable only if H3 = LEARNED)

Usage:  .venv/bin/python scripts/f27_verdicts.py [--report runs/f27/f27_report.json]
"""
from __future__ import annotations

import argparse
import json
import math

# Frozen constants (do not edit after the freeze commit).
H1_CONFIRM, H1_REFUTE = 0.05, 0.10
H3_MIN, H3_T = 0.20, 2.5
H2_RISE, H2_T, H2_NULL = 0.10, 2.5, 0.05
R_CTRL_MAX = 0.10
UNTRAINED_ANCHOR, UNTRAINED_TOL = 0.109, 0.02
PR_FLOOR = 8.0
CONTROL_MAX = 0.05


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0


def welch_t(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = math.sqrt(sd(a) ** 2 / len(a) + sd(b) ** 2 / len(b))
    return (mean(a) - mean(b)) / se if se > 0 else float("nan")


def welch_t_summary(a, b):
    if a["n"] < 2 or b["n"] < 2:
        return float("nan")
    se = math.sqrt(a["std"] ** 2 / a["n"] + b["std"] ** 2 / b["n"])
    return (a["mean"] - b["mean"]) / se if se > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="runs/f27/f27_report.json")
    args = ap.parse_args()
    with open(args.report) as f:
        rep = json.load(f)
    arms, app = rep["arms"], rep["apparatus"]

    # ---- gates (section 6) ---------------------------------------------- #
    gates_ok, notes = True, []
    for tag, shapes in app["all_box"].items():
        if shapes != ["box"]:
            notes.append(f"APPARATUS: {tag} shapes {shapes} != ['box']")
            gates_ok = False
    if not app["physics_matched"]:
        notes.append("APPARATUS: physics NOT matched across arms")
        gates_ok = False
    if abs(app["r"]["ctrl"]) >= R_CTRL_MAX:
        notes.append(f"APPARATUS: r(ctrl) {app['r']['ctrl']:+.3f} not ~0")
        gates_ok = False
    rs = [app["r"][t] for t in ("base", "g2", "loud")]
    if not all(rs[i] < rs[i + 1] for i in range(len(rs) - 1)):
        notes.append(f"APPARATUS: r not monotone in g: {rs}")
        gates_ok = False
    for tag, a in arms.items():
        c = a.get("coherence")
        if c is None:
            notes.append(f"{tag}: coherence not measured")
            gates_ok = False
            continue
        u = c["architectural"]["mean"]
        if abs(u - UNTRAINED_ANCHOR) > UNTRAINED_TOL:
            notes.append(f"VALIDITY: {tag} untrained coherence {u:+.4f} off anchor")
            gates_ok = False
        if c["latent_pr_trained"] < PR_FLOOR:
            notes.append(f"COLLAPSE: {tag} PR {c['latent_pr_trained']:.1f}")
            gates_ok = False
        indep = c.get("independent_baseline")
        if indep is None:
            notes.append(f"CONTROL UNMET on {tag}")
            gates_ok = False
        elif abs(indep) >= CONTROL_MAX:
            notes.append(f"CONTROL tripped on {tag}: |indep| {abs(indep):.3f} "
                         "(prospective rule: train 2 more pairs, gate on mean-of-3)")
            gates_ok = False
    for n in notes:
        print(f"  !! {n}")
    print(f"GATES: {'ALL PASS' if gates_ok else 'FAILED/PENDING -- see notes'}")

    print(f"\n{'arm':5s} {'r':>7s} {'gain':>7s} {'ceiling':>8s} {'coh learned':>12s} "
          f"{'essence':>8s} {'align':>8s}")
    for tag in ("ctrl", "base", "g2", "loud"):
        a = arms.get(tag)
        if a is None:
            continue
        c = a.get("coherence", {})
        print(f"{tag:5s} {a['r']:+7.3f} {a['gain_mean']:7.3f} "
              f"{a['appearance_oracle_gain']:8.3f} {c.get('learned', float('nan')):+12.4f} "
              f"{c.get('essence_learned', float('nan')):+8.4f} {a['align_learned']:+8.4f}")

    # ---- H1 --------------------------------------------------------------- #
    coh_ctrl = arms["ctrl"]["coherence"]["learned"]
    if coh_ctrl < H1_CONFIRM:
        v1 = "SHAPE-CARRIED CONFIRMED"
    elif coh_ctrl >= H1_REFUTE:
        v1 = "REFUTED"
    else:
        v1 = "INCONCLUSIVE"
    print(f"\nH1: learned coherence at r=0, single shape = {coh_ctrl:+.4f} "
          f"(multi-shape F26 value: +0.187) -> {v1}")

    # ---- H3 gate ---------------------------------------------------------- #
    g_loud, g_ctrl = arms["loud"]["gain_per_seed"], arms["ctrl"]["gain_per_seed"]
    gd = mean(g_loud) - mean(g_ctrl)
    gt = welch_t(g_loud, g_ctrl)
    h3 = "MATERIAL LEARNED" if (gd >= H3_MIN and gt >= H3_T) else "NOT LEARNED"
    print(f"\nH3: gain(loud) - gain(ctrl) = {gd:+.3f} (t {gt:.2f}) -> {h3}")

    # ---- H2 --------------------------------------------------------------- #
    if h3 != "MATERIAL LEARNED":
        print("H2: WITHHELD (H3 gate failed -- a coherence null would be "
              "uninformative)")
        return
    cl = arms["loud"]["coherence"]
    cc = arms["ctrl"]["coherence"]
    diff = cl["learned"] - cc["learned"]
    t = welch_t_summary(
        {"mean": cl["learned"], "std": cl["trained_shared"]["std"],
         "n": cl["trained_shared"]["n"]},
        {"mean": cc["learned"], "std": cc["trained_shared"]["std"],
         "n": cc["trained_shared"]["n"]})
    if diff >= H2_RISE and t >= H2_T:
        v2 = "RISES"
    elif abs(diff) < H2_NULL:
        v2 = "NULL"
    else:
        v2 = "INCONCLUSIVE"
    print(f"H2: learned_coh(loud) - learned_coh(ctrl) = {diff:+.4f} (t {t:.2f}) -> {v2}")
    if v2 == "RISES":
        print("   frozen reading: unity was SHADOWED by geometry, not impossible "
              "-- the Marble property needs a geometry-quiet world.")
    elif v2 == "NULL":
        print("   frozen reading: the strongest negative of the program -- even "
              "with material as the ONLY learnable shared structure, and learned "
              "(H3), the heads do not unify. Architecture-level, at this scale.")


if __name__ == "__main__":
    main()
