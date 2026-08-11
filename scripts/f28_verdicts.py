#!/usr/bin/env python
"""F28 frozen decision rules (docs/DIRECTIONAL_COHERENCE.md sections 6-7).

Reads runs/f28/f28_report.json and emits the preregistered verdicts.
Committed in the SAME commit as the preregistration, before any directional-
coherence number existed anywhere.

    H1   CLEAN / ENTANGLED / BIASED / INCONCLUSIVE      (learned_D at ctrl)
    H2a  POWERED / UNDERPOWERED                          (loud anchor, gates H2's NULL)
    H2   CONTENT / NULL / INCONCLUSIVE                   (PRIMARY: g2 vs ctrl)
    H3   PASS / FAIL / INCONCLUSIVE at ctrl; coupled arms co-reported
    H4   descriptive Spearman(learned_D, r), no verdict

Usage:  .venv/bin/python scripts/f28_verdicts.py [--report runs/f28/f28_report.json]
"""
from __future__ import annotations

import argparse
import json
import math

# Frozen constants (do not edit after the freeze commit).
H1_CLEAN, H1_TRIP = 0.05, 0.10          # |D| bands at ctrl
H2_MIN, H2_T, H2_NULL = 0.10, 2.5, 0.05  # shared for H2 and the H2a anchor
H3_CLEAN, H3_TRIP = 0.05, 0.10          # disjoint ctrl bounds
H3_GAP = 0.10                            # shared - disjoint "beyond content" bar
R_CTRL_MAX = 0.10
PR_FLOOR = 8.0
PROBE_R2_TRAINED, PROBE_R2_UNTRAINED = 0.25, 0.10
JAC_NORM_FLOOR = 1e-4
STITCH_R2_MIN = 0.5
VERDICT_ARMS = ("ctrl", "g2", "loud")

# F27b magnitude-coherence profile, reprinted for the H4 dissociation read.
MAGNITUDE_LEARNED = {"ctrl": 0.434, "base": 0.423, "g2": 0.420, "loud": 0.205}


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
    ap.add_argument("--report", default="runs/f28/f28_report.json")
    args = ap.parse_args()
    with open(args.report) as f:
        rep = json.load(f)
    arms, app = rep["arms"], rep["apparatus"]

    # ---- gates (section 7) ---------------------------------------------- #
    gates_ok, notes = True, []
    voided = set()  # arms whose verdicts are withheld
    if abs(app["r"]["ctrl"]) >= R_CTRL_MAX:
        notes.append(f"APPARATUS: r(ctrl) {app['r']['ctrl']:+.3f} not ~0")
        gates_ok = False
    rs = [app["r"][t] for t in ("base", "g2", "loud")]
    if not all(rs[i] < rs[i + 1] for i in range(len(rs) - 1)):
        notes.append(f"APPARATUS: r not monotone in g: {rs}")
        gates_ok = False
    for tag, a in arms.items():
        g = a["gates"]
        if g["latent_pr_min"] < PR_FLOOR:
            notes.append(f"COLLAPSE: {tag} min PR {g['latent_pr_min']:.1f}")
            voided.add(tag)
        if g["probe_r2_min_trained"] < PROBE_R2_TRAINED:
            notes.append(f"PROBES: {tag} trained-probe min R2 "
                         f"{g['probe_r2_min_trained']:.3f} < {PROBE_R2_TRAINED}")
            voided.add(tag)
        if g["probe_r2_min_untrained"] < PROBE_R2_UNTRAINED:
            notes.append(f"PROBES: {tag} untrained-probe min R2 "
                         f"{g['probe_r2_min_untrained']:.3f} < {PROBE_R2_UNTRAINED}")
            voided.add(tag)
        if g["jac_norm_median_min"] < JAC_NORM_FLOOR:
            notes.append(f"JACOBIANS: {tag} median norm {g['jac_norm_median_min']:.2e}")
            voided.add(tag)
    disjoint_void = set()
    for tag, a in arms.items():
        dj = a.get("disjoint")
        if dj is None or dj["stitch_r2"] < STITCH_R2_MIN:
            r2 = "missing" if dj is None else f"{dj['stitch_r2']:.3f}"
            notes.append(f"STITCH: {tag} R2 {r2} < {STITCH_R2_MIN} -> disjoint cell VOID")
            disjoint_void.add(tag)
    for n in notes:
        print(f"  !! {n}")
    if voided & set(VERDICT_ARMS):
        gates_ok = False
    status = "ALL PASS" if gates_ok and not voided else "FAILED/PENDING -- see notes"
    if gates_ok and not voided and disjoint_void:
        status += f" (disjoint cells VOID: {', '.join(sorted(disjoint_void))})"
    print(f"GATES: {status}")

    print(f"\n{'arm':5s} {'r':>7s} {'learned_D':>12s} {'behav_D':>9s} "
          f"{'mism|abs|':>10s} {'disjoint':>9s} {'stitchR2':>9s} {'magnitude':>10s}")
    for tag in ("ctrl", "base", "g2", "loud"):
        a = arms.get(tag)
        if a is None:
            continue
        ld = a["learned_D_per_seed"]
        bd = a.get("behavior_learned_D_per_seed", [])
        dj = a.get("disjoint") or {}
        djs = "VOID" if tag in disjoint_void else f"{dj.get('D', float('nan')):+9.4f}"
        print(f"{tag:5s} {a['r']:+7.3f} {mean(ld):+8.4f}±{sd(ld):.3f} "
              f"{mean(bd) if bd else float('nan'):+9.4f} "
              f"{a['mismatched_abs_mean']:10.4f} {djs:>9s} "
              f"{dj.get('stitch_r2', float('nan')):9.3f} "
              f"{MAGNITUDE_LEARNED.get(tag, float('nan')):+10.3f}")

    # ---- H1 --------------------------------------------------------------- #
    d_ctrl = mean(arms["ctrl"]["learned_D_per_seed"])
    if abs(d_ctrl) < H1_CLEAN:
        v1 = "CLEAN"
    elif d_ctrl >= H1_TRIP:
        v1 = "ENTANGLED"
    elif d_ctrl <= -H1_TRIP:
        v1 = "BIASED"
    else:
        v1 = "INCONCLUSIVE"
    print(f"\nH1: learned_D(ctrl) = {d_ctrl:+.4f} -> {v1}")
    if v1 == "ENTANGLED":
        print("   frozen reading: directionality does NOT escape substrate "
              "entanglement; H2/H2a withheld for content-reading.")
        return
    if v1 == "BIASED":
        print("   frozen reading: instrument bias; run VOID.")
        return
    if v1 == "INCONCLUSIVE":
        print("   H2/H2a read against an unclean null -- reported, graded "
              "INCONCLUSIVE-tainted.")

    # ---- H2a anchor ------------------------------------------------------- #
    ld_loud = arms["loud"]["learned_D_per_seed"]
    ld_ctrl = arms["ctrl"]["learned_D_per_seed"]
    ad = mean(ld_loud) - mean(ld_ctrl)
    at = welch_t(ld_loud, ld_ctrl)
    h2a = "POWERED" if (ad >= H2_MIN and at >= H2_T) else "UNDERPOWERED"
    print(f"\nH2a: learned_D(loud) - learned_D(ctrl) = {ad:+.4f} (t {at:.2f}) -> {h2a}")

    # ---- H2 (PRIMARY: g2 vs ctrl) ----------------------------------------- #
    ld_g2 = arms["g2"]["learned_D_per_seed"]
    diff = mean(ld_g2) - mean(ld_ctrl)
    t = welch_t(ld_g2, ld_ctrl)
    if diff >= H2_MIN and t >= H2_T:
        v2 = "CONTENT"
    elif abs(diff) < H2_NULL:
        v2 = "NULL (unreadable -- H2a UNDERPOWERED)" if h2a != "POWERED" else "NULL"
    else:
        v2 = "INCONCLUSIVE"
    print(f"H2:  learned_D(g2) - learned_D(ctrl) = {diff:+.4f} (t {t:.2f}) -> {v2}")
    if v2 == "CONTENT":
        print("   frozen reading: a coherence-family metric finally discriminates "
              "content at AUTHORED strength (50%-blended map); the F22-F27 lever "
              "arcs can be re-asked against an instrument that can see the answer.")
    elif v2 == "NULL":
        print("   frozen reading: matched content at authored strength is invisible "
              "to directional reads even where near-definitional content is visible "
              "(H2a POWERED); the authored-coupling regime is below this "
              "instrument's floor.")

    # ---- H3 --------------------------------------------------------------- #
    print()
    if "ctrl" in disjoint_void:
        print("H3: ctrl disjoint cell VOID (stitch gate) -> verdict withheld")
    else:
        dj_ctrl = arms["ctrl"]["disjoint"]["D"]
        if abs(dj_ctrl) < H3_CLEAN:
            v3 = "PASS"
        elif abs(dj_ctrl) >= H3_TRIP:
            v3 = "FAIL"
        else:
            v3 = "INCONCLUSIVE"
        print(f"H3: D_disjoint(ctrl) = {dj_ctrl:+.4f} -> {v3}")
        if v3 == "FAIL":
            print("   frozen reading: the statistic can be manufactured without "
                  "joint training OR content; interpretation of all arms withheld "
                  "pending diagnosis.")
    for tag in ("base", "g2", "loud"):
        if tag in disjoint_void or tag not in arms:
            continue
        gap = mean(arms[tag]["learned_D_per_seed"]) - arms[tag]["disjoint"]["D"]
        note = ("content-without-substrate; no unity evidence" if abs(gap) < H2_NULL
                else "joint training adds alignment BEYOND content + linear stitch"
                if gap >= H3_GAP else "co-reported, between frozen notes")
        print(f"    {tag}: shared - disjoint = {gap:+.4f}  ({note})")

    # ---- H4 (descriptive) ------------------------------------------------- #
    tags = [t for t in ("ctrl", "base", "g2", "loud") if t in arms]
    ds = [mean(arms[t]["learned_D_per_seed"]) for t in tags]
    rho = spearman([arms[t]["r"] for t in tags], ds)
    print(f"\nH4 (descriptive): Spearman(learned_D, r) = {rho:+.2f} over {len(tags)} arms; "
          "magnitude profile for contrast: "
          + ", ".join(f"{t} {MAGNITUDE_LEARNED[t]:+.3f}" for t in tags))


if __name__ == "__main__":
    main()
