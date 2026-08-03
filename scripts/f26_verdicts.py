#!/usr/bin/env python
"""F26 frozen decision rules (docs/COHERENCE_VS_R.md sections 3-4).

Reads runs/f26/f26_report.json and emits the preregistered verdict on the
coherence-vs-r curve, the ctrl-lowest sub-check, and the descriptive k=1
series. Committed in the SAME commit as the preregistration, before any new
measurement ran.

Usage:  .venv/bin/python scripts/f26_verdicts.py [--report runs/f26/f26_report.json]
"""
from __future__ import annotations

import argparse
import json
import math

# Frozen constants (do not edit after the freeze commit).
RISE_MIN, RISE_T, RHO_MIN = 0.05, 2.5, 0.6
FLAT_RANGE = 0.05
VALIDITY_BAND = (0.08, 0.22)  # base arm must bracket F13's +0.146
PR_FLOOR = 8.0
CONTROL_MAX = 0.05


def welch_t_summary(a, b):
    if a["n"] < 2 or b["n"] < 2:
        return float("nan")
    se = math.sqrt(a["std"] ** 2 / a["n"] + b["std"] ** 2 / b["n"])
    return (a["mean"] - b["mean"]) / se if se > 0 else float("nan")


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    d = math.sqrt(vx * vy)
    return cov / d if d > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="runs/f26/f26_report.json")
    args = ap.parse_args()
    with open(args.report) as f:
        rep = json.load(f)
    cont = rep["cont"]

    # ---- gates (section 4) ---------------------------------------------- #
    gates_ok, notes = True, []
    base = cont.get("base")
    if base is None or not (VALIDITY_BAND[0] <= base["learned"] <= VALIDITY_BAND[1]):
        notes.append(f"VALIDITY: base learned "
                     f"{'missing' if base is None else base['learned']} "
                     f"outside {VALIDITY_BAND}")
        gates_ok = False
    for tag, a in cont.items():
        if a["latent_pr_trained"] < PR_FLOOR:
            notes.append(f"COLLAPSE on {tag}: PR {a['latent_pr_trained']:.1f}")
            gates_ok = False
        indep = a.get("independent_baseline")
        if indep is None:
            notes.append(f"CONTROL UNMET on {tag}")
            gates_ok = False
        elif abs(indep) >= CONTROL_MAX:
            notes.append(f"CONTROL failed on {tag}: |indep| {abs(indep):.3f}")
            gates_ok = False
    for n in notes:
        print(f"  !! {n}")
    print(f"GATES: {'ALL PASS' if gates_ok else 'FAILED -- verdict withheld'}")

    order = sorted(cont.values(), key=lambda a: a["r"])
    print(f"\n{'r':>6s} {'n':>3s} {'learned':>9s} {'sd':>7s} {'PR':>6s} {'essence':>8s}")
    for a in order:
        print(f"{a['r']:6.3f} {a['n']:3d} {a['learned']:+9.4f} "
              f"{a['trained_shared']['std']:7.4f} {a['latent_pr_trained']:6.1f} "
              f"{a['essence_learned']:+8.4f}")

    # ---- H1 (section 3) -------------------------------------------------- #
    lo = min(cont.values(), key=lambda a: a["r"])
    hi = max(cont.values(), key=lambda a: a["r"])
    diff = hi["learned"] - lo["learned"]
    t = welch_t_summary(
        {"mean": hi["learned"], "std": hi["trained_shared"]["std"], "n": hi["n"]},
        {"mean": lo["learned"], "std": lo["trained_shared"]["std"], "n": lo["n"]})
    rho = spearman([a["r"] for a in order], [a["learned"] for a in order])
    means = [a["learned"] for a in order]
    spread = max(means) - min(means)
    if diff >= RISE_MIN and t >= RISE_T and rho >= RHO_MIN:
        verdict = "RISES"
    elif spread < FLAT_RANGE:
        verdict = "FLAT"
    else:
        verdict = "MIXED"
    print(f"\nH1: learned(r={hi['r']:.3f}) - learned(r={lo['r']:.3f}) = {diff:+.4f} "
          f"(Welch t {t:.2f}); Spearman rho {rho:+.2f}; spread {spread:.4f}")
    print(f"H1 -> {verdict}")
    if verdict == "FLAT":
        print("   consequence (frozen): the pre-fill reading is RETRACTED; "
              "F25's discussion paragraph gets a dated correction.")
    elif verdict == "MIXED":
        print("   consequence (frozen): pre-fill unsupported unless ctrl is "
              "lowest; summary is 'no interpretable r-dependence'.")

    # ---- ctrl-lowest sub-check ------------------------------------------- #
    ranked = sorted(cont.items(), key=lambda kv: kv[1]["learned"])
    pos = [tag for tag, _ in ranked].index("ctrl") + 1
    print(f"\nsub-check: ctrl ranks {pos} of {len(ranked)} lowest "
          f"(P1 predicts 1) -> {'PASS' if pos == 1 else 'bottom-two' if pos == 2 else 'FAIL'}")
    if pos > 2:
        print("   frozen consequence: the pre-fill story is dead regardless "
              "of the overall verdict.")

    # ---- H2 (descriptive) ------------------------------------------------ #
    fsq = rep.get("fsq_k1", {})
    if fsq:
        srt = sorted(fsq.values(), key=lambda a: a["r"])
        print("\nH2 (descriptive) k=1 learned coherence vs r:")
        for a in srt:
            print(f"  r={a['r']:.3f}  learned {a['learned']:+.4f}  (n={a['n']})")
        vals = [a["learned"] for a in srt]
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        mono = ("monotone down" if all(d < 0 for d in diffs)
                else "monotone up" if all(d > 0 for d in diffs) else "mixed")
        print(f"  series -> {mono} (P3 predicts monotone down)")


if __name__ == "__main__":
    main()
