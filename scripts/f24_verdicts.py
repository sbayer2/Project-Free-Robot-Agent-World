#!/usr/bin/env python
"""F24 frozen decision rules (docs/CONVERGENCE.md sections 4-6, 8-9).

Reads runs/f24/f24_report.json (written by scripts/f24_measure.py) and emits
the preregistered verdicts. This file is committed in the SAME commit as the
preregistration, before any measurement ran -- editing it after the report
exists voids the freeze.

    H1  RISES / NULL / INCONCLUSIVE     learned alignment vs coupling strength
    H2  WORLD-SPECIFIC / NULL / INCONCLUSIVE   within-world vs cross-world
    H3  CONVERGENT / NOT CONVERGENT / INCONCLUSIVE  per coupling, + content

Usage:  .venv/bin/python scripts/f24_verdicts.py [--report runs/f24/f24_report.json]
"""
from __future__ import annotations

import argparse
import json
import math

# Frozen constants (do not edit after the freeze commit).
H1_RISE_MIN, H1_T, H1_NULL = 0.05, 2.5, 0.03
H2_MIN, H2_T, H2_NULL = 0.03, 2.5, 0.02
H3_EXCESS, H3_NOT = 0.15, 0.05
H3B_RATIO = 1.2
P1_T, P3_FLOOR = 2.5, 0.10
MIN_SEEDS = {"ctrl": 3}  # every other arm needs >= 6 usable seeds
DEFAULT_MIN_SEEDS = 6


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


def one_sample_t(d):
    if len(d) < 2:
        return float("nan")
    se = sd(d) / math.sqrt(len(d))
    return mean(d) / se if se > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="runs/f24/f24_report.json")
    args = ap.parse_args()
    with open(args.report) as f:
        rep = json.load(f)
    arms = rep["arms"]

    # ---- gates (section 8) --------------------------------------------- #
    gates_ok = True
    for label, g in rep["gates"].items():
        ok_self = g["self"] == 1.0
        ok_shuf = g["shuffle"] < 3 * g["chance"]
        n_used = arms[label]["n_trained_used"]
        need = MIN_SEEDS.get(label, DEFAULT_MIN_SEEDS)
        ok_n = n_used >= need
        print(f"gate {label:5s} self={'PASS' if ok_self else 'FAIL'} "
              f"shuffle={'PASS' if ok_shuf else 'FAIL'} ({g['shuffle']:.4f} "
              f"vs 3x chance {3 * g['chance']:.4f}) "
              f"seeds={'PASS' if ok_n else 'FAIL'} ({n_used}/{need}) "
              f"dropped={len(arms[label]['dropped_below_pr'])}")
        if label != "ctrl" and not (ok_self and ok_shuf and ok_n):
            gates_ok = False
    print(f"GATES: {'ALL PASS' if gates_ok else 'FAILED -- verdicts withheld'}")

    def learned_per_seed(label):
        u = arms[label]["untrained"]["knn_mean"]
        return [v - u for v in arms[label]["trained"]["per_unit_mean_knn"]]

    # ---- P1 / P3 per arm ------------------------------------------------ #
    for label in arms:
        t_units = arms[label]["trained"]["per_unit_mean_knn"]
        u_units = arms[label]["untrained"]["per_unit_mean_knn"]
        t = welch_t(t_units, u_units)
        learned = arms[label]["learned_knn"]
        u_mean = arms[label]["untrained"]["knn_mean"]
        print(f"arm {label:5s} r={arms[label]['r']:.3f} "
              f"trained={mean(t_units):.4f} untrained={u_mean:.4f} "
              f"learned={learned:+.4f} welch_t={t:.2f} "
              f"P1={'PASS' if (learned > 0 and t >= P1_T) else 'FAIL'} "
              f"P3(floor>={P3_FLOOR})={'PASS' if u_mean >= P3_FLOOR else 'FAIL'}")

    # ---- H1 (section 4) -------------------------------------------------- #
    a, b = learned_per_seed("base"), learned_per_seed("g15")
    primary = mean(b) - mean(a)
    t = welch_t(b, a)
    if primary >= H1_RISE_MIN and t >= H1_T:
        v = "RISES"
    elif abs(primary) < H1_NULL:
        v = "NULL"
    else:
        v = "INCONCLUSIVE"
    print(f"\nH1 PRIMARY = learned(g15) - learned(base) = {primary:+.4f}, "
          f"Welch t = {t:.2f} -> {v}")
    for label in ("ctrl", "loud"):
        if label in arms:
            print(f"   secondary learned({label}, r={arms[label]['r']:.3f}) "
                  f"= {mean(learned_per_seed(label)):+.4f}")

    # ---- H2 (section 5) -------------------------------------------------- #
    cx = rep["cross"].get("base|loud")
    if cx:
        u_gap = arms["loud"]["untrained"]["knn_mean"] - cx["untrained_knn_mean"]
        within = arms["loud"]["trained"]["per_unit_mean_knn"]
        cross_b = cx["per_seed_mean_knn_b"]
        d = [w - c - u_gap for w, c in zip(within, cross_b, strict=True)]
        t2 = one_sample_t(d)
        md = mean(d)
        if md >= H2_MIN and t2 >= H2_T:
            v2 = "WORLD-SPECIFIC"
        elif abs(md) < H2_NULL:
            v2 = "NULL"
        else:
            v2 = "INCONCLUSIVE"
        print(f"\nH2 loud-side paired d = {md:+.4f} (t = {t2:.2f}, n = {len(d)}) -> {v2}")
        u_gap_a = arms["base"]["untrained"]["knn_mean"] - cx["untrained_knn_mean"]
        da = [w - c - u_gap_a for w, c in zip(
            arms["base"]["trained"]["per_unit_mean_knn"], cx["per_seed_mean_knn_a"],
            strict=True)]
        print(f"   base-side mirror d = {mean(da):+.4f} (t = {one_sample_t(da):.2f}, "
              f"n = {len(da)})")
        for key, c in rep["cross"].items():
            print(f"   cross {key:10s} trained {c['knn_mean']:.4f} "
                  f"untrained {c['untrained_knn_mean']:.4f} "
                  f"learned {c['knn_mean'] - c['untrained_knn_mean']:+.4f}")

    # ---- H3 (section 6) -------------------------------------------------- #
    for label, s in rep.get("fsq", {}).items():
        ags = [p["agreement"] for p in s["pairs"]]
        excess = [p["agreement"] - p["null_mean"] for p in s["pairs"]]
        above = all(p["agreement"] > p["null_p95"] for p in s["pairs"])
        if above and mean(excess) >= H3_EXCESS:
            v3 = "CONVERGENT"
        elif mean(excess) < H3_NOT:
            v3 = "NOT CONVERGENT"
        else:
            v3 = "INCONCLUSIVE"
        mi_shape = mean(s["mi"]["shape"])
        mi_fac = {f: mean(s["mi"][f]) for f in s["mi"] if f != "shape"}
        best_f = max(mi_fac, key=mi_fac.get)
        if mi_shape > H3B_RATIO * mi_fac[best_f]:
            content = "shape-coded"
        elif mi_fac[best_f] > H3B_RATIO * mi_shape:
            content = f"material-coded ({best_f})"
        else:
            content = "mixed"
        print(f"\nH3 {label} (r={s['r']:.3f}): agreements "
              f"{[f'{x:.3f}' for x in ags]} mean excess {mean(excess):+.3f} "
              f"all>p95={above} -> {v3}")
        print(f"   content: MI shape {mi_shape:.3f} vs " +
              " ".join(f"{f} {v:.3f}" for f, v in mi_fac.items()) +
              f" -> {content}")


if __name__ == "__main__":
    main()
