"""Cross-view pooling verdicts (docs/CROSSVIEW_POOLING.md §5).

Committed with the preregistration, before any pooling-variant model is
trained on a loud world. Reads the reports scripts/crossview_measure.py
writes, one per arm and world, as <root>/<arm>_<world>.json with arm in
P0 P1 P1c P2 P2c (c = matched control) and world in loud, ctrl.

Rules, in order:
- Gates. G1: every run's final PR >= 8. G2: every ctrl run of an arm reads
  friction (nested) < 0.10. A G1 or G2 failure withholds every comparison
  that includes the failing arm. G3 (reported, withholds nothing): P0's loud
  mean is within 0.10 of the above-floor baseline 0.398.
- Each comparison is graded on the primary (nested) readout and on the fixed
  alpha = 1e-3 readout, by the same bars: RISES (d >= 0.10, t >= 2.5 and the
  held-out difference > 0), NULL (|d| < 0.05), FALLS (d <= -0.10, t <= -2.5),
  otherwise INCONCLUSIVE.
- If either readout grades RISES or FALLS and the other does not grade the
  same, the verdict is READOUT-DEPENDENT; otherwise the primary's grade stands.
- A NULL is reported with its bound |d| + t95 * SE (Welch, primary readout)
  and each arm's sd. If the bound reaches 0.10 it reads "NULL, bound above bar".
- D1: the best of the five per-view constructions (mean over P0 seeds) against
  z0, P0's mean nested friction: high (>= z0 + 0.15), about z (<= z0 + 0.05),
  otherwise unresolved.

    python scripts/crossview_verdicts.py --root runs/xv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st

ARMS = ("P0", "P1", "P1c", "P2", "P2c")
VIEW_POOL = {"P0": "mean", "P1": "stats", "P1c": "stats_ctrl", "P2": "attn",
             "P2c": "attn_ctrl"}
MIN_PR = 8.0
CTRL_FRICTION_MAX = 0.10
RISE, NULL, TBAR = 0.10, 0.05, 2.5
BASELINE_P0, G3_TOL = 0.398, 0.10
D1_HIGH, D1_LOW = 0.15, 0.05
PRIMARY = (("H1 stats mixing", "P1", "P1c"), ("H2 attention mixing", "P2", "P2c"))
SECONDARY = (("P1 vs P0", "P1", "P0"), ("P1c vs P0", "P1c", "P0"),
             ("P2 vs P0", "P2", "P0"), ("P2c vs P0", "P2c", "P0"),
             ("H3 mechanism", "P2", "P1"))


def t95(df: float) -> float:
    """One-sided 95 % Student t quantile."""
    from scipy import stats
    return float(stats.t.ppf(0.95, df))


def welch(a: list[float], b: list[float]) -> dict:
    """mean(a) - mean(b), Welch t, SE and Satterthwaite df."""
    va, vb = st.variance(a), st.variance(b)
    sa, sb = va / len(a), vb / len(b)
    se = math.sqrt(sa + sb)
    d = st.mean(a) - st.mean(b)
    t = d / se if se > 0 else (math.inf if d > 0 else -math.inf if d < 0 else 0.0)
    df = ((sa + sb) ** 2 / (sa ** 2 / (len(a) - 1) + sb ** 2 / (len(b) - 1))
          if sa + sb > 0 else float(len(a) + len(b) - 2))
    return {"delta": d, "t": t, "se": se, "df": df,
            "sd_a": math.sqrt(va), "sd_b": math.sqrt(vb)}


def classify(delta: float, t: float, delta_h: float) -> str:
    if delta >= RISE and t >= TBAR and delta_h > 0:
        return "RISES"
    if delta <= -RISE and t <= -TBAR:
        return "FALLS"
    if abs(delta) < NULL:
        return "NULL"
    return "INCONCLUSIVE"


def combine(primary: str, fixed: str) -> str:
    """§5's readout-dependent rule."""
    if (primary in ("RISES", "FALLS") or fixed in ("RISES", "FALLS")) and primary != fixed:
        return "READOUT-DEPENDENT"
    return primary


def col(rep: dict, key: str) -> list[float]:
    """One readout over a report's runs; key 'fixed', 'heldout' or a row field."""
    rows = list(rep["runs"].values())
    if key == "fixed":
        return [r["z_r2_fixed"]["friction"] for r in rows]
    if key == "heldout":
        return [r["z_r2_heldout"]["friction"] for r in rows]
    return [r[key] for r in rows]


def gates(reps: dict) -> dict:
    g1 = {a: all(p is not None and math.isfinite(p) and p >= MIN_PR
                 for w in ("loud", "ctrl") for p in col(reps[a][w], "latent_pr"))
          for a in ARMS}
    g2 = {a: all(f < CTRL_FRICTION_MAX for f in col(reps[a]["ctrl"], "friction_nested"))
          for a in ARMS}
    p0 = st.mean(col(reps["P0"]["loud"], "friction_nested"))
    return {"G1_health": g1, "G2_control": g2,
            "G3_apparatus": {"p0_mean": p0, "baseline": BASELINE_P0,
                             "pass": abs(p0 - BASELINE_P0) <= G3_TOL}}


def compare(reps: dict, a: str, b: str, g: dict) -> dict:
    la, lb = reps[a]["loud"], reps[b]["loud"]
    n = welch(col(la, "friction_nested"), col(lb, "friction_nested"))
    f = welch(col(la, "fixed"), col(lb, "fixed"))
    dh = st.mean(col(la, "heldout")) - st.mean(col(lb, "heldout"))
    gn, gf = classify(n["delta"], n["t"], dh), classify(f["delta"], f["t"], dh)
    out = {"arms": [a, b], "nested": {**n, "grade": gn}, "fixed": {**f, "grade": gf},
           "heldout_delta": dh,
           "mean_std_rank": [st.mean(col(la, "std_rank")), st.mean(col(lb, "std_rank"))],
           "verdict": combine(gn, gf)}
    if out["verdict"] == "NULL":
        bound = abs(n["delta"]) + t95(n["df"]) * n["se"]
        out["null_bound"] = bound
        if bound >= RISE:
            out["verdict"] = "NULL, bound above bar"
    failed = [x for x in (a, b) if not (g["G1_health"][x] and g["G2_control"][x])]
    if failed:
        out["withheld_by"] = failed
        out["verdict"] = "WITHHELD"
    return out


def d1(reps: dict) -> dict:
    rows = list(reps["P0"]["loud"]["runs"].values())
    means = {k: st.mean(r["d1"][k] for r in rows) for k in rows[0]["d1"]}
    best = max(means, key=means.get)
    z0 = st.mean(col(reps["P0"]["loud"], "friction_nested"))
    v = means[best]
    reading = ("high" if v >= z0 + D1_HIGH else "about z" if v <= z0 + D1_LOW
               else "unresolved")
    return {"constructions": means, "best": best, "D1": v, "z0": z0, "reading": reading}


def descriptive(reps: dict) -> dict:
    out = {}
    for a in ARMS:
        rows = list(reps[a]["loud"]["runs"].values())
        out[a] = {k: [r["z_r2_fixed"][k] for r in rows] for k in ("density", "restitution")}
        out[a]["heldout_gain"] = [r["heldout_gain"] for r in rows]
        out[a]["latent_pr"] = [r["latent_pr"] for r in rows]
    return out


def check_arms(reps: dict) -> None:
    for a in ARMS:
        for w, rep in reps[a].items():
            pools = {r.get("view_pool") for r in rep["runs"].values()}
            if pools != {VIEW_POOL[a]}:
                raise ValueError(f"{a}_{w}: view_pool {pools}, expected {VIEW_POOL[a]}")


def verdicts(reps: dict) -> dict:
    check_arms(reps)
    g = gates(reps)
    return {"gates": g,
            "primary": {name: compare(reps, a, b, g) for name, a, b in PRIMARY},
            "secondary": {name: compare(reps, a, b, g) for name, a, b in SECONDARY},
            "D1": d1(reps), "descriptive": descriptive(reps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/xv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    reps = {a: {w: json.load(open(os.path.join(args.root, f"{a}_{w}.json")))
                for w in ("loud", "ctrl")} for a in ARMS}
    v = verdicts(reps)
    g = v["gates"]
    for a in ARMS:
        print(f"[xv] {a:3s} G1 {'PASS' if g['G1_health'][a] else 'FAIL'}"
              f"  G2 {'PASS' if g['G2_control'][a] else 'FAIL'}")
    g3 = g["G3_apparatus"]
    print(f"[xv] G3 P0 loud {g3['p0_mean']:.3f} vs {g3['baseline']} "
          f"{'PASS' if g3['pass'] else 'FAIL (reported)'}")
    for part in ("primary", "secondary"):
        for name, c in v[part].items():
            n, f = c["nested"], c["fixed"]
            print(f"[xv] {name:20s} nested d {n['delta']:+.3f} t {n['t']:+.2f} {n['grade']:12s}"
                  f" fixed d {f['delta']:+.3f} t {f['t']:+.2f} {f['grade']:12s}"
                  f" held-out d {c['heldout_delta']:+.3f} -> {c['verdict']}"
                  + (f" (bound {c['null_bound']:.3f}; sd {n['sd_a']:.3f}/{n['sd_b']:.3f})"
                     if "null_bound" in c else ""))
    d = v["D1"]
    print(f"[xv] D1 {d['D1']:.3f} ({d['best']}) vs z0 {d['z0']:.3f} -> {d['reading']}")
    json.dump(v, open(args.out or os.path.join(args.root, "crossview_verdicts.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()
