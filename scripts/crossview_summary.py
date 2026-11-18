"""Tables for the F36 write-up, generated from the cross-view pooling reports.

Reads only what scripts/crossview_measure.py wrote, <root>/<arm>_<world>.json,
and reuses scripts/crossview_verdicts.py for loading, the device and
view_pool checks, the gates and every comparison, so no number here is
computed a second way. Added after the verdicts (2026-11-18); it grades
nothing and changes no frozen rule. It prints Markdown:

- gates per arm, with each arm's lowest final PR per world and its highest
  ctrl nested friction (G2 is per run, so the maximum is the check);
- the G1 failures by run, the counts per world, and Fisher's exact test of
  P0's ctrl failures against the mixer and control arms' (descriptive);
- per-arm friction z R^2 (nested, fixed alpha, held out, nested under 1e-4
  input noise) as mean +- sd, and the median final PR;
- every comparison as d (t) on each readout and the held-out d, beside its
  verdict. A WITHHELD comparison carries no per-readout grade;
- D1: each construction's mean over P0 seeds against z0, with the other
  order (mean of per-seed maxima) beside it;
- restitution z R^2 on the fixed-alpha readout per arm (P4's anchor).

    python scripts/crossview_summary.py --root runs/xv > runs/xv/crossview_summary.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossview_verdicts as V  # noqa: E402

WORLDS = ("loud", "ctrl")
MIXERS = ("P1", "P1c", "P2", "P2c")
P4_FLOOR = 0.97


def load(root: str) -> dict:
    return {a: {w: json.load(open(os.path.join(root, f"{a}_{w}.json"))) for w in WORLDS}
            for a in V.ARMS}


def ms(xs: list[float]) -> str:
    return f"{st.mean(xs):+.3f} ± {st.stdev(xs):.3f}"


def g1_failures(reps: dict) -> list[dict]:
    """Every run whose final PR fails G1 (below the floor or not finite)."""
    out = []
    for a in V.ARMS:
        for w in WORLDS:
            for run, r in reps[a][w]["runs"].items():
                p = r["latent_pr"]
                if p is None or not math.isfinite(p) or p < V.MIN_PR:
                    out.append({"arm": a, "world": w, "run": os.path.basename(run), "pr": p})
    return out


def g1_counts(reps: dict) -> dict:
    """Failures over runs per group, and Fisher's exact test of P0 ctrl
    against the mixer and control arms' ctrl runs (descriptive only)."""
    from scipy import stats
    fails = g1_failures(reps)

    def count(arms, w):
        n = sum(len(reps[a][w]["runs"]) for a in arms)
        k = sum(1 for f in fails if f["arm"] in arms and f["world"] == w)
        return k, n
    p0c, mxc = count(("P0",), "ctrl"), count(MIXERS, "ctrl")
    table = [[p0c[0], p0c[1] - p0c[0]], [mxc[0], mxc[1] - mxc[0]]]
    return {"P0_loud": count(("P0",), "loud"), "P0_ctrl": p0c,
            "mixers_loud": count(MIXERS, "loud"), "mixers_ctrl": mxc,
            "fisher_two_sided": float(stats.fisher_exact(table)[1]),
            "fisher_one_sided": float(stats.fisher_exact(table, alternative="less")[1])}


def arm_rows(reps: dict) -> list[dict]:
    rows = []
    for a in V.ARMS:
        for w in WORLDS:
            rep = reps[a][w]
            row = {"arm": a, "world": w, "n": len(rep["runs"]),
                   "nested": V.col(rep, "friction_nested"), "fixed": V.col(rep, "fixed"),
                   "heldout": V.col(rep, "heldout"),
                   "pr_median": st.median(V.col(rep, "latent_pr"))}
            runs = list(rep["runs"].values())
            if all("noise" in r for r in runs):
                row["noise"] = [r["noise"]["friction_nested"] for r in runs]
            rows.append(row)
    return rows


def d1_orders(reps: dict) -> dict:
    """D1 as graded (best construction by mean over seeds) and the other
    order (the mean over seeds of each seed's best construction)."""
    runs = list(reps["P0"]["loud"]["runs"].values())
    return {"max_of_means": V.d1(reps)["D1"],
            "mean_of_maxima": st.mean(max(r["d1"].values()) for r in runs)}


def restitution(reps: dict) -> dict:
    out = {}
    for a in V.ARMS:
        xs = [r["z_r2_fixed"]["restitution"] for r in reps[a]["loud"]["runs"].values()]
        out[a] = {"mean": st.mean(xs), "min": min(xs), "n": len(xs),
                  "below": sum(x < P4_FLOOR for x in xs)}
    return out


def cell(s: str) -> str:
    """A Markdown table cell: a literal | would split the cell."""
    return s.replace("|", "\\|")


def withheld_by(c: dict, g: dict) -> str:
    """Which gate withheld the comparison, per failing arm."""
    parts = []
    for gate, key in (("G1", "G1_health"), ("G2", "G2_control")):
        arms = [a for a in c["withheld_by"] if not g[key][a]]
        if arms:
            parts.append(f"{gate}: {', '.join(arms)}")
    return "; ".join(parts)


def comparison_line(name: str, c: dict, g: dict) -> str:
    n, f = c["nested"], c["fixed"]
    cells = [name, " vs ".join(c["arms"]), f"{n['delta']:+.3f} ({n['t']:+.2f})",
             f"{f['delta']:+.3f} ({f['t']:+.2f})", f"{c['heldout_delta']:+.3f}"]
    if c["verdict"] == "WITHHELD":
        cells.append(f"WITHHELD ({withheld_by(c, g)})")
    else:
        cells.append(f"{c['verdict']} (nested {n['grade']}, fixed {f['grade']})")
    return "| " + " | ".join(cells) + " |"


def render(reps: dict) -> str:
    v = V.verdicts(reps)
    g = v["gates"]
    L = ["## Gates", "",
         "| arm | G1 | lowest PR, loud | lowest PR, ctrl | G2 | highest ctrl nested friction |",
         "|---|---|---|---|---|---|"]
    for a in V.ARMS:
        prs = {w: V.col(reps[a][w], "latent_pr") for w in WORLDS}
        L.append(f"| {a} | {'PASS' if g['G1_health'][a] else 'FAIL'} | {min(prs['loud']):.1f} "
                 f"| {min(prs['ctrl']):.1f} | {'PASS' if g['G2_control'][a] else 'FAIL'} "
                 f"| {max(V.col(reps[a]['ctrl'], 'friction_nested')):+.3f} |")
    g3 = g["G3_apparatus"]
    L += ["", f"G3: P0 loud nested {g3['p0_mean']:.3f} vs {g3['baseline']} "
          f"({g3['p0_mean'] - g3['baseline']:+.3f}), {'PASS' if g3['pass'] else 'FAIL'}.",
          "", "## G1 failures (final PR < 8)", "", "| run | final PR |", "|---|---|"]
    for f in g1_failures(reps):
        L.append(f"| {f['run']} | {f['pr']:.3f} |")
    c = g1_counts(reps)
    L += ["", f"Mixer and control arms: {c['mixers_ctrl'][0]} of {c['mixers_ctrl'][1]} ctrl runs, "
          f"{c['mixers_loud'][0]} of {c['mixers_loud'][1]} loud. P0: {c['P0_ctrl'][0]} of "
          f"{c['P0_ctrl'][1]} ctrl, {c['P0_loud'][0]} of {c['P0_loud'][1]} loud. "
          f"P0 ctrl vs mixer and control ctrl: Fisher exact two-sided "
          f"p {c['fisher_two_sided']:.3f} (one-sided {c['fisher_one_sided']:.3f}).",
          "", "## Friction z R² per arm (mean ± sd)", "",
          "| arm | world | n | nested | fixed α | held out | nested, 1e-4 noise | median PR |",
          "|---|---|---|---|---|---|---|---|"]
    for r in arm_rows(reps):
        noise = ms(r["noise"]) if "noise" in r else "n/a"
        L.append(f"| {r['arm']} | {r['world']} | {r['n']} | {ms(r['nested'])} | {ms(r['fixed'])} "
                 f"| {ms(r['heldout'])} | {noise} | {r['pr_median']:.1f} |")
    L += ["", "## Comparisons (loud world; Δ and Welch t)", "",
          "| comparison | arms | nested Δ (t) | fixed Δ (t) | held-out Δ | verdict |",
          "|---|---|---|---|---|---|"]
    for part in ("primary", "secondary"):
        for name, cmp in v[part].items():
            L.append(comparison_line(name, cmp, g))
    d, o = v["D1"], d1_orders(reps)
    L += ["", "## D1 (P0 loud, per-view constructions, nested)", "",
          "| construction | mean over seeds |", "|---|---|"]
    for k, x in d["constructions"].items():
        L.append(f"| {cell(k)} | {x:.3f} |")
    L += ["", f"Best: {d['best']} {d['D1']:.3f} vs z0 {d['z0']:.3f} ({d['D1'] - d['z0']:+.3f}): "
          f"{d['reading']}. Other order (mean of per-seed maxima): {o['mean_of_maxima']:.3f}.",
          "", "## Restitution z R² (loud, fixed α)", "",
          f"| arm | mean | lowest | runs below {P4_FLOOR} |", "|---|---|---|---|"]
    for a, x in restitution(reps).items():
        L.append(f"| {a} | {x['mean']:.3f} | {x['min']:.3f} | {x['below']} of {x['n']} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/xv")
    args = ap.parse_args()
    sys.stdout.write(render(load(args.root)))


if __name__ == "__main__":
    main()
