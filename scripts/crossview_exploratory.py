"""F36 exploratory addendum: post hoc, NOT registered.

Nothing here can rescue or reverse a WITHHELD verdict; a claim about mixing
needs a new preregistration (F36's option c). The script prints no grade of
any kind. Its parts are E1-E3, so they are not confused with F36's
registered D1 (the per-view constructions). It reads only what the F36 runs
already wrote: the crossview_measure reports (<root>/<arm>_<world>.json),
each run's metrics.json history, and the seed-100 stability runs of
docs/CROSSVIEW_POOLING.md section 3. It trains nothing and computes no new
readout. Its design was agreed with the cloud session (cloud-notes a82eb91)
and committed, with its predictions, before it was run on the F36 runs.

- E1, per-run exclusion: drop every run that failed G1 (final PR < 8) and
  recompute each loud comparison. Its as-registered and exclusion deltas sit
  side by side, each with Welch t and a 95 % CI. A comparison whose arms lost
  no loud run is identical by construction, and is listed as such, not
  recomputed.
- E2, the ctrl trajectories: for every ctrl run, and as references the
  seed-100 stability runs, the PR at a few epochs, the first epoch below 8,
  the lowest PR and its epoch, whether it recovered, and the final loss.
- E3, loud runs, per arm (n 20): Spearman rho between final PR and nested
  friction z R^2, with a seeded percentile bootstrap 95 % CI. This is
  descriptive only. PR and the readout come from the same z, so rho != 0
  does not mean that PR causes a bad readout.

    python scripts/crossview_exploratory.py --root runs/xv \\
        --stability runs/xv_cam/stability > runs/xv/crossview_exploratory.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossview_summary as S  # noqa: E402
import crossview_verdicts as V  # noqa: E402

LABEL = ("**EXPLORATORY (post hoc; not registered).** Nothing here can rescue or "
         "reverse a WITHHELD verdict; a claim about mixing needs a new "
         "preregistration (F36's option c).")
TRAJ_EPOCHS = (0, 5, 10, 20, 30, 40, 49)
STABILITY = {"P1": "stats_s100", "P1c": "stats_ctrl_s100", "P2": "attn_s100",
             "P2c": "attn_ctrl_s100"}
BOOT, BOOT_SEED = 10_000, 0


def t975(df: float) -> float:
    from scipy import stats
    return float(stats.t.ppf(0.975, df))


def ci(w: dict) -> tuple[float, float]:
    h = t975(w["df"]) * w["se"]
    return w["delta"] - h, w["delta"] + h


def drop_runs(rep: dict, runs: set[str]) -> dict:
    return {**rep, "runs": {k: v for k, v in rep["runs"].items()
                            if os.path.basename(k) not in runs}}


def e1(reps: dict) -> list[dict]:
    """Each loud comparison as registered and with the G1-failing runs dropped."""
    failing = {f["run"] for f in S.g1_failures(reps) if f["world"] == "loud"}
    kept = {a: drop_runs(reps[a]["loud"], failing) for a in V.ARMS}
    out = []
    for name, a, b in V.PRIMARY + V.SECONDARY:
        dropped = sorted(r for r in failing
                         if r.startswith((f"{a}_", f"{b}_")))
        row = {"name": name, "arms": (a, b), "dropped": dropped}
        if dropped:
            for label, src in (("registered", {x: reps[x]["loud"] for x in (a, b)}),
                               ("exclusion", {x: kept[x] for x in (a, b)})):
                row[label] = {}
                for ro in ("friction_nested", "fixed"):
                    w = V.welch(V.col(src[a], ro), V.col(src[b], ro))
                    row[label][ro] = {**w, "ci": ci(w),
                                      "n": (len(src[a]["runs"]), len(src[b]["runs"]))}
                row[label]["heldout"] = (st.mean(V.col(src[a], "heldout"))
                                         - st.mean(V.col(src[b], "heldout")))
        out.append(row)
    return out


def trajectory(history: list[dict]) -> dict:
    pr = [h["latent_pr"] for h in history]
    below = [i for i, p in enumerate(pr) if p < V.MIN_PR]
    lo = min(range(len(pr)), key=pr.__getitem__)
    return {"at": {e: pr[e] for e in TRAJ_EPOCHS if e < len(pr)},
            "first_below": below[0] if below else None,
            "min": pr[lo], "min_epoch": lo, "final": pr[-1],
            "recovered": bool(below) and pr[-1] >= V.MIN_PR,
            "final_loss": history[-1]["train_loss"]}


def e2(reps: dict, stability: str | None) -> list[dict]:
    rows = []
    for a in V.ARMS:
        for run in reps[a]["ctrl"]["runs"]:
            h = json.load(open(os.path.join(run, "metrics.json")))["history"]
            rows.append({"run": os.path.basename(run), "arm": a, "kind": "ctrl",
                         **trajectory(h)})
    if stability:
        for a, sub in STABILITY.items():
            p = os.path.join(stability, sub, "metrics.json")
            if os.path.exists(p):
                rows.append({"run": f"{sub} (stability, §3)", "arm": a, "kind": "stability",
                             **trajectory(json.load(open(p))["history"])})
    return rows


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=xs.__getitem__)
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


def boot_ci(x: list[float], y: list[float], b: int = BOOT, seed: int = BOOT_SEED):
    import numpy as np
    rng = np.random.default_rng(seed)
    n, vals = len(x), []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        r = spearman([x[i] for i in idx], [y[i] for i in idx])
        if math.isfinite(r):
            vals.append(r)
    if not vals:  # rho is undefined on every resample (a constant input)
        return math.nan, math.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def fmt_rho(x: float) -> str:
    return f"{x:+.2f}" if math.isfinite(x) else "n/a"



def e3(reps: dict) -> list[dict]:
    out = []
    for a in V.ARMS:
        rep = reps[a]["loud"]
        pr, fr = V.col(rep, "latent_pr"), V.col(rep, "friction_nested")
        out.append({"arm": a, "n": len(pr), "rho": spearman(pr, fr), "ci": boot_ci(pr, fr)})
    return out


def render(reps: dict, stability: str | None) -> str:
    V.check_arms(reps)
    L = ["# F36 exploratory addendum", "", LABEL, "",
         "## E1: comparisons with the G1-failing runs dropped (loud world)", "",
         "| comparison | runs dropped | readout | as registered: Δ, t, 95 % CI (n) "
         "| exclusion: Δ, t, 95 % CI (n) |", "|---|---|---|---|---|"]
    unchanged = []
    for r in e1(reps):
        if not r["dropped"]:
            unchanged.append(r["name"])
            continue
        for ro, lab in (("friction_nested", "nested"), ("fixed", "fixed α")):
            cells = []
            for k in ("registered", "exclusion"):
                w = r[k][ro]
                cells.append(f"{w['delta']:+.3f}, t {w['t']:+.2f}, "
                             f"[{w['ci'][0]:+.3f}, {w['ci'][1]:+.3f}] ({w['n'][0]}/{w['n'][1]})")
            L.append(f"| {r['name']} | {', '.join(r['dropped'])} | {lab} | {cells[0]} "
                     f"| {cells[1]} |")
        L.append(f"| {r['name']} | {', '.join(r['dropped'])} | held-out Δ "
                 f"| {r['registered']['heldout']:+.3f} | {r['exclusion']['heldout']:+.3f} |")
    L += ["", "Identical by construction (no G1-failing loud run in either arm): "
          + "; ".join(unchanged) + ".",
          "", "## E2: PR trajectories of the ctrl runs, with the §3 stability runs", "",
          "| run | " + " | ".join(f"PR e{e}" for e in TRAJ_EPOCHS)
          + " | first epoch < 8 | lowest (epoch) | recovered | final loss |",
          "|---|" + "---|" * (len(TRAJ_EPOCHS) + 4)]
    for r in e2(reps, stability):
        at = " | ".join(f"{r['at'][e]:.1f}" if e in r["at"] else "n/a" for e in TRAJ_EPOCHS)
        fb = "never" if r["first_below"] is None else str(r["first_below"])
        rec = "n/a" if r["first_below"] is None else ("yes" if r["recovered"] else "no")
        L.append(f"| {r['run']} | {at} | {fb} | {r['min']:.1f} ({r['min_epoch']}) | {rec} "
                 f"| {r['final_loss']:.4f} |")
    L += ["", "## E3: final PR against nested friction z R², loud runs, per arm", "",
          "Descriptive only. PR and the readout come from the same z, so a nonzero "
          "ρ does not mean that PR causes a bad readout.", "",
          f"| arm | n | Spearman ρ | bootstrap 95 % CI ({BOOT} resamples, seed {BOOT_SEED}) |",
          "|---|---|---|---|"]
    for r in e3(reps):
        L.append(f"| {r['arm']} | {r['n']} | {fmt_rho(r['rho'])} | [{fmt_rho(r['ci'][0])}, "
                 f"{fmt_rho(r['ci'][1])}] |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/xv")
    ap.add_argument("--stability", default="runs/xv_cam/stability")
    args = ap.parse_args()
    sys.stdout.write(render(S.load(args.root), args.stability))


if __name__ == "__main__":
    main()
