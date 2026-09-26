"""F31a verdicts (docs/TRAJECTORY_SUPERVISION.md §5), frozen 2026-09-26.

Committed with the preregistration, before any trajectory-supervised model
existed. Reads the four arm reports written by scripts/f31_measure.py.

    python scripts/f31_verdicts.py --root runs/f31
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st

ARMS = ("loud_joint", "loud_traj", "ctrl_joint", "ctrl_traj")
MIN_PR = 8.0
CTRL_FRICTION_MAX = 0.10
H1 = {"rise": 0.10, "null": 0.05, "t": 2.5}
H2 = {"rise": 0.5, "null": 0.25, "t": 2.5}


def welch_t(a: list[float], b: list[float]) -> float:
    va = st.variance(a) if len(a) > 1 else 0.0
    vb = st.variance(b) if len(b) > 1 else 0.0
    se = math.sqrt(va / len(a) + vb / len(b))
    d = st.mean(a) - st.mean(b)
    return d / se if se > 0 else (math.inf if d > 0 else -math.inf if d < 0 else 0.0)


def classify(delta: float, t: float, rise: float, null: float, tbar: float) -> str:
    if delta >= rise and t >= tbar:
        return "RISES"
    if delta <= -rise and t <= -tbar:
        return "FALLS"
    if abs(delta) < null:
        return "NULL"
    return "INCONCLUSIVE"


def column(rep: dict, key: str, axis: str | None = None) -> list[float]:
    rows = rep["runs"].values()
    return [r["z_r2"][axis] if axis else r[key] for r in rows]


def verdicts(reps: dict[str, dict]) -> dict:
    out: dict = {"gates": {}}
    prs = {a: column(reps[a], "latent_pr") for a in ARMS}
    out["gates"]["G1_health"] = all(p >= MIN_PR for a in ARMS for p in prs[a])
    ctrl_f = column(reps["ctrl_joint"], "", "friction") + column(reps["ctrl_traj"], "", "friction")
    out["gates"]["G2_control"] = all(f < CTRL_FRICTION_MAX for f in ctrl_f)

    lj_f = column(reps["loud_joint"], "", "friction")
    lt_f = column(reps["loud_traj"], "", "friction")
    d1, t1 = st.mean(lt_f) - st.mean(lj_f), welch_t(lt_f, lj_f)
    h1 = classify(d1, t1, H1["rise"], H1["null"], H1["t"])
    if not (out["gates"]["G1_health"] and out["gates"]["G2_control"]):
        h1 = "WITHHELD"
    out["H1_friction"] = {"delta": d1, "t": t1, "verdict": h1,
                          "joint": lj_f, "traj": lt_f}

    lj_g = column(reps["loud_joint"], "heldout_gain")
    lt_g = column(reps["loud_traj"], "heldout_gain")
    d2, t2 = st.mean(lt_g) - st.mean(lj_g), welch_t(lt_g, lj_g)
    h2 = classify(d2, t2, H2["rise"], H2["null"], H2["t"])
    if not out["gates"]["G1_health"]:
        h2 = "WITHHELD"
    out["H2_gain"] = {"delta": d2, "t": t2, "verdict": h2, "joint": lj_g, "traj": lt_g}

    out["descriptive"] = {
        a: {"density": column(reps[a], "", "density"),
            "restitution": column(reps[a], "", "restitution"),
            "escape_epoch": column(reps[a], "escape_epoch"),
            "trajectory_gain": column(reps[a], "heldout_trajectory_gain")}
        for a in ARMS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/f31")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    reps = {a: json.load(open(os.path.join(args.root, f"{a}.json"))) for a in ARMS}
    v = verdicts(reps)
    for g, ok in v["gates"].items():
        print(f"[f31] gate {g:12s} {'PASS' if ok else 'FAIL'}")
    for h in ("H1_friction", "H2_gain"):
        r = v[h]
        print(f"[f31] {h:12s} delta {r['delta']:+.4f}  t {r['t']:+.2f}  -> {r['verdict']}")
    json.dump(v, open(args.out or os.path.join(args.root, "f31_verdicts.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()
