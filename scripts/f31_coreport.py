"""F31a Amendment 1 co-reports (docs/TRAJECTORY_SUPERVISION.md §10).

Committed with the amendment, before any fixed-world model existed. The
frozen verdicts are computed by scripts/f31_verdicts.py unchanged; this
adds the C1 co-report (joint-arm friction z R^2, fixed vs broken world) and
the escaped-only secondary reading, with non-escapers listed.

    python scripts/f31_coreport.py --root runs/f31fix --broken runs/f31
    python scripts/f31_coreport.py --root runs/f31fix --broken runs/f31 --stage-a
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(__file__))
import f31_verdicts as V  # noqa: E402

ESCAPE_GAIN = 2.0     # §4's escape threshold, applied to the final epoch


def friction(rep: dict) -> list[float]:
    return V.column(rep, "", "friction")


def c1(fixed_joint: dict, broken_joint: dict) -> dict:
    fx, bk = friction(fixed_joint), friction(broken_joint)
    d, t = st.mean(fx) - st.mean(bk), V.welch_t(fx, bk)
    return {"fixed": fx, "broken": bk, "delta": d, "t": t,
            "verdict": V.classify(d, t, V.H1["rise"], V.H1["null"], V.H1["t"])}


def split_escape(rep: dict) -> tuple[dict, list[str]]:
    """(report restricted to escaped runs, names of non-escapers)."""
    kept, dropped = {}, []
    for name, row in rep["runs"].items():
        if (row.get("heldout_gain") or 0.0) >= ESCAPE_GAIN:
            kept[name] = row
        else:
            dropped.append(name)
    return {**rep, "runs": kept}, dropped


def escaped_only(reps: dict[str, dict]) -> dict:
    out: dict = {"non_escapers": {}}
    kept = {}
    for a, rep in reps.items():
        kept[a], out["non_escapers"][a] = split_escape(rep)
    for h, key, axis, bars in (("H1_friction", "", "friction", V.H1),
                               ("H2_gain", "heldout_gain", None, V.H2)):
        j = V.column(kept["loud_joint"], key, axis)
        tr = V.column(kept["loud_traj"], key, axis)
        if len(j) < 2 or len(tr) < 2:
            out[h] = {"verdict": "UNREADABLE (fewer than 2 escaped seeds in an arm)"}
            continue
        d, t = st.mean(tr) - st.mean(j), V.welch_t(tr, j)
        out[h] = {"delta": d, "t": t, "n": [len(j), len(tr)],
                  "verdict": V.classify(d, t, bars["rise"], bars["null"], bars["t"])}
    return out


def leakage_adjusted(reps: dict[str, dict]) -> dict:
    """Amendment 2 (a): D_L = (loud_traj - loud_joint) - (ctrl_traj - ctrl_joint)
    on friction z R^2, SE = sqrt(sum of var/n over the four arms)."""
    import math
    f = {a: friction(reps[a]) for a in V.ARMS}
    m = {a: st.mean(v) for a, v in f.items()}
    d = (m["loud_traj"] - m["loud_joint"]) - (m["ctrl_traj"] - m["ctrl_joint"])
    se = math.sqrt(sum(st.variance(v) / len(v) for v in f.values()))
    return {"delta_L": d, "se": se, "t": d / se if se > 0 else float("inf"),
            "loud_diff": m["loud_traj"] - m["loud_joint"],
            "ctrl_diff": m["ctrl_traj"] - m["ctrl_joint"]}


def heldout_readout(reps: dict[str, dict]) -> dict:
    """Amendment 2 (b): per-run friction R^2 on the held-out corner scenes."""
    return {a: {name: row.get("z_r2_heldout", {}).get("friction")
                for name, row in reps[a]["runs"].items()} for a in V.ARMS}


def load(root: str, arm: str) -> dict:
    return json.load(open(os.path.join(root, f"{arm}.json")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/f31fix")
    ap.add_argument("--broken", default="runs/f31")
    ap.add_argument("--stage-a", action="store_true", help="joint arm only (C1)")
    args = ap.parse_args()

    report: dict = {"C1": c1(load(args.root, "loud_joint"), load(args.broken, "loud_joint"))}
    r = report["C1"]
    print(f"[f31-C1] joint friction z R2 fixed {[round(x, 3) for x in r['fixed']]} "
          f"(mean {st.mean(r['fixed']):.3f}) vs broken mean {st.mean(r['broken']):.3f}: "
          f"delta {r['delta']:+.3f} t {r['t']:+.2f} -> {r['verdict']}")
    _, ne = split_escape(load(args.root, "loud_joint"))
    print(f"[f31-C1] loud-joint non-escapers: {ne or 'none'}")
    if not args.stage_a:
        reps = {a: load(args.root, a) for a in V.ARMS}
        report["frozen"] = V.verdicts(reps)
        report["escaped_only"] = escaped_only(reps)
        report["leakage_adjusted"] = leakage_adjusted(reps)
        report["heldout_friction"] = heldout_readout(reps)
        la = report["leakage_adjusted"]
        print(f"[f31-A2a] D_L {la['delta_L']:+.4f} (loud {la['loud_diff']:+.4f} - ctrl "
              f"{la['ctrl_diff']:+.4f}), SE {la['se']:.4f}, t {la['t']:+.2f}")
        for a, runs in report["heldout_friction"].items():
            print(f"[f31-A2b] held-out friction R2 {a}: "
                  + " ".join(f"{v:+.3f}" if v is not None else "n/a" for v in runs.values()))
        for h in ("H1_friction", "H2_gain"):
            f, e = report["frozen"][h], report["escaped_only"][h]
            print(f"[f31] {h:12s} frozen (all seeds): {f['delta']:+.4f} t {f['t']:+.2f} "
                  f"-> {f['verdict']}  |  escaped only: {e.get('verdict')}")
        print(f"[f31] gates {report['frozen']['gates']}; non-escapers "
              f"{report['escaped_only']['non_escapers']}")
    name = "coreport_stage_a.json" if args.stage_a else "coreport.json"
    json.dump(report, open(os.path.join(args.root, name), "w"), indent=2)


if __name__ == "__main__":
    main()
