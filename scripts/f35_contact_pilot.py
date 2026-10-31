"""F35 D7 pilot: does a contact-model change remove the restitution leak into
the friction targets, and what does it cost the drop-onto-ramp probe?

The leak (D6): under the repo contact model (pyramidal cone, friction a soft
constraint sharing the box's restitution-derived solref) a bouncy box creeps
down a slope it should hold, so ramp and kick outcomes read restitution.
Each variant regenerates the battery (f35_generate_battery.py --contact ...),
then runs the code table and the outcome fields on it, and adds:

  creep      downhill travel over 2 s on the 38.7 deg ramp of boxes that
             should hold (raw friction > tan 38.7 + 0.1), median / max, and
             its Pearson r with raw restitution;
  dropramp   normal-velocity restitution: dropramp.e1 (rebound ratio in the
             plane-normal coordinate) against drop.e1 (flat ground, same
             material), as Pearson r and median ratio, plus each field's R^2
             from the essence;
  ablation   restitution read from the ramp and kick outcome families with
             the field definitions varied: kick without stop_s (it repeats
             log_stop's stop event in time), ramp slides floored at a dead
             zone (log10 of max(d, D)), to separate a field-definition leak
             from the physics.

    python scripts/f35_contact_pilot.py --out data/pm_f35_d7pilot
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from f35_code_table import inverse_r2  # noqa: E402
from f35_outcomes import EPS_M  # noqa: E402

VARIANTS = {                     # name: (contact, impratio, solreffriction)
    "repo": ("repo", 1.0, ""),
    "ell_i1": ("elliptic", 1.0, ""),
    "ell_i10": ("elliptic", 10.0, ""),
    "ell_i100": ("elliptic", 100.0, ""),
    "ell_sf": ("elliptic", 1.0, "0.01 1"),
}
HOLD_DEG, HOLD_MARGIN = 38.7, 0.1
WATCH = ("ramp38.7", "kick1.0", "kick3.0", "dropramp", "drop", "tap")


def run(*cmd: str) -> None:
    subprocess.run([sys.executable, *cmd], check=True, stdout=subprocess.DEVNULL)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def creep(data: str) -> dict:
    z = np.load(os.path.join(data, "battery.npz"))
    raw = z["raw"].astype(float)
    r = z[f"traj_ramp{HOLD_DEG:04.1f}"].astype(float)
    a = math.radians(HOLD_DEG)
    downhill = -np.array([0.0, math.cos(a), math.sin(a)])
    travel = (r[:, -1, 1:4] - r[:, 0, 1:4]) @ downhill
    held = raw[:, 1] > math.tan(a) + HOLD_MARGIN
    t = travel[held]
    return {"n_held": int(held.sum()), "median_mm": float(np.median(t) * 1e3),
            "max_mm": float(t.max() * 1e3), "r_restitution": pearson(t, raw[held, 2])}


def dropramp_restitution(data: str) -> dict:
    o = np.load(os.path.join(data, "outcomes.npz"))
    names = list(o["names"])
    v = o["values"]
    dr, d = v[:, names.index("dropramp.e1")], v[:, names.index("drop.e1")]
    ok = np.isfinite(dr) & np.isfinite(d) & (d > 0)
    return {"r_vs_drop_e1": pearson(dr, d), "median_ratio": float(np.median(dr[ok] / d[ok])),
            "undefined": int((~np.isfinite(dr)).sum())}


def leak_ablation(data: str) -> dict[str, list[float]]:
    """Essence R^2 (density, friction, restitution) of the ramp / kick
    families under varied field definitions."""
    o = np.load(os.path.join(data, "outcomes.npz"))
    names, fams, v = list(o["names"]), list(o["families"]), o["values"]
    E = np.load(os.path.join(data, "battery.npz"))["essence"].astype(float)

    def r2(cols: list[np.ndarray]) -> list[float]:
        X = np.stack(cols, 1)
        ok = np.isfinite(X).all(1)
        return inverse_r2(X[ok], E[ok]).tolist()

    def col(n: str) -> np.ndarray:
        return v[:, names.index(n)]

    ramp = [n for n, f in zip(names, fams, strict=True) if f == "ramp"]
    slides = [n for n in ramp if n.endswith("log_slide")]
    kick = [n for n, f in zip(names, fams, strict=True) if f == "kick"]
    out = {"ramp": r2([col(n) for n in ramp]),
           "ramp_slides_only": r2([col(n) for n in slides])}
    for dz in (0.01, 0.02, 0.05):
        floored = [np.log10(np.maximum(10 ** col(n) - EPS_M, dz)) for n in slides]
        out[f"ramp_deadzone_{dz * 100:g}cm"] = r2(floored)
    out["kick"] = r2([col(n) for n in kick])
    out["kick_no_stop_s"] = r2([col(n) for n in kick if not n.endswith("stop_s")])
    return out


def variant(out: str, name: str, spec: tuple, n: int, seed: int, reuse: bool,
            dropramp_deg: float) -> dict:
    data = os.path.join(out, name)
    contact, imp, srf = spec
    if not (reuse and os.path.exists(os.path.join(data, "outcomes.npz"))):
        _generate(data, contact, imp, srf, n, seed, dropramp_deg)
    ct = json.load(open(os.path.join(data, "code_table.json")))
    oc = json.load(open(os.path.join(data, "outcomes.json")))
    return {"spec": {"contact": contact, "impratio": imp, "solreffriction": srf},
            "creep": creep(data), "dropramp_restitution": dropramp_restitution(data),
            "ablation": leak_ablation(data),
            "gate": {a: g["pass"] for a, g in ct["gate"]["axes"].items()},
            "per_recording": {k: ct["per_recording"][f"dropramp{dropramp_deg:04.1f}"
                                                     if k == "dropramp" else k]
                              for k in WATCH},
            "outcome_family": oc["per_family"],
            "outcome_field": {k: oc["per_field"][k] for k in
                              ("drop.e1", "dropramp.e1", "dropramp.log_slide",
                               "ramp38.7.log_slide", "kick1.0.log_stop", "kick3.0.log_stop")},
            "diagnostics": oc["diagnostics"]}


def _generate(data: str, contact: str, imp: float, srf: str, n: int, seed: int,
              dropramp_deg: float) -> None:
    run(os.path.join(HERE, "f35_generate_battery.py"), "--sample", str(n), "--seed", str(seed),
        "--out", data, "--contact", contact, "--impratio", str(imp), "--solreffriction", srf,
        "--dropramp-deg", str(dropramp_deg))
    run(os.path.join(HERE, "f35_code_table.py"), "--data", data,
        "--out", os.path.join(data, "code_table.json"))
    run(os.path.join(HERE, "f35_outcomes.py"), "--data", data)


def f3(r: list[float]) -> str:
    return " ".join(f"{x:+.2f}" for x in r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--only", nargs="*", default=None, help="subset of variants")
    ap.add_argument("--reuse", action="store_true", help="skip variants already generated")
    ap.add_argument("--dropramp-deg", type=float, default=38.7)
    args = ap.parse_args()
    names = args.only or list(VARIANTS)
    res = {}
    for name in names:
        res[name] = variant(args.out, name, VARIANTS[name], args.sample, args.seed, args.reuse,
                            args.dropramp_deg)
        print(f"[pilot] {name} done", flush=True)
    print("\nCREEP on the 38.7 deg ramp, boxes that should hold "
          f"(n = {res[names[0]]['creep']['n_held']})")
    for k, r in res.items():
        c = r["creep"]
        print(f"  {k:9s} median {c['median_mm']:7.1f} mm  max {c['max_mm']:7.1f} mm  "
              f"r(restitution) {c['r_restitution']:+.2f}")
    print("\nPER RECORDING (density friction restitution), 5 mm, CV alpha")
    for w in WATCH:
        print(f"  {w}")
        for k, r in res.items():
            print(f"    {k:9s} {f3(r['per_recording'][w])}")
    print("\nOUTCOME FAMILIES (essence <- fields)")
    for fam in ("ramp", "kick", "dropramp"):
        print(f"  {fam}")
        for k, r in res.items():
            print(f"    {k:9s} {f3(r['outcome_family'][fam])}")
    print("\nDROPRAMP normal-velocity restitution vs flat drop; field R^2 from essence")
    for k, r in res.items():
        d, f = r["dropramp_restitution"], r["outcome_field"]
        print(f"  {k:9s} r(dropramp.e1, drop.e1) {d['r_vs_drop_e1']:+.2f}  median ratio "
              f"{d['median_ratio']:.2f}  R2 dropramp.e1 {f['dropramp.e1']:+.2f}  "
              f"drop.e1 {f['drop.e1']:+.2f}  tumbled {r['diagnostics']['dropramp_tumbled']}")
    print("\nABLATION: essence R^2 (density friction restitution) from the ramp / kick "
          "outcome families, field definitions varied")
    for a in res[names[0]]["ablation"]:
        print(f"  {a}")
        for k, r in res.items():
            print(f"    {k:9s} {f3(r['ablation'][a])}")
    print("\nGATE")
    for k, r in res.items():
        print(f"  {k:9s} " + "  ".join(f"{a} {'PASS' if p else 'FAIL'}"
                                      for a, p in r["gate"].items()))
    json.dump(res, open(os.path.join(args.out, "pilot.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
