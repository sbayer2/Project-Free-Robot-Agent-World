"""F35 battery across material seeds: build, gate and read out one design
iteration on several sampled worlds, then summarize as mean +- sd with
sign/pass counts (docs/CROSS_INTERACTION.md §9). Each seed is a fresh
MaterialSampler draw of 512 materials; the 5 mm gate noise is drawn per
recording as in f35_code_table.py.

    python scripts/f35_seeds.py --design D7 --out data/pm_f35_d7_seeds
    python scripts/f35_seeds.py --design D6 --out data/pm_f35_d6_seeds
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from f35_contact_pilot import creep  # noqa: E402

AXES = ("density", "friction", "restitution")


def run(*cmd: str, log: str | None = None) -> None:
    with open(log or os.devnull, "w") as fh:
        subprocess.run([sys.executable, *cmd], check=True, stdout=fh)


def one(root: str, design: str, seed: int, n: int, reuse: bool) -> dict:
    data = os.path.join(root, f"s{seed}")
    if not (reuse and os.path.exists(os.path.join(data, "outcomes.json"))):
        run(os.path.join(HERE, "f35_generate_battery.py"), "--design", design,
            "--sample", str(n), "--seed", str(seed), "--out", data)
        run(os.path.join(HERE, "f35_code_table.py"), "--data", data,
            "--out", os.path.join(data, "code_table.json"),
            log=os.path.join(data, "code_table.txt"))
        run(os.path.join(HERE, "f35_outcomes.py"), "--data", data,
            log=os.path.join(data, "outcomes.txt"))
    ct = json.load(open(os.path.join(data, "code_table.json")))
    oc = json.load(open(os.path.join(data, "outcomes.json")))
    man = json.load(open(os.path.join(data, "manifest.json")))
    dropramp = f"dropramp{man['dropramp_deg']:04.1f}"
    return {"design_iteration": man["design_iteration"],
            "gate": {a: g["pass"] for a, g in ct["gate"]["axes"].items()},
            "pairs_shared": sum(bool(v) for v in ct["gate"]["pairs"].values()),
            "pairs_total": len(ct["gate"]["pairs"]),
            "family_5mm": ct["code_table"]["5mm"],
            "per_recording": {k: ct["per_recording"][k] for k in ("ramp38.7", "kick1.0",
                                                                  "kick3.0", dropramp)},
            "dropramp_recording": dropramp,
            "outcome_family": oc["per_family"],
            "diagnostics": oc["diagnostics"],
            "creep": creep(data)}


def ms(x) -> str:
    x = np.asarray(x, float)
    return f"{x.mean():+.2f} +- {x.std(ddof=1) if len(x) > 1 else 0.0:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="D7")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 1, 2, 3, 4])
    ap.add_argument("--sample", type=int, default=512)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = {}
    for s in args.seeds:
        res[s] = one(args.out, args.design, s, args.sample, args.reuse)
        print(f"[seeds] {args.design} seed {s} done", flush=True)
    R = list(res.values())
    print(f"\n{args.design}: {len(R)} seeds {args.seeds}, {args.sample} scenes each "
          f"(manifest design_iteration: {sorted({r['design_iteration'] for r in R})})")
    print("\nGATE passes per axis (seeds); shared pairs")
    for a in AXES:
        print(f"  {a:12s} {sum(r['gate'][a] for r in R)}/{len(R)}")
    print(f"  pairs shared {ms([r['pairs_shared'] for r in R])} of {R[0]['pairs_total']}")
    print("\nFAMILY TABLE, 5 mm, CV alpha (density / friction / restitution)")
    for fam in R[0]["family_5mm"]:
        print(f"  {fam:9s} " + "  ".join(ms([r['family_5mm'][fam][k] for r in R])
                                         for k in range(3)))
    print("\nPER RECORDING, 5 mm")
    keys = ["ramp38.7", "kick1.0", "kick3.0"]
    for k in keys + ["dropramp"]:
        rows = [r["per_recording"][r["dropramp_recording"] if k == "dropramp" else k] for r in R]
        print(f"  {k:9s} " + "  ".join(ms([x[i] for x in rows]) for i in range(3)))
    print("\nOUTCOME FAMILIES (essence <- fields)")
    for fam in R[0]["outcome_family"]:
        print(f"  {fam:9s} " + "  ".join(ms([r['outcome_family'][fam][k] for r in R])
                                         for k in range(3)))
    print("\nCOVARIATES: count, r with (density, friction, restitution)")
    for c in ("dropramp_tumbled", "kick3.0_tipped"):
        n = [r["diagnostics"][c] for r in R]
        rr = [r["diagnostics"].get(f"{c}_r_essence", [np.nan] * 3) for r in R]
        print(f"  {c:17s} n {ms(n)}  r " + "  ".join(ms([x[i] for x in rr]) for i in range(3)))
    print("\nCREEP of held boxes on the 38.7 deg ramp (mm; r with restitution, friction)")
    c = [r["creep"] for r in R]
    print(f"  median {ms([x['median_mm'] for x in c])}  max {ms([x['max_mm'] for x in c])}  "
          f"r(e) {ms([x['r_restitution'] for x in c])}  r(mu) {ms([x['r_friction'] for x in c])}")
    json.dump({str(k): v for k, v in res.items()}, open(os.path.join(args.out, "seeds.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()
