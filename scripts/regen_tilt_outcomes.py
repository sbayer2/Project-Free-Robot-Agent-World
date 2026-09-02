"""Regenerate the TILT probe outcomes of an existing dataset in place.

User-approved follow-up to the F34 instrument-flag fix (the tilt ramp ran
at ~0.349 deg instead of 20 for the project's entire history; commit
c6db6da fixed the call sites). This re-simulates ONLY the tilt probe for
every scene, with the stored spec and the fixed code, and rewrites each
sample.json with the corrected outcome. Drop/push outcomes, renders,
splits, specs, and every other field are byte-preserved.

Material is reconstructed from the stored record (physics from
physics.raw, visual from material_truth.appearance_params — visual does
not affect dynamics but build_mjcf requires it). PhysicsConfig defaults
match the original generation (sim 2.0 s, 60 fps, drop settle values in
the stored data confirm them).

An audit trail (old vs new tilt outcome per scene + summary deltas) is
written to <data>/tilt_regen_audit.json.

Usage:
    .venv/bin/python scripts/regen_tilt_outcomes.py --data data/pm_big
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pseudomarble import materials as M  # noqa: E402
from pseudomarble import probes as P  # noqa: E402
from pseudomarble.config import PhysicsConfig  # noqa: E402
from pseudomarble.data.generate_mujoco import run_tilt  # noqa: E402


def material_from_sample(sample: dict) -> M.Material:
    raw = sample["physics"]["raw"]
    ap = sample["material_truth"]["appearance_params"]
    return M.Material(
        name=str(sample["input"]["material"]),
        visual=M.VisualProps(
            base_color=tuple(ap["base_color"]), roughness=ap["roughness"],
            metallic=ap["metallic"], transmission=ap["transmission"],
            ior=ap["ior"]),
        physics=M.PhysicsProps(density=raw["density"], friction=raw["friction"],
                               restitution=raw["restitution"]),
        tags=(),
    )


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="simulate and report deltas without writing")
    args = ap.parse_args(argv)

    cfg = PhysicsConfig()
    scenes = sorted(d for d in os.listdir(args.data)
                    if os.path.isfile(os.path.join(args.data, d, "sample.json")))
    audit: dict = {"data": args.data, "n_scenes": len(scenes), "scenes": {}}
    n_toppled_flips = 0

    for i, sid in enumerate(scenes):
        path = os.path.join(args.data, sid, "sample.json")
        sample = json.load(open(path))
        probes = sample["behavior"]["probes"]
        idx = next(j for j, p in enumerate(probes) if p["probe"] == "tilt")
        old = probes[idx]["outcome"]
        spec = P.TiltSpec(angle_deg=probes[idx]["spec"]["angle_deg"])
        mat = material_from_sample(sample)
        traj = run_tilt(sample["input"]["shape"], mat, spec, cfg)
        new = P.summarize(traj).to_dict()
        probes[idx]["outcome"] = new
        if bool(old["toppled"]) != bool(new["toppled"]):
            n_toppled_flips += 1
        audit["scenes"][sid] = {"old": old, "new": new}
        if not args.dry_run:
            with open(path, "w") as fh:
                json.dump(sample, fh, indent=2)
        if (i + 1) % 50 == 0 or i == len(scenes) - 1:
            print(f"[regen-tilt] {i + 1}/{len(scenes)} scenes", flush=True)

    def mean(key: str, which: str) -> float:
        vals = [float(a[which][key]) for a in audit["scenes"].values()]
        return sum(vals) / len(vals) if vals else 0.0

    audit["summary"] = {
        "toppled_flips": n_toppled_flips,
        "mean_old": {k: mean(k, "old") for k in
                     ("settle_time", "slid_distance", "path_length",
                      "final_tilt_deg", "n_bounces")},
        "mean_new": {k: mean(k, "new") for k in
                     ("settle_time", "slid_distance", "path_length",
                      "final_tilt_deg", "n_bounces")},
        "dry_run": args.dry_run,
    }
    audit_path = os.path.join(args.data, "tilt_regen_audit.json")
    with open(audit_path, "w") as fh:
        json.dump(audit, fh, indent=1)
    print(f"[regen-tilt] toppled flips: {n_toppled_flips}/{len(scenes)}")
    print("[regen-tilt] means old->new:")
    for k in audit["summary"]["mean_old"]:
        print(f"  {k}: {audit['summary']['mean_old'][k]:.4f} -> "
              f"{audit['summary']['mean_new'][k]:.4f}")
    print(f"[regen-tilt] audit -> {audit_path}")
    print("REGEN-TILT DONE" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
