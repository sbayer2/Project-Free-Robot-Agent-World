"""Check that two MuJoCo worlds differ only in their cameras.

Used before comparing a world re-rendered with another camera layout against the
original: every sample.json field must match scene for scene except the camera
positions (appearance.frames[i].position) and stored trajectories (present only
when a world was built with --keep-trajectory). Also reports the new world's
camera heights and the roughness<->friction correlation of both worlds.

    python scripts/compare_worlds.py data/pm_f31fix_loud data/pm_f31fix_loud_cam
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

SKIP = re.compile(r"^appearance\.frames\[\d+\]\.position$|\.trajectory\[")


def flatten(d: object, prefix: str = "") -> dict[str, object]:
    """Flatten nested dicts and non-numeric lists into dotted keys."""
    out: dict[str, object] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(d, list) and d and not all(isinstance(x, (int, float)) for x in d):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


def scene_ids(world: str) -> list[str]:
    return sorted(d for d in os.listdir(world) if os.path.isdir(os.path.join(world, d)))


def load(world: str, scene: str) -> dict:
    with open(os.path.join(world, scene, "sample.json")) as f:
        return json.load(f)


def rough_friction_r(world: str) -> float:
    rough, fric = [], []
    for s in scene_ids(world):
        j = load(world, s)
        rough.append(j["material_truth"]["appearance_params"]["roughness"])
        fric.append(j["physics"]["raw"]["friction"])
    return float(np.corrcoef(rough, fric)[0, 1])


def compare(old: str, new: str) -> dict:
    ids = scene_ids(old)
    if ids != scene_ids(new):
        raise SystemExit(f"scene ids differ between {old} and {new}")
    differing: dict[str, int] = {}
    n_values = 0
    cam_z = []
    for s in ids:
        a, b = flatten(load(old, s)), flatten(load(new, s))
        for k in set(a) | set(b):
            if SKIP.search(k):
                continue
            n_values += 1
            if a.get(k) != b.get(k):
                differing[k] = differing.get(k, 0) + 1
        cam_z += [fr["position"][2] for fr in load(new, s)["appearance"]["frames"]]
    return {
        "scenes": len(ids),
        "values_compared": n_values,
        "differing": differing,
        "new_camera_z": [float(min(cam_z)), float(max(cam_z))],
        "rough_friction_r": {"old": rough_friction_r(old), "new": rough_friction_r(new)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("old")
    ap.add_argument("new")
    args = ap.parse_args()
    rep = compare(args.old, args.new)
    print(json.dumps(rep, indent=2))
    sys.exit(1 if rep["differing"] else 0)


if __name__ == "__main__":
    main()
