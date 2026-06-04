"""The ``sample.json`` contract — shared by every data generator.

Both the MuJoCo generator (primary) and the Blender generator (optional,
high-fidelity) write *identical* sample records through these helpers, so the
downstream loader, the coherence benchmark, and the held-out splits never have
to care which engine produced a scene. Swapping generators is swapping the code
*behind* this contract, not the contract itself.

Pure-Python and dependency-free, so it imports anywhere (CI, Blender's Python,
MLX on the Mac).

Schema (one scene -> ``<scene_id>/sample.json``)::

    {
      "scene_id":  "train_000001",
      "split":     "train",
      "generator": "mujoco",                 # provenance: which engine rendered it
      "input":     {"shape": "...", "material": "..."},   # the single shared input
      "appearance": {
        "resolution": 256,
        "frames": [{"index", "file", "position", "look_at"}]
      },
      "physics": {                           # authored from materials.py
        "raw":        {"density", "friction", "restitution"},
        "normalized": {"density", "friction", "restitution"},
        "tags": [...]
      },
      "dynamics": {"fps": 60, "trajectory": [{"frame", "position"}]}
    }

``input`` drives BOTH ``appearance`` and ``physics`` — that pairing is the point.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from pseudomarble import materials as M
from pseudomarble.config import PHYSICS_NORMALIZERS

SCHEMA_VERSION = 2  # v2 adds the `behavior` block (probe outcomes) + material_truth
Pose = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
# Camera placement (shared so every generator covers the object identically).
# --------------------------------------------------------------------------- #
def fibonacci_sphere_poses(num_views: int, radius: float) -> List[Pose]:
    """Evenly distributed camera positions on a sphere (deterministic)."""
    poses: List[Pose] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(num_views):
        y = 1.0 - (i / max(1, num_views - 1)) * 2.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        poses.append(
            (math.cos(theta) * r * radius, y * radius, math.sin(theta) * r * radius)
        )
    return poses


def pose_to_az_el_dist(pos: Pose) -> Tuple[float, float, float]:
    """Convert a Cartesian camera position to (azimuth°, elevation°, distance).

    MuJoCo's free camera is parameterized this way (looking at the origin), so
    generators that use a turntable camera can reuse ``fibonacci_sphere_poses``.
    """
    x, y, z = pos
    distance = math.sqrt(x * x + y * y + z * z)
    if distance == 0.0:
        return (0.0, 0.0, 0.0)
    azimuth = math.degrees(math.atan2(y, x))
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, z / distance))))
    return (azimuth, elevation, distance)


# --------------------------------------------------------------------------- #
# Physics labels (the "essence" target: density, friction, restitution).
# --------------------------------------------------------------------------- #
def _physics_labels(physics: "M.PhysicsProps", tags: Sequence[str]) -> Dict:
    d, f, r = physics.density, physics.friction, physics.restitution
    return {
        "raw": {"density": d, "friction": f, "restitution": r},
        "normalized": {
            "density": d / PHYSICS_NORMALIZERS["density"],
            "friction": f / PHYSICS_NORMALIZERS["friction"],
            "restitution": r / PHYSICS_NORMALIZERS["restitution"],
        },
        "tags": list(tags),
    }


def physics_labels(material) -> Dict:
    """Physics labels from a named material (str) OR a Material object.

    Accepting a Material object is what lets continuously-sampled materials —
    which have no entry in the discrete library — flow through the same schema.
    """
    if isinstance(material, str):
        mat = M.get(material)
    else:
        mat = material
    return _physics_labels(mat.physics, mat.tags)


def material_truth(sample) -> Dict:
    """Ground-truth essence for a continuously-sampled material (for analysis).

    Records the hidden latent factors, the appearance params actually rendered,
    and the nearest named anchor — none of which the model sees as input; they
    let us measure how well it *recovers* the essence from images alone.
    """
    mat = sample.material
    v = mat.visual
    return {
        "factors": dict(sample.factors),
        "nearest_anchor": sample.nearest_anchor,
        "appearance_params": {
            "base_color": list(v.base_color),
            "roughness": v.roughness,
            "metallic": v.metallic,
            "transmission": v.transmission,
            "ior": v.ior,
        },
    }


# --------------------------------------------------------------------------- #
# Record + manifest builders.
# --------------------------------------------------------------------------- #
def build_frame(index: int, file: str, position: Sequence[float],
                look_at: Sequence[float]) -> Dict:
    return {
        "index": int(index),
        "file": file,
        "position": [float(v) for v in position],
        "look_at": [float(v) for v in look_at],
    }


def build_probe_record(spec: Dict, outcome: Dict,
                       trajectory: Optional[List[Dict]] = None) -> Dict:
    """One probe's record: its spec, the summary outcome, optional raw trajectory."""
    rec = {"probe": spec.get("kind", "unknown"), "spec": spec, "outcome": outcome}
    if trajectory is not None:
        rec["trajectory"] = trajectory
    return rec


def build_sample_record(
    scene_id: str,
    split: str,
    shape: str,
    frames: List[Dict],
    resolution: int,
    generator: str,
    *,
    material_name: Optional[str] = None,
    material=None,
    material_id: Optional[str] = None,
    behavior: Optional[List[Dict]] = None,
    material_truth_block: Optional[Dict] = None,
    trajectory: Optional[List[Dict]] = None,
    fps: int = 60,
) -> Dict:
    """Assemble one scene's ``sample.json`` record (identical across generators).

    Supply EITHER ``material_name`` (discrete library path, e.g. Blender
    primitives) OR a ``material`` object + ``material_id`` (continuous-sampler
    path). ``behavior`` is the v2 probe-outcome list; when present it is the
    primary training target.
    """
    if material is not None:
        phys = physics_labels(material)
        mat_label = material_id or "sampled"
    elif material_name is not None:
        phys = physics_labels(material_name)
        mat_label = material_name
    else:
        raise ValueError("provide material_name or (material + material_id)")

    record = {
        "scene_id": scene_id,
        "split": split,
        "schema_version": SCHEMA_VERSION,
        "generator": generator,
        "input": {"shape": shape, "material": mat_label},
        "appearance": {"resolution": int(resolution), "frames": frames},
        "physics": phys,
        "behavior": {"probes": behavior or []},
        "dynamics": {"fps": int(fps), "trajectory": trajectory or []},
    }
    if material_truth_block is not None:
        record["material_truth"] = material_truth_block
    return record


def build_manifest(generator: str, split_summary: Dict,
                   held_out_pairs: Sequence[Sequence[str]],
                   scenes: List[Dict]) -> Dict:
    """Top-level dataset manifest (one per generation run)."""
    return {
        "version": SCHEMA_VERSION,
        "generator": generator,
        "split_summary": dict(split_summary),
        "held_out_combinations": [list(p) for p in held_out_pairs],
        "scenes": [
            {
                "scene_id": s["scene_id"],
                "split": s["split"],
                "shape": s["input"]["shape"],
                "material": s["input"]["material"],
            }
            for s in scenes
        ],
    }
