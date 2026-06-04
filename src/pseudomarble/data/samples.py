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

SCHEMA_VERSION = 1
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
# Physics labels (authored from the material library — no engine required).
# --------------------------------------------------------------------------- #
def physics_labels(material_name: str) -> Dict:
    density, friction, restitution = M.physics_vector(material_name)
    mat = M.get(material_name)
    return {
        "raw": {"density": density, "friction": friction, "restitution": restitution},
        "normalized": {
            "density": density / PHYSICS_NORMALIZERS["density"],
            "friction": friction / PHYSICS_NORMALIZERS["friction"],
            "restitution": restitution / PHYSICS_NORMALIZERS["restitution"],
        },
        "tags": list(mat.tags),
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


def build_sample_record(
    scene_id: str,
    split: str,
    shape: str,
    material_name: str,
    frames: List[Dict],
    resolution: int,
    generator: str,
    trajectory: Optional[List[Dict]] = None,
    fps: int = 60,
) -> Dict:
    """Assemble one scene's ``sample.json`` record (identical across generators)."""
    return {
        "scene_id": scene_id,
        "split": split,
        "generator": generator,
        "input": {"shape": shape, "material": material_name},
        "appearance": {"resolution": int(resolution), "frames": frames},
        "physics": physics_labels(material_name),
        "dynamics": {"fps": int(fps), "trajectory": trajectory or []},
    }


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
