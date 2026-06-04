"""Configuration dataclasses and shared constants for the pseudo-marble stack.

Kept pure-Python and dependency-free so it imports anywhere (Blender's Python,
MLX on the Mac, or CI). Values are intentionally modest to fit an M5 / 64 GB
unified-memory budget — this is a *pseudo*-marble; we trade fidelity for the
ability to run, perturb, and reason about the whole loop on commodity hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# Targets for the physics decoder live on very different scales (density in the
# thousands, friction/restitution in [0,1]). These normalizers map them into a
# comparable range so the coherence loss is not dominated by density's magnitude.
PHYSICS_NORMALIZERS = {
    "density": 8000.0,  # divide kg/m^3 by this -> roughly [0,1]
    "friction": 1.5,
    "restitution": 1.0,
}


@dataclass(frozen=True)
class RenderConfig:
    """Multi-view render settings (the renderer / appearance path)."""

    resolution: int = 256
    num_views: int = 16
    camera_radius: float = 2.6  # metres from object centre
    camera_fov_deg: float = 45.0
    engine: str = "CYCLES"  # "CYCLES" (photoreal) or "BLENDER_EEVEE_NEXT"
    samples: int = 64  # Cycles samples; low to keep generation fast
    seed: int = 0


@dataclass(frozen=True)
class PhysicsConfig:
    """Rigid-body ground-truth settings (the simulator / physics path)."""

    gravity: float = -9.81
    drop_height: float = 1.0
    sim_seconds: float = 2.0
    fps: int = 60
    substeps: int = 10
    collision_method: str = "coacd"  # "coacd" | "vhacd" | "convex_hull"
    collision_max_hulls: int = 16


@dataclass(frozen=True)
class DatagenConfig:
    """Top-level data-generation run configuration."""

    output_dir: str = "data/pseudo_marble"
    num_scenes: int = 256
    seed: int = 1234
    holdout_frac: float = 0.2  # fraction of (shape,material) pairs held out for test
    render: RenderConfig = field(default_factory=RenderConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the MLX shared-latent model (see docs/ARCHITECTURE.md).

    Implemented incrementally — these are the agreed dimensions, not a promise
    that every field is wired up yet.
    """

    latent_dim: int = 256
    encoder_width: int = 512
    render_decoder_width: int = 512
    physics_decoder_width: int = 256
    num_gaussians: int = 4096  # simplified MLX splat budget (not full 3DGS)
    coherence_weight: float = 1.0
