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

    # Soft-topple option (mitigates the chaotic binary `toppled` near tipping points;
    # see docs/FINDINGS.md F8). With reps > 0 the PUSH probe is run reps extra times
    # with small action jitter and the `toppled` field records P(topple) in [0,1]
    # (a smooth target) instead of the hard bool. reps = 0 keeps the binary default.
    topple_jitter_reps: int = 0
    topple_jitter_impulse_rel: float = 0.03   # relative push-impulse jitter (std)
    topple_jitter_azimuth_deg: float = 2.0    # push-azimuth jitter in degrees (std)


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
    """Shape of the shared-latent model (see docs/ARCHITECTURE.md).

    The architecture is backend-agnostic: ``models/mlx_net.py`` (trains on the
    Mac) and ``models/numpy_net.py`` (a forward-only reference that runs/tests in
    any session) build the *same* layers from these dimensions.
    """

    # Encoder (multi-view CNN -> shared latent z).
    image_size: int = 128          # advisory; encoder global-pools, so any HxW works
    in_channels: int = 3
    conv_channels: tuple = (32, 64, 128)
    conv_kernel: int = 3
    conv_stride: int = 2
    latent_dim: int = 256
    # FSQ information bottleneck (F17): 0 = off (continuous z, the default).
    # When k > 0, z is projected to k dims, each quantized to 3 levels
    # ({-1,0,1} via round(tanh), straight-through gradient), then expanded
    # back to latent_dim before the heads. The latent then carries AT MOST
    # k trits = k*log2(3) ~ 1.585*k bits about the scene.
    latent_trits: int = 0
    encoder_width: int = 512

    # Behavior head (z -> flattened drop/tilt/push outcomes). Equals
    # probes.BEHAVIOR_DIM (PROBE_ORDER x OUTCOME_FIELDS); a test guards the match.
    behavior_dim: int = 21
    behavior_head_width: int = 256
    behavior_weight: float = 1.0   # the real target's loss weight; set 0 to train a
                                   # render-only model (and render_weight=0 -> behavior-only)
                                   # for the coherence experiment's independent baseline.

    # Auxiliary essence head (z -> density/friction/restitution, normalized).
    essence_dim: int = 3
    essence_head_width: int = 128
    essence_weight: float = 0.3

    # Render head (z -> reconstructed canonical appearance). A lightweight conv
    # decoder (nearest-upsample + conv), NOT a Gaussian splat decoder: simpler,
    # CPU/Metal-portable, and sufficient because the coherence experiment measures
    # whether appearance and behavior move together, not photorealism. Output is
    # image_size x image_size x 3; image_size must be render_seed * 2^k.
    render_seed: int = 4           # spatial size of the decoder's seed feature map
    render_channels: int = 32      # channels in each decoder block
    render_weight: float = 1.0     # reconstruction loss weight
    coherence_weight: float = 1.0  # reserved for the coherence experiment


def conv_output_channels(cfg: "ModelConfig") -> int:
    """Channels after the conv stack (= last conv layer's channel count)."""
    return cfg.conv_channels[-1]


def num_upsample_steps(cfg: "ModelConfig") -> int:
    """How many 2x upsamples take the decoder seed to image_size (must be exact)."""
    ratio = cfg.image_size / cfg.render_seed
    steps = 0
    v = cfg.render_seed
    while v < cfg.image_size:
        v *= 2
        steps += 1
    if v != cfg.image_size:
        raise ValueError(
            f"image_size ({cfg.image_size}) must be render_seed ({cfg.render_seed}) "
            f"* 2^k; got ratio {ratio}"
        )
    return steps
