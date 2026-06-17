"""MuJoCo data-generation pipeline — PRIMARY generator, behavior-based task.

For each object (a primitive SHAPE made of a continuously-sampled MATERIAL) this:
  1. renders multi-view images (appearance — what the model sees), and
  2. *acts* on it with three probes and records the OUTCOME (behavior — the
     target): DROP (bounces/settling), TILT (does it slide), PUSH (slide vs
     topple — shape x material interact here).

This is the version that tests "understanding the physical essence of a thing"
rather than "regress three material constants": the materials are continuous (no
10-row lookup to memorize), and the targets are consequences of actions.

The appearance↔physics coupling still lives in one MJCF geom (rgba + density +
friction); the difference is that those values now come from MaterialSampler, and
we additionally simulate probes to get behavior.

Run on your Mac (``pip install -e ".[mujoco]"``)::

    python -m pseudomarble.data.generate_mujoco \
        --output data/pseudo_marble --num-scenes 64 --views 16 --resolution 256 \
        --render-workers 0 --sim-workers 0      # 0 = phase-specific auto

Scenes are fully independent, so generation is embarrassingly parallel — across
*processes*, not threads (a MuJoCo render/sim context is per-process), with manifest
order preserved regardless of finish order. Crucially the two stages have OPPOSITE
optimal widths on a unified-memory Mac, so they run as separate phases:

  * **render** (GPU/Metal) — ``--render-workers``, kept small: there is one GPU
    sharing the 64 GB / 307 GB-s pool with the CPU, so a worker-per-core just
    queues on it (oversubscription, not speedup);
  * **simulate** (CPU ``mj_step``) — ``--sim-workers``, wide: the GPU is idle in
    this phase, so it scales ~linearly across the performance cores.

``--workers`` remains a combined fallback for both when the per-phase flags are 0.

The MJCF builder, outcome summarization, the assemble/serialize seam, and the
parallel scheduler (``parallel.py``) are pure-Python and unit-tested without a
MuJoCo runtime; rendering/simulation are guarded behind ``mujoco``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pseudomarble import materials as M
from pseudomarble import probes as P
from pseudomarble.config import PhysicsConfig, RenderConfig
from pseudomarble.data import samples
from pseudomarble.data.parallel import (
    default_cpu_workers,
    default_render_workers,
    ordered_parallel_map,
    resolve_workers,
)
from pseudomarble.materials import MaterialSampler
from pseudomarble.splits import (
    DEFAULT_REGION_HOLDOUT,
    EXTRAPOLATION_REGION_HOLDOUT,
    RegionHoldout,
)

try:
    import mujoco  # type: ignore

    _HAVE_MUJOCO = True
except Exception:  # pragma: no cover - depends on runtime
    mujoco = None  # type: ignore
    _HAVE_MUJOCO = False


# MuJoCo native primitives. Concave/custom shapes (cup, cone, torus) need a mesh
# asset — the GSO/Objaverse path (docs/GSO_EXPERIMENT.md). ``half_height`` is the
# object's half-extent in z, used to place it on the ground and to find where a
# push lands.
SHAPE_TO_GEOM: Dict[str, Dict] = {
    "box": {"type": "box", "size": "0.15 0.15 0.15", "half_height": 0.15},
    "cube": {"type": "box", "size": "0.15 0.15 0.15", "half_height": 0.15},
    "sphere": {"type": "sphere", "size": "0.18", "half_height": 0.18},
    "cylinder": {"type": "cylinder", "size": "0.10 0.25", "half_height": 0.25},
    "capsule": {"type": "capsule", "size": "0.10 0.20", "half_height": 0.30},
    "ellipsoid": {"type": "ellipsoid", "size": "0.20 0.15 0.10", "half_height": 0.10},
}
# A tall, tippy default set: spans squat (box/ellipsoid) to tall (cylinder/
# capsule), so PUSH produces both sliding and toppling depending on shape.
DEFAULT_SHAPES = ["box", "sphere", "cylinder", "capsule", "ellipsoid"]


def _require_mujoco() -> None:
    if not _HAVE_MUJOCO:
        raise RuntimeError(
            "generate_mujoco requires the 'mujoco' package (arm64 wheel on Mac):\n"
            "  pip install -e \".[mujoco]\"\n"
            "The MJCF builder + outcome summarization work without it for testing."
        )


def half_height(shape: str) -> float:
    return SHAPE_TO_GEOM[shape]["half_height"]


def _restitution_to_solref(restitution: float) -> str:
    """Map restitution -> MuJoCo solref (-stiffness -damping); bouncier == less damping."""
    stiffness = 2000.0
    damping = 30.0 * (1.0 - max(0.0, min(1.0, restitution)))
    return f"-{stiffness:.0f} -{damping:.2f}"


@dataclass(frozen=True)
class MeshAsset:
    """A real (e.g. Google-Scanned-Objects) object for the mesh path of build_mjcf.

    Unlike the primitive path, ``mass`` is set EXPLICITLY (GSO ships measured mass)
    rather than derived from density x volume; MuJoCo computes inertia from the
    mesh scaled to that mass. ``friction`` and ``restitution`` are *assumed* — GSO
    does not measure them (see docs/GSO_EXPERIMENT.md), so honest analysis treats
    them as priors, not signal. ``texture_path`` lets MuJoCo render the real
    scanned appearance; ``collision_paths`` (convex decomposition parts) preserve
    concavity, otherwise the single mesh geom collides as its convex hull.
    """

    name: str
    visual_path: str                          # OBJ/STL/PLY mesh for rendering
    mass: float                               # kg — measured (GSO) or estimated
    half_height: float = 0.15                 # z half-extent, for placement/push
    scale: float = 1.0
    friction: float = 0.5                     # ASSUMED (not measured)
    restitution: float = 0.3                  # ASSUMED (not measured)
    collision_paths: Tuple[str, ...] = field(default_factory=tuple)
    rgba: Tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    texture_path: Optional[str] = None


def build_mjcf(
    shape: Optional[str] = None,
    material: Optional[M.Material] = None,
    object_z: Optional[float] = None,
    ground_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    gravity: float = -9.81,
    *,
    mesh: Optional[MeshAsset] = None,
) -> str:
    """Build a single-scene MJCF string. Pure-Python: unit-testable, no runtime.

    Two modes:
      * primitive — pass ``shape`` + ``material`` (the synthetic MuJoCo path).
      * mesh      — pass ``mesh`` (a ``MeshAsset``) for real scanned objects (GSO).

    ``object_z`` sets the body's starting height (drop height, or resting height
    if None). ``ground_euler`` tilts the ground plane (the TILT probe).
    """
    if mesh is not None:
        return build_mesh_mjcf(mesh, object_z=object_z, ground_euler=ground_euler,
                               gravity=gravity)
    if shape is None or material is None:
        raise ValueError("primitive mode needs shape + material; or pass mesh=...")
    if shape not in SHAPE_TO_GEOM:
        raise ValueError(
            f"shape {shape!r} is not a MuJoCo primitive; have "
            f"{sorted(SHAPE_TO_GEOM)}. Concave/custom shapes need a mesh asset "
            "(the GSO/Objaverse path)."
        )
    geom = SHAPE_TO_GEOM[shape]
    if object_z is None:
        object_z = geom["half_height"] + 0.001  # resting on the ground
    v = material.visual
    p = material.physics
    r, g, b, a = v.base_color
    specular = 0.5 * v.metallic + 0.1
    shininess = max(0.0, min(1.0, 1.0 - v.roughness))
    reflectance = 0.6 * v.metallic
    solref = _restitution_to_solref(p.restitution)
    gx, gy, gz = ground_euler

    return f"""<mujoco model="pseudo_marble">
  <option gravity="0 0 {gravity}" timestep="0.002"/>
  <visual>
    <global offwidth="1280" offheight="1280"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4"/>
  </visual>
  <asset>
    <material name="m_obj" rgba="{r} {g} {b} {a}"
              specular="{specular:.3f}" shininess="{shininess:.3f}"
              reflectance="{reflectance:.3f}"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"
          euler="{gx} {gy} {gz}" friction="{p.friction} 0.005 0.0001"/>
    <body name="object" pos="0 0 {object_z}">
      <freejoint/>
      <geom name="obj" type="{geom['type']}" size="{geom['size']}"
            material="m_obj"
            density="{p.density}" friction="{p.friction} 0.005 0.0001"
            solref="{solref}" solimp="0.9 0.95 0.001"/>
    </body>
  </worldbody>
</mujoco>"""


def build_mesh_mjcf(
    mesh: MeshAsset,
    object_z: Optional[float] = None,
    ground_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    gravity: float = -9.81,
) -> str:
    """Build a single-scene MJCF for a real mesh object. Pure-Python, testable.

    Mass is set explicitly (measured); MuJoCo derives inertia from the mesh. If
    ``collision_paths`` are given (convex decomposition), the visual mesh is
    render-only and the convex parts carry collision + an even share of the mass
    (approximate inertia); otherwise the single mesh geom collides as its convex
    hull. Texture, if provided, renders the real scanned appearance.
    """
    if object_z is None:
        object_z = mesh.half_height + 0.001
    gx, gy, gz = ground_euler
    s = mesh.scale
    fr = mesh.friction
    solref = _restitution_to_solref(mesh.restitution)

    assets = [f'<mesh name="{mesh.name}_vis" file="{mesh.visual_path}" '
              f'scale="{s} {s} {s}"/>']
    for i, cp in enumerate(mesh.collision_paths):
        assets.append(f'<mesh name="{mesh.name}_col{i}" file="{cp}" scale="{s} {s} {s}"/>')
    if mesh.texture_path:
        assets.append(f'<texture name="{mesh.name}_tex" type="2d" file="{mesh.texture_path}"/>')
        assets.append(f'<material name="{mesh.name}_mat" texture="{mesh.name}_tex"/>')
        appearance = f'material="{mesh.name}_mat"'
    else:
        r, g, b, a = mesh.rgba
        appearance = f'rgba="{r} {g} {b} {a}"'

    if mesh.collision_paths:
        n = len(mesh.collision_paths)
        part_mass = mesh.mass / n
        # Visual geom is render-only AND mass-less: without mass="0" MuJoCo would
        # give the full mesh default density (1000 kg/m^3) and add phantom mass on
        # top of the measured mass carried by the collision parts.
        geoms = [f'<geom name="vis" type="mesh" mesh="{mesh.name}_vis" {appearance} '
                 f'contype="0" conaffinity="0" group="2" mass="0"/>']
        for i in range(n):
            geoms.append(
                f'<geom name="col{i}" type="mesh" mesh="{mesh.name}_col{i}" '
                f'mass="{part_mass}" friction="{fr} 0.005 0.0001" '
                f'solref="{solref}" solimp="0.9 0.95 0.001" group="3"/>')
    else:
        geoms = [f'<geom name="obj" type="mesh" mesh="{mesh.name}_vis" {appearance} '
                 f'mass="{mesh.mass}" friction="{fr} 0.005 0.0001" '
                 f'solref="{solref}" solimp="0.9 0.95 0.001"/>']

    asset_block = "\n    ".join(assets)
    geom_block = "\n      ".join(geoms)
    return f"""<mujoco model="pseudo_marble_gso">
  <option gravity="0 0 {gravity}" timestep="0.002"/>
  <visual>
    <global offwidth="1280" offheight="1280"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4"/>
  </visual>
  <asset>
    {asset_block}
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"
          euler="{gx} {gy} {gz}" friction="{fr} 0.005 0.0001"/>
    <body name="object" pos="0 0 {object_z}">
      <freejoint/>
      {geom_block}
    </body>
  </worldbody>
</mujoco>"""


# --------------------------------------------------------------------------- #
# Runtime: rendering + probe simulation (require mujoco).
# --------------------------------------------------------------------------- #
def _quat_to_up(quat) -> List[float]:
    """World-frame direction of the body's local +Z axis, from a (w,x,y,z) quat."""
    w, x, y, z = quat
    return [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]


def _save_png(path: str, rgb) -> None:
    try:
        import imageio.v3 as iio  # type: ignore

        iio.imwrite(path, rgb)
        return
    except Exception:
        pass
    from PIL import Image  # type: ignore

    Image.fromarray(rgb).save(path)


def render_views(shape: str, material: M.Material, renders_dir: str,
                 render_cfg: RenderConfig) -> List[Dict]:
    """Static multi-view renders of the object resting on the ground."""
    _require_mujoco()
    os.makedirs(renders_dir, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(build_mjcf(shape, material))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    frames: List[Dict] = []
    renderer = mujoco.Renderer(model, render_cfg.resolution, render_cfg.resolution)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, half_height(shape)]
    for i, pos in enumerate(samples.fibonacci_sphere_poses(
            render_cfg.num_views, render_cfg.camera_radius)):
        az, el, dist = samples.pose_to_az_el_dist(pos)
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        renderer.update_scene(data, camera=cam)
        _save_png(os.path.join(renders_dir, f"view_{i:03d}.png"), renderer.render())
        frames.append(samples.build_frame(i, f"view_{i:03d}.png", pos, list(cam.lookat)))
    return frames


def _simulate(model, data, sim_seconds: float, fps: int,
              force_fn=None) -> List[Dict]:
    """Step the sim, recording (t, pos, up) at ``fps``. ``force_fn(t)`` may set
    ``data.xfrc_applied`` for the object body to deliver a push."""
    _require_mujoco()
    body = model.body("object").id
    n_steps = int(sim_seconds / model.opt.timestep)
    steps_per_frame = max(1, int((1.0 / fps) / model.opt.timestep))
    traj: List[Dict] = []
    for step in range(n_steps):
        t = step * model.opt.timestep
        if force_fn is not None:
            data.xfrc_applied[body] = force_fn(t)
        mujoco.mj_step(model, data)
        if step % steps_per_frame == 0:
            traj.append({
                "t": round(t, 4),
                "pos": [float(c) for c in data.xpos[body]],
                "up": [round(c, 4) for c in _quat_to_up(data.xquat[body])],
            })
    return traj


def run_drop(shape: str, material: M.Material, spec: P.DropSpec,
             physics_cfg: PhysicsConfig) -> List[Dict]:
    _require_mujoco()
    z = half_height(shape) + spec.height
    model = mujoco.MjModel.from_xml_string(build_mjcf(shape, material, object_z=z))
    return _simulate(model, mujoco.MjData(model), physics_cfg.sim_seconds, physics_cfg.fps)


def run_tilt(shape: str, material: M.Material, spec: P.TiltSpec,
             physics_cfg: PhysicsConfig) -> List[Dict]:
    _require_mujoco()
    angle = math.radians(spec.angle_deg)
    z = half_height(shape) + 0.05
    model = mujoco.MjModel.from_xml_string(
        build_mjcf(shape, material, object_z=z, ground_euler=(angle, 0.0, 0.0))
    )
    return _simulate(model, mujoco.MjData(model), physics_cfg.sim_seconds, physics_cfg.fps)


def run_push(shape: str, material: M.Material, spec: P.PushSpec,
             physics_cfg: PhysicsConfig) -> List[Dict]:
    _require_mujoco()
    model = mujoco.MjModel.from_xml_string(build_mjcf(shape, material))
    data = mujoco.MjData(model)
    settle, window = 0.4, 0.1                      # let it rest, then push briefly
    force = spec.impulse / window                  # impulse = force * window
    rz = (spec.height_frac - 0.5) * 2.0 * half_height(shape)
    az = math.radians(spec.azimuth_deg)
    fx, fy = force * math.cos(az), force * math.sin(az)
    # torque = r x F, r = (0,0,rz): (-rz*Fy, rz*Fx, 0) -> tips a tall object over.
    wrench = [fx, fy, 0.0, -rz * fy, rz * fx, 0.0]

    def force_fn(t):
        return wrench if settle <= t < settle + window else [0.0] * 6

    return _simulate(model, data, physics_cfg.sim_seconds, physics_cfg.fps, force_fn)


_PROBE_RUNNERS = {"drop": run_drop, "tilt": run_tilt, "push": run_push}


def _jitter_seed(material: M.Material) -> int:
    """Deterministic per-material seed so soft-topple sampling is reproducible
    across runs without threading a seed through the phased pipeline."""
    return int(abs(material.physics.density) * 1000.0) % (2 ** 31)


def _soft_topple_final_tilts(shape: str, material: M.Material, base_spec: "P.PushSpec",
                             physics_cfg: PhysicsConfig, base_tilt: float) -> List[float]:
    """Final tilts from the base push plus ``topple_jitter_reps`` action-jittered
    pushes — the sample used to estimate P(topple). Uses ``_PROBE_RUNNERS['push']``
    so it follows any (test) monkeypatch of the push runner."""
    import random

    rng = random.Random(_jitter_seed(material))
    push = _PROBE_RUNNERS["push"]
    tilts = [base_tilt]
    for _ in range(physics_cfg.topple_jitter_reps):
        spec = P.PushSpec(
            impulse=base_spec.impulse * (1.0 + rng.gauss(0.0, physics_cfg.topple_jitter_impulse_rel)),
            height_frac=base_spec.height_frac,
            azimuth_deg=base_spec.azimuth_deg + rng.gauss(0.0, physics_cfg.topple_jitter_azimuth_deg),
        )
        tilts.append(P.summarize(push(shape, material, spec, physics_cfg)).final_tilt_deg)
    return tilts


def run_probes(shape: str, material: M.Material,
               physics_cfg: PhysicsConfig, keep_trajectory: bool = False) -> List[Dict]:
    """Run the drop+tilt+push battery and summarize each into a probe record.

    With ``physics_cfg.topple_jitter_reps > 0`` the push probe's ``toppled`` field
    holds a smooth P(topple) in [0,1] (jitter-averaged) instead of the hard bool —
    the F8 mitigation; the deterministic fields still come from the base push."""
    records: List[Dict] = []
    for spec in P.default_probes():
        traj = _PROBE_RUNNERS[spec.kind](shape, material, spec, physics_cfg)
        outcome = P.summarize(traj)
        outcome_dict = outcome.to_dict()
        spec_dict = P.spec_to_dict(spec)
        if spec.kind == "push" and physics_cfg.topple_jitter_reps > 0:
            tilts = _soft_topple_final_tilts(shape, material, spec, physics_cfg,
                                             outcome.final_tilt_deg)
            outcome_dict["toppled"] = P.soft_topple_probability(tilts)
            spec_dict["topple_jitter"] = {
                "reps": physics_cfg.topple_jitter_reps,
                "impulse_rel": physics_cfg.topple_jitter_impulse_rel,
                "azimuth_deg": physics_cfg.topple_jitter_azimuth_deg,
                "n_samples": len(tilts),
            }
        records.append(samples.build_probe_record(
            spec_dict, outcome_dict,
            trajectory=traj if keep_trajectory else None,
        ))
    return records


# --------------------------------------------------------------------------- #
# Scene built in three separable stages so the GPU-bound render and the CPU-bound
# simulation can be scheduled at DIFFERENT widths (see data/parallel.py):
#   render_scene   — GPU (Metal): multi-view images           → frames
#   simulate_scene — CPU (mj_step): drop/tilt/push battery     → behavior records
#   assemble_scene — cheap, no mujoco: build the record + write sample.json
# build_scene runs all three in order (the serial / backward-compatible path).
# --------------------------------------------------------------------------- #
def render_scene(scene_id: str, shape: str, sample: "M.MaterialSample",
                 out_root: str, render_cfg: RenderConfig) -> List[Dict]:
    """GPU stage: render the multi-view images for one scene; returns the frames."""
    scene_dir = os.path.join(out_root, scene_id)
    return render_views(shape, sample.material, os.path.join(scene_dir, "renders"),
                        render_cfg)


def simulate_scene(shape: str, sample: "M.MaterialSample",
                   physics_cfg: PhysicsConfig, keep_trajectory: bool = False) -> List[Dict]:
    """CPU stage: run the drop/tilt/push battery for one scene; returns behavior."""
    return run_probes(shape, sample.material, physics_cfg, keep_trajectory)


def assemble_scene(scene_id: str, shape: str, sample: "M.MaterialSample", split: str,
                   out_root: str, render_cfg: RenderConfig, physics_cfg: PhysicsConfig,
                   frames: List[Dict], behavior: List[Dict]) -> Dict:
    """Assemble the paired record from rendered frames + simulated behavior, and
    write ``sample.json``. Pure-Python (no mujoco): the join/serialize seam."""
    record = samples.build_sample_record(
        scene_id=scene_id, split=split, shape=shape, frames=frames,
        resolution=render_cfg.resolution, generator="mujoco",
        material=sample.material, material_id=scene_id,
        behavior=behavior, material_truth_block=samples.material_truth(sample),
        fps=physics_cfg.fps,
    )
    scene_dir = os.path.join(out_root, scene_id)
    os.makedirs(scene_dir, exist_ok=True)
    with open(os.path.join(scene_dir, "sample.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    return record


def build_scene(scene_id: str, shape: str, sample: "M.MaterialSample", split: str,
                out_root: str, render_cfg: RenderConfig, physics_cfg: PhysicsConfig,
                keep_trajectory: bool = False) -> Dict:
    """Generate one paired sample (appearance + behavior) and write sample.json.

    Runs the three stages in order; the parallel ``main`` instead schedules the
    render and sim stages separately so each can use its own worker width.
    """
    frames = render_scene(scene_id, shape, sample, out_root, render_cfg)
    behavior = simulate_scene(shape, sample, physics_cfg, keep_trajectory)
    return assemble_scene(scene_id, shape, sample, split, out_root, render_cfg,
                          physics_cfg, frames, behavior)


def _render_task(task: Tuple) -> List[Dict]:
    """Module-level (picklable) GPU-stage worker: render one scene's views.

    Each worker builds its own ``MjModel`` / ``Renderer`` (MuJoCo contexts are
    per-process and not thread-safe — exactly why this fans out across *processes*)
    and writes only into that scene's ``renders/`` dir, so workers share no state.
    """
    rec, out_root, render_cfg = task
    return render_scene(rec["scene_id"], rec["shape"], rec["sample"], out_root, render_cfg)


def _sim_task(task: Tuple) -> List[Dict]:
    """Module-level (picklable) CPU-stage worker: run one scene's probe battery."""
    rec, physics_cfg, keep_traj = task
    return simulate_scene(rec["shape"], rec["sample"], physics_cfg, keep_traj)


def assign_scenes(shapes: List[str], holdout: RegionHoldout, num_scenes: int,
                  seed: int) -> List[Dict]:
    """Sample (shape, continuous material) scenes and label train/test by region."""
    import random

    from pseudomarble.config import PHYSICS_NORMALIZERS as N

    sampler = MaterialSampler(seed=seed)
    rng = random.Random(seed)
    out: List[Dict] = []
    for i in range(num_scenes):
        shape = rng.choice(shapes)
        ms = sampler.sample(material_id=f"scene_{i:06d}")
        p = ms.material.physics
        essence_norm = {
            "density": p.density / N["density"],
            "friction": p.friction / N["friction"],
            "restitution": p.restitution / N["restitution"],
        }
        split = holdout.label(essence_norm, shape)
        out.append({"scene_id": f"{split}_{i:06d}", "shape": shape,
                    "sample": ms, "split": split})
    return out


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pseudo-marble MuJoCo behavior-based generation")
    p.add_argument("--output", default="data/pseudo_marble")
    p.add_argument("--num-scenes", type=int, default=16)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--views", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--holdout-kind", default="extrapolation",
                   choices=["interpolation", "extrapolation"],
                   help="held-out essence region: extrapolation (corner, real teeth) "
                        "or interpolation (interior box, weak)")
    p.add_argument("--shapes", default=",".join(DEFAULT_SHAPES),
                   help="comma-separated MuJoCo primitive shape ids")
    p.add_argument("--keep-trajectory", action="store_true",
                   help="store full per-probe trajectories (larger files)")
    p.add_argument("--topple-jitter-reps", type=int, default=0,
                   help="if > 0, record the push `toppled` field as a smooth P(topple) "
                        "in [0,1] averaged over this many action-jittered pushes (the F8 "
                        "mitigation for chaotic binary toppling); 0 = hard bool (default)")
    p.add_argument("--topple-jitter-impulse", type=float, default=0.03,
                   help="relative push-impulse jitter std for soft topple")
    p.add_argument("--topple-jitter-azimuth", type=float, default=2.0,
                   help="push-azimuth jitter std (degrees) for soft topple")
    p.add_argument("--workers", type=int, default=0,
                   help="combined fallback width for BOTH phases when the per-phase "
                        "flags are 0 (0 = phase-specific auto). Render and sim have "
                        "opposite optima on a unified-memory Mac, so prefer the "
                        "per-phase flags below")
    p.add_argument("--render-workers", type=int, default=0,
                   help="GPU-stage (Metal render) processes (0 = auto, kept small: "
                        "one GPU shares the memory bus, so a worker-per-core just "
                        "queues on it — oversubscription, not speedup)")
    p.add_argument("--sim-workers", type=int, default=0,
                   help="CPU-stage (drop/tilt/push mj_step) processes (0 = auto: "
                        "most cores; the GPU is idle during this phase so it scales "
                        "~linearly across the performance cores)")
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    _require_mujoco()
    args = parse_args(argv)
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    render_cfg = RenderConfig(resolution=args.resolution, num_views=args.views)
    physics_cfg = PhysicsConfig(
        topple_jitter_reps=args.topple_jitter_reps,
        topple_jitter_impulse_rel=args.topple_jitter_impulse,
        topple_jitter_azimuth_deg=args.topple_jitter_azimuth,
    )

    holdout = (EXTRAPOLATION_REGION_HOLDOUT if args.holdout_kind == "extrapolation"
               else DEFAULT_REGION_HOLDOUT)
    assignments = assign_scenes(shapes, holdout, args.num_scenes, args.seed)
    os.makedirs(args.output, exist_ok=True)
    n = len(assignments)

    # Per-phase widths: explicit per-phase flag > --workers fallback > phase auto.
    render_workers = resolve_workers(args.render_workers or args.workers, n,
                                     default=default_render_workers())
    sim_workers = resolve_workers(args.sim_workers or args.workers, n,
                                  default=default_cpu_workers())
    print(f"[pseudo-marble:mujoco] generating {n} scenes "
          f"(render phase: {render_workers} proc / sim phase: {sim_workers} proc)")

    # Phase 1 — GPU: render every scene's views (narrow; one shared GPU).
    render_tasks = [(rec, args.output, render_cfg) for rec in assignments]
    frames_list: List[List[Dict]] = ordered_parallel_map(
        _render_task, render_tasks, render_workers,
        on_done=lambda i, _f: print(
            f"[pseudo-marble:mujoco] rendered {assignments[i]['scene_id']}"))

    # Phase 2 — CPU: run the drop/tilt/push battery (wide; GPU idle now).
    sim_tasks = [(rec, physics_cfg, args.keep_trajectory) for rec in assignments]
    behavior_list: List[List[Dict]] = ordered_parallel_map(
        _sim_task, sim_tasks, sim_workers,
        on_done=lambda i, _b: print(
            f"[pseudo-marble:mujoco] simulated {assignments[i]['scene_id']}"))

    # Phase 3 — cheap join: assemble each record + write sample.json (serial).
    scenes: List[Dict] = []
    for i, rec in enumerate(assignments):
        record = assemble_scene(rec["scene_id"], rec["shape"], rec["sample"],
                                rec["split"], args.output, render_cfg, physics_cfg,
                                frames_list[i], behavior_list[i])
        scenes.append(record)
        print(f"[pseudo-marble:mujoco] built {rec['scene_id']} "
              f"({rec['shape']} / ess~{rec['sample'].nearest_anchor} / {rec['split']})")

    n_test = sum(1 for s in scenes if s["split"] == "test")
    manifest = samples.build_manifest(
        "mujoco",
        {"n_train": len(scenes) - n_test, "n_test": n_test, "n_scenes": len(scenes)},
        [],  # held-out combos are a continuous region now, not a list
        scenes,
    )
    manifest["holdout_region"] = {
        "kind": holdout.kind,
        "density": list(holdout.density or []),
        "friction": list(holdout.friction or []),
        "restitution": list(holdout.restitution or []),
    }
    with open(os.path.join(args.output, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[pseudo-marble:mujoco] wrote {len(scenes)} scenes "
          f"({n_test} test) -> {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
