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
        --output data/pseudo_marble --num-scenes 64 --views 16 --resolution 256

The MJCF builder and outcome summarization are pure-Python and unit-tested
without a MuJoCo runtime; rendering/simulation are guarded behind ``mujoco``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

from pseudomarble import materials as M
from pseudomarble import probes as P
from pseudomarble.config import PhysicsConfig, RenderConfig
from pseudomarble.data import samples
from pseudomarble.materials import MaterialSampler
from pseudomarble.splits import DEFAULT_REGION_HOLDOUT, RegionHoldout

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


def build_mjcf(
    shape: str,
    material: M.Material,
    object_z: Optional[float] = None,
    ground_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    gravity: float = -9.81,
) -> str:
    """Build a single-scene MJCF string. Pure-Python: unit-testable, no runtime.

    ``object_z`` sets the body's starting height (drop height, or resting height
    if None). ``ground_euler`` tilts the ground plane (the TILT probe). The geom
    binds appearance (rgba) and physics (density->mass, friction,
    solref-from-restitution) in one element — the Marble idea in one place.
    """
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


def run_probes(shape: str, material: M.Material,
               physics_cfg: PhysicsConfig, keep_trajectory: bool = False) -> List[Dict]:
    """Run the drop+tilt+push battery and summarize each into a probe record."""
    records: List[Dict] = []
    for spec in P.default_probes():
        traj = _PROBE_RUNNERS[spec.kind](shape, material, spec, physics_cfg)
        outcome = P.summarize(traj)
        records.append(samples.build_probe_record(
            P.spec_to_dict(spec), outcome.to_dict(),
            trajectory=traj if keep_trajectory else None,
        ))
    return records


def build_scene(scene_id: str, shape: str, sample: "M.MaterialSample", split: str,
                out_root: str, render_cfg: RenderConfig, physics_cfg: PhysicsConfig,
                keep_trajectory: bool = False) -> Dict:
    """Generate one paired sample (appearance + behavior) and write sample.json."""
    scene_dir = os.path.join(out_root, scene_id)
    frames = render_views(shape, sample.material, os.path.join(scene_dir, "renders"),
                          render_cfg)
    behavior = run_probes(shape, sample.material, physics_cfg, keep_trajectory)
    record = samples.build_sample_record(
        scene_id=scene_id, split=split, shape=shape, frames=frames,
        resolution=render_cfg.resolution, generator="mujoco",
        material=sample.material, material_id=scene_id,
        behavior=behavior, material_truth_block=samples.material_truth(sample),
        fps=physics_cfg.fps,
    )
    os.makedirs(scene_dir, exist_ok=True)
    with open(os.path.join(scene_dir, "sample.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    return record


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
    p.add_argument("--shapes", default=",".join(DEFAULT_SHAPES),
                   help="comma-separated MuJoCo primitive shape ids")
    p.add_argument("--keep-trajectory", action="store_true",
                   help="store full per-probe trajectories (larger files)")
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    _require_mujoco()
    args = parse_args(argv)
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    render_cfg = RenderConfig(resolution=args.resolution, num_views=args.views)
    physics_cfg = PhysicsConfig()

    assignments = assign_scenes(shapes, DEFAULT_REGION_HOLDOUT, args.num_scenes, args.seed)
    os.makedirs(args.output, exist_ok=True)
    scenes: List[Dict] = []
    for rec in assignments:
        out = build_scene(rec["scene_id"], rec["shape"], rec["sample"], rec["split"],
                          args.output, render_cfg, physics_cfg, args.keep_trajectory)
        scenes.append(out)
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
        "friction": list(DEFAULT_REGION_HOLDOUT.friction or []),
        "restitution": list(DEFAULT_REGION_HOLDOUT.restitution or []),
    }
    with open(os.path.join(args.output, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[pseudo-marble:mujoco] wrote {len(scenes)} scenes "
          f"({n_test} test) -> {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
