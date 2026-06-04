"""MuJoCo data-generation pipeline — the PRIMARY generator.

Why MuJoCo is the default (see docs/ARCHITECTURE.md):
  * arm64-native wheels, runs on a MacBook Pro with no Docker / no bpy quirks;
  * the appearance<->physics coupling lives in ONE geom definition — ``rgba``
    (appearance) + ``density`` (-> auto mass) + ``friction`` (physics) — which is
    literally our ``Material`` dataclass expressed in MJCF;
  * physics ground truth (mass, contacts, trajectory) is the engine's *native*
    output, so there is no separate "bake" step the way Blender needs.

Honest catch: MuJoCo's renderer is not photorealistic — it cannot do glass
transmission or ice convincingly, so the appearance->physics cue is weaker for
*transparent* materials specifically. Opaque materials (steel, rubber, wood,
foam, stone, ceramic, plastic, cardboard) render fine. We accept this because the
experiment measures coherence, not photorealism (see TAXONOMY_NOTES).

MuJoCo also has no first-class restitution scalar (bounce emerges from
solref/solimp). The restitution *label* always comes from ``materials.py``; we
only approximate bounce in the simulated trajectory via solref.

Run (on your Mac, with ``pip install -e ".[mujoco]"``)::

    python -m pseudomarble.data.generate_mujoco \
        --output data/pseudo_marble --num-scenes 16 --views 16 --resolution 256

The MJCF builder is pure-Python and import-guard-free, so it is unit-tested
without a MuJoCo runtime. Rendering / simulation are guarded behind the
``mujoco`` import.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

from pseudomarble import materials as M
from pseudomarble.config import PhysicsConfig, RenderConfig
from pseudomarble.data import samples
from pseudomarble.splits import assign_scene_materials, make_combination_split

try:
    import mujoco  # type: ignore

    _HAVE_MUJOCO = True
except Exception:  # pragma: no cover - depends on runtime
    mujoco = None  # type: ignore
    _HAVE_MUJOCO = False


# MuJoCo native primitive geoms. Concave / custom shapes (cup, cone, torus) come
# in as mesh assets — that is the Objaverse / Google-Scanned-Objects path
# (docs/GSO_EXPERIMENT.md), out of scope for the primitive smoke test.
SHAPE_TO_GEOM: Dict[str, Dict] = {
    "box": {"type": "box", "size": "0.15 0.15 0.15"},
    "cube": {"type": "box", "size": "0.15 0.15 0.15"},
    "sphere": {"type": "sphere", "size": "0.18"},
    "cylinder": {"type": "cylinder", "size": "0.15 0.18"},
    "capsule": {"type": "capsule", "size": "0.12 0.15"},
    "ellipsoid": {"type": "ellipsoid", "size": "0.20 0.15 0.10"},
}
DEFAULT_SHAPES = ["box", "sphere", "cylinder", "capsule", "ellipsoid"]


def _require_mujoco() -> None:
    if not _HAVE_MUJOCO:
        raise RuntimeError(
            "generate_mujoco requires the 'mujoco' package (arm64 wheel on Mac):\n"
            "  pip install -e \".[mujoco]\"\n"
            "The MJCF builder (build_mjcf) works without it for testing."
        )


def _restitution_to_solref(restitution: float) -> str:
    """Map a restitution coefficient to MuJoCo solref (-stiffness -damping).

    Crude but monotonic: bouncier material -> less contact damping. The authored
    restitution label is what the model learns; this only shapes the trajectory.
    """
    stiffness = 2000.0
    damping = 30.0 * (1.0 - max(0.0, min(1.0, restitution)))  # less damping == bouncier
    return f"-{stiffness:.0f} -{damping:.2f}"


def build_mjcf(shape: str, material: M.Material, drop_height: float = 1.0,
               gravity: float = -9.81) -> str:
    """Build a single-scene MJCF string. Pure-Python: unit-testable, no runtime.

    The geom binds appearance (rgba) and physics (density -> mass, friction,
    solref-from-restitution) in one place — the whole Marble idea in one element.
    """
    if shape not in SHAPE_TO_GEOM:
        raise ValueError(
            f"shape {shape!r} is not a MuJoCo primitive; have "
            f"{sorted(SHAPE_TO_GEOM)}. Concave/custom shapes need a mesh asset "
            "(the GSO/Objaverse path)."
        )
    geom = SHAPE_TO_GEOM[shape]
    v = material.visual
    p = material.physics
    r, g, b, a = v.base_color
    # MuJoCo specular/shininess are scalars; approximate from metallic/roughness.
    specular = 0.5 * v.metallic + 0.1
    shininess = max(0.0, min(1.0, 1.0 - v.roughness))
    reflectance = 0.6 * v.metallic
    solref = _restitution_to_solref(p.restitution)

    return f"""<mujoco model="pseudo_marble">
  <option gravity="0 0 {gravity}" timestep="0.002"/>
  <visual>
    <global offwidth="1280" offheight="1280"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4"/>
  </visual>
  <asset>
    <material name="m_{material.name}" rgba="{r} {g} {b} {a}"
              specular="{specular:.3f}" shininess="{shininess:.3f}"
              reflectance="{reflectance:.3f}"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"
          friction="{p.friction} 0.005 0.0001"/>
    <body name="object" pos="0 0 {drop_height}">
      <freejoint/>
      <geom name="obj" type="{geom['type']}" size="{geom['size']}"
            material="m_{material.name}"
            density="{p.density}" friction="{p.friction} 0.005 0.0001"
            solref="{solref}" solimp="0.9 0.95 0.001"/>
    </body>
  </worldbody>
</mujoco>"""


def _save_png(path: str, rgb) -> None:
    try:
        import imageio.v3 as iio  # type: ignore

        iio.imwrite(path, rgb)
        return
    except Exception:
        pass
    from PIL import Image  # type: ignore

    Image.fromarray(rgb).save(path)


def render_and_simulate(
    shape: str, material: M.Material, scene_dir: str,
    render_cfg: RenderConfig, physics_cfg: PhysicsConfig,
) -> tuple[List[Dict], List[Dict]]:
    """Render N turntable views and bake a drop trajectory. Requires mujoco."""
    _require_mujoco()
    os.makedirs(scene_dir, exist_ok=True)
    renders_dir = os.path.join(scene_dir, "renders")
    os.makedirs(renders_dir, exist_ok=True)

    xml = build_mjcf(shape, material, physics_cfg.drop_height, physics_cfg.gravity)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # --- appearance path: multi-view renders from a turntable free camera ---
    frames: List[Dict] = []
    renderer = mujoco.Renderer(model, render_cfg.resolution, render_cfg.resolution)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, physics_cfg.drop_height]
    for i, pos in enumerate(samples.fibonacci_sphere_poses(
            render_cfg.num_views, render_cfg.camera_radius)):
        az, el, dist = samples.pose_to_az_el_dist(pos)
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        fname = f"view_{i:03d}.png"
        _save_png(os.path.join(renders_dir, fname), rgb)
        frames.append(samples.build_frame(i, fname, pos, list(cam.lookat)))

    # --- physics path: bake the drop and record the object trajectory ---
    trajectory: List[Dict] = []
    n_steps = int(physics_cfg.sim_seconds / model.opt.timestep)
    steps_per_frame = max(1, int((1.0 / physics_cfg.fps) / model.opt.timestep))
    obj_body = model.body("object").id
    frame_idx = 0
    for step in range(n_steps):
        mujoco.mj_step(model, data)
        if step % steps_per_frame == 0:
            frame_idx += 1
            pos = data.xpos[obj_body]
            trajectory.append({"frame": frame_idx, "position": [float(c) for c in pos]})
    return frames, trajectory


def build_scene(scene_id: str, shape: str, material_name: str, split: str,
                out_root: str, render_cfg: RenderConfig,
                physics_cfg: PhysicsConfig) -> Dict:
    """Generate one paired sample and write its sample.json."""
    material = M.get(material_name)
    scene_dir = os.path.join(out_root, scene_id)
    frames, trajectory = render_and_simulate(
        shape, material, scene_dir, render_cfg, physics_cfg
    )
    record = samples.build_sample_record(
        scene_id=scene_id, split=split, shape=shape, material_name=material_name,
        frames=frames, resolution=render_cfg.resolution, generator="mujoco",
        trajectory=trajectory, fps=physics_cfg.fps,
    )
    os.makedirs(scene_dir, exist_ok=True)
    with open(os.path.join(scene_dir, "sample.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    return record


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pseudo-marble MuJoCo data generation")
    p.add_argument("--output", default="data/pseudo_marble")
    p.add_argument("--num-scenes", type=int, default=16)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--views", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--shapes", default=",".join(DEFAULT_SHAPES),
                   help="comma-separated MuJoCo primitive shape ids")
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    _require_mujoco()
    args = parse_args(argv)
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    mat_names = list(M.names())

    split = make_combination_split(shapes, mat_names, args.holdout_frac, args.seed)
    assignments = assign_scene_materials(
        shapes, mat_names, split, args.num_scenes, args.seed
    )
    render_cfg = RenderConfig(resolution=args.resolution, num_views=args.views)
    physics_cfg = PhysicsConfig()

    os.makedirs(args.output, exist_ok=True)
    scenes: List[Dict] = []
    for rec in assignments:
        out = build_scene(
            rec["scene_id"], rec["shape"], rec["material"], rec["split"],
            args.output, render_cfg, physics_cfg,
        )
        scenes.append(out)
        print(f"[pseudo-marble:mujoco] built {rec['scene_id']} "
              f"({rec['shape']} / {rec['material']} / {rec['split']})")

    manifest = samples.build_manifest(
        "mujoco", split.summary(), split.test_pairs, scenes
    )
    with open(os.path.join(args.output, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[pseudo-marble:mujoco] wrote {len(scenes)} scenes -> {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
