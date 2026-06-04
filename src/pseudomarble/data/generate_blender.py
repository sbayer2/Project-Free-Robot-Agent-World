"""Blender (bpy) data-generation pipeline — the renderer + simulator ground truth.

Run headless inside Blender, NOT in plain CPython::

    blender --background --python src/pseudomarble/data/generate_blender.py -- \
        --output data/pseudo_marble --num-scenes 64 --resolution 256 --views 16

For every scene it produces *paired* data from one material identity:

  * appearance path : N RGB renders from known camera poses (+ camera JSON),
                      directly consumable by a Gaussian-splat decoder.
  * physics path    : (density, friction, restitution) labels, object mass,
                      collision-shape metadata, and a short rigid-body drop
                      trajectory as dynamic ground truth.

The single shared input ("latent input") is the scene description: shape id +
material name. Appearance and physics are emitted from that *same* description,
which is the whole point — the coupling is in the data by construction, and the
model's job is to recover and generalize it.

This module is import-guarded: importing it without ``bpy`` raises a clear error
only when you actually try to build a scene, so the rest of the package and the
test suite stay importable in CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

# ``bpy`` only exists inside Blender. Keep the import soft so CI can import the
# file for linting without a Blender runtime.
try:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore

    _HAVE_BPY = True
except Exception:  # pragma: no cover - depends on runtime
    bpy = None  # type: ignore
    Vector = None  # type: ignore
    _HAVE_BPY = False


# --------------------------------------------------------------------------- #
# Package-relative imports that also work when Blender runs this file by path.
# --------------------------------------------------------------------------- #
def _ensure_package_on_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.abspath(os.path.join(here, "..", ".."))  # .../src
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


_ensure_package_on_path()
from pseudomarble import materials as M  # noqa: E402
from pseudomarble.data import samples  # noqa: E402  (shared sample.json contract)


def _require_bpy() -> None:
    if not _HAVE_BPY:
        raise RuntimeError(
            "generate_blender requires Blender's Python (bpy). Run via:\n"
            "  blender --background --python "
            "src/pseudomarble/data/generate_blender.py -- <args>"
        )


# --------------------------------------------------------------------------- #
# Scene construction
# --------------------------------------------------------------------------- #
def reset_scene() -> None:
    _require_bpy()
    bpy.ops.wm.read_factory_settings(use_empty=True)


def add_primitive_shape(shape_id: str):
    """Add a unit-scale primitive. Real runs swap this for Objaverse imports.

    Primitives give us a self-contained, dependency-free smoke test of the whole
    pipeline before wiring in a 3D asset library.
    """
    _require_bpy()
    builders = {
        "cube": lambda: bpy.ops.mesh.primitive_cube_add(size=1.0),
        "sphere": lambda: bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5),
        "cylinder": lambda: bpy.ops.mesh.primitive_cylinder_add(radius=0.4, depth=1.0),
        "cone": lambda: bpy.ops.mesh.primitive_cone_add(radius1=0.5, depth=1.0),
        "torus": lambda: bpy.ops.mesh.primitive_torus_add(
            major_radius=0.4, minor_radius=0.15
        ),
        "cup": _build_cup,  # an actual concave shape, to exercise the cavity case
    }
    if shape_id not in builders:
        raise ValueError(f"unknown primitive shape {shape_id!r}; have {sorted(builders)}")
    builders[shape_id]()
    return bpy.context.active_object


def _build_cup() -> None:
    """A crude open-top cup: the concavity that convex-hull collision would lose."""
    _require_bpy()
    bpy.ops.mesh.primitive_cylinder_add(radius=0.35, depth=0.8)
    outer = bpy.context.active_object
    bpy.ops.mesh.primitive_cylinder_add(radius=0.30, depth=0.8, location=(0, 0, 0.1))
    inner = bpy.context.active_object
    mod = outer.modifiers.new("hollow", type="BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.object = inner
    bpy.context.view_layer.objects.active = outer
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(inner, do_unlink=True)


def apply_material(obj, material: M.Material) -> None:
    """Bind BOTH the visual BSDF and the rigid-body physics from one identity."""
    _require_bpy()
    v = material.visual

    mat = bpy.data.materials.new(name=f"pm_{material.name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = v.base_color
    bsdf.inputs["Roughness"].default_value = v.roughness
    bsdf.inputs["Metallic"].default_value = v.metallic
    # Input names vary across Blender versions; set transmission/IOR defensively.
    for key in ("Transmission Weight", "Transmission"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = v.transmission
            break
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = v.ior
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # Physics: friction / restitution / density-derived mass (Bullet rigid body).
    p = material.physics
    bpy.ops.rigidbody.object_add()
    rb = obj.rigid_body
    rb.friction = p.friction
    rb.restitution = p.restitution
    rb.linear_damping = p.linear_damping
    rb.angular_damping = p.angular_damping
    rb.collision_shape = "CONVEX_HULL"  # placeholder; faithful hulls via collision.py
    # Approximate mass from a unit-ish bounding volume * density. Real runs use
    # the watertight mesh volume from mesh_validate.mass_from_mesh().
    dims = obj.dimensions
    approx_vol = max(1e-4, dims.x * dims.y * dims.z * 0.5)
    rb.mass = p.density * approx_vol


def setup_world_and_light() -> None:
    _require_bpy()
    world = bpy.data.worlds.new("pm_world")
    world.use_nodes = True
    bpy.context.scene.world = world
    bpy.ops.object.light_add(type="SUN")
    bpy.context.active_object.data.energy = 3.0


def render_views(obj, out_dir: str, num_views: int, radius: float, res: int,
                 engine: str, samples: int) -> List[Dict]:
    _require_bpy()
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = scene.render.resolution_y = res
    if engine == "CYCLES":
        scene.cycles.samples = samples

    cam_data = bpy.data.cameras.new("pm_cam")
    cam = bpy.data.objects.new("pm_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    os.makedirs(out_dir, exist_ok=True)
    target = obj.location
    frames: List[Dict] = []
    for i, pos in enumerate(samples.fibonacci_sphere_poses(num_views, radius)):
        cam.location = Vector(pos)
        _look_at(cam, target)
        path = os.path.join(out_dir, f"view_{i:03d}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        frames.append(
            samples.build_frame(i, os.path.basename(path), pos, list(target))
        )
    return frames


def _look_at(cam, target) -> None:
    _require_bpy()
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def simulate_drop(obj, cfg_seconds: float, fps: int) -> List[Dict]:
    """Bake a short rigid-body drop and record the object's trajectory."""
    _require_bpy()
    scene = bpy.context.scene
    # Ground plane (passive collider).
    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0, 0, -1.0))
    ground = bpy.context.active_object
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = "PASSIVE"

    n_frames = int(cfg_seconds * fps)
    scene.frame_start = 1
    scene.frame_end = n_frames
    scene.rigidbody_world.point_cache.frame_end = n_frames
    bpy.ops.ptcache.bake_all(bake=True)

    traj: List[Dict] = []
    for f in range(1, n_frames + 1):
        scene.frame_set(f)
        loc = obj.matrix_world.translation
        traj.append({"frame": f, "position": [loc.x, loc.y, loc.z]})
    return traj


def build_scene(scene_id: str, shape_id: str, material_name: str, split: str,
                out_root: str, render_cfg, physics_cfg) -> Dict:
    """Generate one paired sample and return its manifest record."""
    _require_bpy()
    reset_scene()
    setup_world_and_light()
    obj = add_primitive_shape(shape_id)
    material = M.get(material_name)
    apply_material(obj, material)

    scene_dir = os.path.join(out_root, scene_id)
    frames = render_views(
        obj, os.path.join(scene_dir, "renders"),
        render_cfg.num_views, render_cfg.camera_radius, render_cfg.resolution,
        render_cfg.engine, render_cfg.samples,
    )
    trajectory = simulate_drop(obj, physics_cfg.sim_seconds, physics_cfg.fps)

    record = samples.build_sample_record(
        scene_id=scene_id, split=split, shape=shape_id, material_name=material_name,
        frames=frames, resolution=render_cfg.resolution, generator="blender",
        trajectory=trajectory, fps=physics_cfg.fps,
    )
    with open(os.path.join(scene_dir, "sample.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    return record


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pseudo-marble Blender data generation")
    p.add_argument("--output", default="data/pseudo_marble")
    p.add_argument("--num-scenes", type=int, default=16)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--views", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument(
        "--shapes",
        default="cube,sphere,cylinder,cone,torus,cup",
        help="comma-separated primitive shape ids (smoke test) or asset ids",
    )
    p.add_argument("--engine", default="CYCLES")
    p.add_argument("--samples", type=int, default=64)
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    _require_bpy()
    from pseudomarble.config import PhysicsConfig, RenderConfig
    from pseudomarble.splits import assign_scene_materials, make_combination_split

    args = parse_args(argv)
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    mat_names = list(M.names())

    split = make_combination_split(shapes, mat_names, args.holdout_frac, args.seed)
    assignments = assign_scene_materials(
        shapes, mat_names, split, args.num_scenes, args.seed
    )

    render_cfg = RenderConfig(
        resolution=args.resolution, num_views=args.views,
        engine=args.engine, samples=args.samples,
    )
    physics_cfg = PhysicsConfig()

    os.makedirs(args.output, exist_ok=True)
    scenes: List[Dict] = []
    for rec in assignments:
        out = build_scene(
            rec["scene_id"], rec["shape"], rec["material"], rec["split"],
            args.output, render_cfg, physics_cfg,
        )
        scenes.append(out)
        print(f"[pseudo-marble:blender] built {rec['scene_id']} "
              f"({rec['shape']} / {rec['material']} / {rec['split']})")

    manifest = samples.build_manifest(
        "blender", split.summary(), split.test_pairs, scenes
    )
    with open(os.path.join(args.output, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[pseudo-marble:blender] wrote {len(scenes)} scenes -> {args.output}")


if __name__ == "__main__":
    # Strip Blender's own args; everything after "--" is ours.
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
