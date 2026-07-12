"""GSO (Google Scanned Objects) data generator — the "real eigenvector" endpoint.

This is the bridge from the synthetic world (where the appearance↔physics coupling
is *authored* by ``MaterialSampler``) to **real measured objects**, so a positive
result can be about reality rather than about the generator (see
docs/GSO_EXPERIMENT.md).

What it does, honestly:
  * loads scanned meshes + their **measured mass** (GSO ships mass in model.sdf),
  * renders multi-view appearance from the real scanned texture,
  * runs the same drop/tilt/push probes on the mesh — exercising the concave path
    (``mesh_validate`` watertightness gate + ``collision.py`` CoACD/V-HACD
    decomposition) so a real cavity (mug, bowl) survives instead of collapsing to
    a convex hull,
  * writes the **same** ``sample.json`` contract, but with a *thinner, honest*
    physics block: mass is measured, density is derived, **friction/restitution
    are assumed** (GSO does not measure them) — recorded in ``physics.provenance``.

Two honest caveats baked in:
  1. The only real physics signal is **mass** (and mass-driven behavior); friction
     and restitution are priors, so analysis should mask them.
  2. Materials are baked per object — you cannot recombine shape×material — so the
     generalization split holds out whole **categories** (``splits.make_category_holdout``),
     not synthetic combinations.

Pure-Python parts (object discovery, mass/category parsing, record assembly) are
unit-tested without any heavy deps; mesh loading (trimesh) and rendering/simulation
(mujoco) are imported lazily.

Run on a machine with the meshes + deps::

    pip install -e ".[gso]"   # trimesh + mujoco + image IO
    python -m pseudomarble.data.generate_gso \
        --gso-root /path/to/google_scanned_objects \
        --output data/pm_gso --num-objects 200 --views 16 --resolution 128
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from pseudomarble import probes as P
from pseudomarble.config import PhysicsConfig, RenderConfig
from pseudomarble.data import samples
from pseudomarble.data.parallel import (
    default_cpu_workers,
    ordered_parallel_map,
    resolve_workers,
)
from pseudomarble.splits import make_category_holdout, make_object_holdout

MESH_EXTS = (".obj", ".glb", ".gltf", ".stl", ".ply")
TEXTURE_HINTS = ("texture", "albedo", "basecolor", "diffuse")


@dataclass(frozen=True)
class GsoObject:
    object_id: str
    directory: str
    mesh_path: str
    category: str
    mass_kg: Optional[float]
    mass_source: str                 # "sdf" | "json" | "pbtxt" | "estimated" | "unknown"
    texture_path: Optional[str] = None


# --------------------------------------------------------------------------- #
# Metadata parsing (pure-Python, unit-tested).
# --------------------------------------------------------------------------- #
def read_mass(obj_dir: str) -> Tuple[Optional[float], str]:
    """Return ``(mass_kg, source)``. GSO ships mass in model.sdf's <inertial>;
    fall back to JSON/pbtxt metadata, else unknown."""
    sdf = os.path.join(obj_dir, "model.sdf")
    if os.path.exists(sdf):
        # Regex, not an XML parser: SDF files come from an external dataset, and a
        # full XML parser (xml.etree) is exposed to entity-expansion attacks
        # (bandit B314). A targeted match for <mass> avoids that surface entirely.
        try:
            m = re.search(r"<mass>\s*([0-9.eE+-]+)\s*</mass>", open(sdf).read())
            if m:
                return float(m.group(1)), "sdf"
        except Exception:
            pass
    for name in ("meta.json", "metadata.json", "model.json"):
        p = os.path.join(obj_dir, name)
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                for key, scale in (("mass_kg", 1.0), ("mass", 1.0), ("mass_grams", 1e-3)):
                    if key in d:
                        return float(d[key]) * scale, "json"
            except Exception:
                pass
    pb = os.path.join(obj_dir, "metadata.pbtxt")
    if os.path.exists(pb):
        try:
            m = re.search(r"mass\s*:\s*([0-9.eE+-]+)", open(pb).read())
            if m:
                return float(m.group(1)), "pbtxt"
        except Exception:
            pass
    return None, "unknown"


def read_category(obj_dir: str) -> str:
    """Best-effort object category from GSO metadata; 'unknown' if not found."""
    pb = os.path.join(obj_dir, "metadata.pbtxt")
    if os.path.exists(pb):
        try:
            txt = open(pb).read()
            m = re.search(r"(?:categor(?:y|ies)|class)\s*:\s*\"?([A-Za-z0-9_ /-]+)\"?", txt)
            if m:
                return m.group(1).strip().lower()
        except Exception:
            pass
    for name in ("meta.json", "metadata.json", "model.json"):
        p = os.path.join(obj_dir, name)
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                for key in ("category", "class", "categories"):
                    if key in d:
                        v = d[key]
                        return (v[0] if isinstance(v, list) and v else v).strip().lower()
            except Exception:
                pass
    return "unknown"


def _find_first(obj_dir: str, exts: Tuple[str, ...]) -> Optional[str]:
    # Prefer a conventional meshes/model.obj, else first matching file by walk.
    preferred = os.path.join(obj_dir, "meshes", "model.obj")
    if os.path.exists(preferred):
        return preferred
    for dirpath, _, files in os.walk(obj_dir):
        for f in sorted(files):
            if f.lower().endswith(exts):
                return os.path.join(dirpath, f)
    return None


def _find_texture(obj_dir: str) -> Optional[str]:
    for dirpath, _, files in os.walk(obj_dir):
        for f in sorted(files):
            low = f.lower()
            if low.endswith((".png", ".jpg", ".jpeg")) and any(h in low for h in TEXTURE_HINTS):
                return os.path.join(dirpath, f)
    # fall back to any png/jpg under a materials/textures dir
    for dirpath, _, files in os.walk(obj_dir):
        if "texture" in dirpath.lower():
            for f in sorted(files):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    return os.path.join(dirpath, f)
    return None


def discover_objects(root: str, max_objects: Optional[int] = None) -> List[GsoObject]:
    """Scan a GSO root for object directories (each containing a mesh)."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"GSO root not found: {root!r}")
    objects: List[GsoObject] = []
    for entry in sorted(os.listdir(root)):
        obj_dir = os.path.join(root, entry)
        if not os.path.isdir(obj_dir):
            continue
        mesh = _find_first(obj_dir, MESH_EXTS)
        if mesh is None:
            continue
        mass, source = read_mass(obj_dir)
        objects.append(GsoObject(
            object_id=entry, directory=obj_dir, mesh_path=mesh,
            category=read_category(obj_dir), mass_kg=mass, mass_source=source,
            texture_path=_find_texture(obj_dir),
        ))
        if max_objects is not None and len(objects) >= max_objects:
            break
    return objects


def gso_truth_block(obj: GsoObject) -> Dict:
    """Analysis-only ground truth (never seen by the model): category + mass provenance."""
    return {
        "object_id": obj.object_id,
        "category": obj.category,
        "mass_kg": obj.mass_kg,
        "mass_source": obj.mass_source,
        "note": "friction/restitution are assumed (GSO does not measure them)",
    }


# --------------------------------------------------------------------------- #
# Runtime: mesh bounds, rendering, probes (require trimesh / mujoco).
# --------------------------------------------------------------------------- #
def mesh_halfheight_and_density(mesh_path: str, mass_kg: Optional[float],
                                scale: float) -> Tuple[float, Optional[float]]:
    """Return (z half-extent after scaling, derived density or None). Needs trimesh."""
    import trimesh  # type: ignore

    mesh = trimesh.load(mesh_path, force="mesh")
    half_h = float(mesh.extents[2]) * scale / 2.0
    density = None
    if mass_kg is not None and mesh.is_watertight:
        vol = float(abs(mesh.volume)) * (scale ** 3)
        if vol > 1e-9:
            density = mass_kg / vol
    return max(half_h, 1e-3), density


def to_mesh_asset(obj: GsoObject, half_height: float, scale: float, friction: float,
                  collision_paths: Tuple[str, ...] = ()) -> "object":
    from pseudomarble.data.generate_mujoco import MeshAsset
    tex = os.path.abspath(obj.texture_path) if obj.texture_path else None
    return MeshAsset(
        name=re.sub(r"[^A-Za-z0-9_]", "_", obj.object_id)[:40] or "obj",
        visual_path=os.path.abspath(obj.mesh_path),  # absolute so MuJoCo finds it
        mass=obj.mass_kg if obj.mass_kg is not None else 1.0,
        half_height=half_height, scale=scale, friction=friction,
        collision_paths=collision_paths, texture_path=tex,
    )


def watertight(mesh_path: str) -> Optional[bool]:
    """Watertightness via mesh_validate (mass = density x volume needs it).
    Returns None if trimesh is unavailable. Needs trimesh at runtime."""
    from pseudomarble.data.mesh_validate import validate_mesh
    return validate_mesh(mesh_path).watertight


def build_collision(mesh_path: str, scene_dir: str, method: str,
                    max_hulls: int) -> Tuple[Tuple[str, ...], Dict]:
    """Exercise the concave path: decompose the mesh into convex parts so a real
    cavity (a mug, a bowl) survives, instead of collapsing to a convex hull.

    ``method='convex_hull'`` short-circuits to MuJoCo's single-geom hull collision
    (no decomposition, pure-Python — testable). Otherwise ``collision.py`` runs
    CoACD/V-HACD and writes convex part meshes; it warns + degrades to a hull if no
    backend is installed. Returns ``(part_paths, metadata)``.
    """
    if method == "convex_hull":
        return (), {"method": "convex_hull", "n_parts": 1,
                    "concavity_preserved": False,
                    "note": "single mesh geom -> MuJoCo convex-hull collision"}
    from pseudomarble.data.collision import compute_collision
    out = os.path.abspath(os.path.join(scene_dir, "collision"))
    res = compute_collision(mesh_path, out, method=method, max_hulls=max_hulls)
    return tuple(os.path.abspath(p) for p in res.part_paths), {
        "method": res.method, "n_parts": res.n_parts,
        "concavity_preserved": (not res.degraded) and res.n_parts > 1,
        "degraded": res.degraded, "note": res.note,
    }


def auto_camera_radius(half_height: float, configured: float) -> float:
    """Camera orbit radius: the configured value when positive, else adaptive —
    4x the object's z half-extent (floored) so metre-scale objects (ABO
    furniture) frame as well as centimetre-scale scans. Pure (unit-tested)."""
    if configured > 0:
        return configured
    return 4.0 * max(half_height, 0.05)


def render_views_mesh(mesh_asset, renders_dir: str, render_cfg: RenderConfig) -> List[Dict]:
    """Static multi-view renders of the scanned object resting on the ground."""
    import mujoco  # type: ignore

    from pseudomarble.data.generate_mujoco import _save_png, build_mjcf

    os.makedirs(renders_dir, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(build_mjcf(mesh=mesh_asset))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frames: List[Dict] = []
    renderer = mujoco.Renderer(model, render_cfg.resolution, render_cfg.resolution)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, mesh_asset.half_height]
    radius = auto_camera_radius(mesh_asset.half_height, render_cfg.camera_radius)
    for i, pos in enumerate(samples.fibonacci_sphere_poses(
            render_cfg.num_views, radius)):
        az, el, dist = samples.pose_to_az_el_dist(pos)
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        renderer.update_scene(data, camera=cam)
        _save_png(os.path.join(renders_dir, f"view_{i:03d}.png"), renderer.render())
        frames.append(samples.build_frame(i, f"view_{i:03d}.png", pos, list(cam.lookat)))
    return frames


def run_probes_mesh(mesh_asset, physics_cfg: PhysicsConfig,
                    keep_trajectory: bool = False) -> List[Dict]:
    """Run the drop/tilt/push battery on the mesh (mirrors generate_mujoco.run_probes)."""
    import math

    import mujoco  # type: ignore

    from pseudomarble.data.generate_mujoco import _simulate, build_mjcf

    records: List[Dict] = []
    for spec in P.default_probes():
        if spec.kind == "drop":
            z = mesh_asset.half_height + spec.height
            model = mujoco.MjModel.from_xml_string(build_mjcf(mesh=mesh_asset, object_z=z))
            traj = _simulate(model, mujoco.MjData(model),
                             physics_cfg.sim_seconds, physics_cfg.fps)
        elif spec.kind == "tilt":
            z = mesh_asset.half_height + 0.05
            model = mujoco.MjModel.from_xml_string(
                build_mjcf(mesh=mesh_asset, object_z=z,
                           ground_euler=(math.radians(spec.angle_deg), 0.0, 0.0)))
            traj = _simulate(model, mujoco.MjData(model),
                             physics_cfg.sim_seconds, physics_cfg.fps)
        else:  # push
            model = mujoco.MjModel.from_xml_string(build_mjcf(mesh=mesh_asset))
            data = mujoco.MjData(model)
            settle, window = 0.4, 0.1
            force = spec.impulse / window
            rz = (spec.height_frac - 0.5) * 2.0 * mesh_asset.half_height
            az = math.radians(spec.azimuth_deg)
            fx, fy = force * math.cos(az), force * math.sin(az)
            wrench = [fx, fy, 0.0, -rz * fy, rz * fx, 0.0]
            body = model.body("object").id

            def force_fn(t, w=wrench, b=body):
                return w if settle <= t < settle + window else [0.0] * 6

            traj = _simulate(model, data, physics_cfg.sim_seconds, physics_cfg.fps, force_fn)
        outcome = P.summarize(traj)
        records.append(samples.build_probe_record(
            P.spec_to_dict(spec), outcome.to_dict(),
            trajectory=traj if keep_trajectory else None))
    return records


def build_scene_gso(scene_id: str, obj: GsoObject, split: str, out_root: str,
                    render_cfg: RenderConfig, physics_cfg: PhysicsConfig,
                    scale: float, friction: float, keep_trajectory: bool = False) -> Dict:
    """Generate one GSO scene (appearance + behavior + measured physics) -> sample.json."""
    scene_dir = os.path.join(out_root, scene_id)
    is_watertight = watertight(obj.mesh_path)            # mesh_validate gate
    half_h, density = mesh_halfheight_and_density(obj.mesh_path, obj.mass_kg, scale)
    collision_paths, collision_meta = build_collision(   # concave path (collision.py)
        obj.mesh_path, scene_dir, physics_cfg.collision_method,
        physics_cfg.collision_max_hulls)
    mesh_asset = to_mesh_asset(obj, half_h, scale, friction, collision_paths)

    frames = render_views_mesh(mesh_asset, os.path.join(scene_dir, "renders"), render_cfg)
    behavior = run_probes_mesh(mesh_asset, physics_cfg, keep_trajectory)
    physics = samples.physics_labels_measured(
        mass_kg=mesh_asset.mass, density=density,
        mass_source=obj.mass_source if obj.mass_kg is not None else "estimated")
    truth = gso_truth_block(obj)
    truth["watertight"] = is_watertight          # density/mass trustworthy only if True
    truth["collision"] = collision_meta          # method, n_parts, concavity_preserved
    record = samples.build_sample_record(
        scene_id=scene_id, split=split, shape="mesh", frames=frames,
        resolution=render_cfg.resolution, generator="gso",
        material_id=obj.object_id, physics_block=physics,
        behavior=behavior, material_truth_block=truth,
        fps=physics_cfg.fps)
    os.makedirs(scene_dir, exist_ok=True)
    with open(os.path.join(scene_dir, "sample.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    return record


def _build_scene_gso_task(task: Tuple) -> Dict:
    """Module-level (picklable) worker: build one GSO scene from a packed task.

    Like the MuJoCo path, each scene is self-contained — its own mesh validation,
    convex decomposition, render and probe contexts, and its own output dir — so
    workers share no state and the dataset fans out across processes cleanly.
    """
    obj, scene_id, sp, out_root, render_cfg, physics_cfg, scale, friction, keep = task
    return build_scene_gso(scene_id, obj, sp, out_root, render_cfg, physics_cfg,
                           scale, friction, keep)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pseudo-marble GSO (real-object) generation")
    p.add_argument("--gso-root", required=True, help="directory of scanned-object folders")
    p.add_argument("--output", default="data/pm_gso")
    p.add_argument("--num-objects", type=int, default=None)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--views", type=int, default=16)
    p.add_argument("--camera-radius", type=float, default=2.6,
                   help="camera orbit radius in metres; <=0 = adaptive "
                        "(4x object half-height — use for metre-scale objects)")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--collision-method", default="coacd",
                   choices=["coacd", "vhacd", "convex_hull"],
                   help="concave collision: decompose (coacd/vhacd) or hull (loses cavities)")
    p.add_argument("--friction", type=float, default=samples.ASSUMED_FRICTION,
                   help="assumed friction (GSO does not measure it)")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--holdout-categories", default="",
                   help="comma-separated categories to hold out for test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-trajectory", action="store_true")
    p.add_argument("--workers", type=int, default=0,
                   help="parallel worker processes (0 = auto = a conservative CPU "
                        "default that leaves the shared GPU + memory bus headroom; "
                        "objects are independent so it scales ~linearly with cores)")
    return p.parse_args(argv)


def plan_split(objects: List[GsoObject], holdout_categories: str, holdout_frac: float,
               seed: int):
    """Choose a category split when categories are known, else a random object split."""
    pairs = [(o.object_id, o.category) for o in objects]
    known = {c for _, c in pairs if c != "unknown"}
    if holdout_categories.strip():
        cats = [c.strip().lower() for c in holdout_categories.split(",") if c.strip()]
        return make_category_holdout(pairs, holdout_categories=cats, seed=seed)
    if len(known) > 1:
        return make_category_holdout(pairs, holdout_frac=holdout_frac, seed=seed)
    return make_object_holdout([o.object_id for o in objects], holdout_frac, seed)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    objects = discover_objects(args.gso_root, args.num_objects)
    if not objects:
        raise SystemExit(f"no scanned objects with a mesh found under {args.gso_root!r}")
    split = plan_split(objects, args.holdout_categories, args.holdout_frac, args.seed)
    test_ids = set(split.test_ids)
    n_measured = sum(1 for o in objects if o.mass_kg is not None)
    print(f"[gso] {len(objects)} objects ({n_measured} with measured mass); "
          f"{len(split.train_ids)} train / {len(split.test_ids)} test "
          f"(holdout categories: {split.holdout_categories or 'random objects'})")

    render_cfg = RenderConfig(resolution=args.resolution, num_views=args.views,
                              camera_radius=args.camera_radius)
    physics_cfg = replace(PhysicsConfig(), collision_method=args.collision_method)
    os.makedirs(args.output, exist_ok=True)
    # Per-object work mixes mesh decomposition + sim (CPU-heavy) with render (GPU),
    # so it stays a single phase for now (a full render/sim/decompose split is a
    # follow-up), but the auto width is the conservative CPU default — NOT
    # os.cpu_count() — to leave the one shared GPU + memory bus headroom.
    workers = resolve_workers(args.workers, len(objects), default=default_cpu_workers())
    print(f"[gso] generating {len(objects)} scenes on {workers} worker process(es)")

    tasks = []
    metas = []  # parallel list of (scene_id, obj) for ordered progress prints
    for i, obj in enumerate(objects):
        sp = "test" if obj.object_id in test_ids else "train"
        scene_id = f"{sp}_{i:06d}"
        tasks.append((obj, scene_id, sp, args.output, render_cfg, physics_cfg,
                      args.scale, args.friction, args.keep_trajectory))
        metas.append((scene_id, obj, sp))

    def _progress(i: int, _record: Dict) -> None:
        scene_id, obj, sp = metas[i]
        print(f"[gso] built {scene_id} ({obj.object_id} / {obj.category} / {sp})")

    scenes: List[Dict] = ordered_parallel_map(
        _build_scene_gso_task, tasks, workers, on_done=_progress)

    n_test = sum(1 for s in scenes if s["split"] == "test")
    manifest = samples.build_manifest(
        "gso", {"n_train": len(scenes) - n_test, "n_test": n_test, "n_scenes": len(scenes)},
        [], scenes)
    manifest["holdout_categories"] = list(split.holdout_categories)
    with open(os.path.join(args.output, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[gso] wrote {len(scenes)} scenes ({n_test} test) -> {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
