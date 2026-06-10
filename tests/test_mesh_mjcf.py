"""Tests for the mesh path of build_mjcf (GSO real objects). Pure-Python: the
MJCF is a string, validated by XML parsing — no trimesh/mujoco runtime needed.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble import materials as M  # noqa: E402
from pseudomarble.data.generate_mujoco import (  # noqa: E402
    MeshAsset,
    build_mesh_mjcf,
    build_mjcf,
)


def _object_geoms(xml):
    root = ET.fromstring(xml)
    body = next(b for b in root.iter("body") if b.get("name") == "object")
    return root, list(body.iter("geom"))


def test_single_mesh_geom_carries_measured_mass():
    asset = MeshAsset(name="mug", visual_path="meshes/model.obj", mass=0.25,
                      half_height=0.08)
    xml = build_mjcf(mesh=asset)
    ET.fromstring(xml)  # well-formed
    root, geoms = _object_geoms(xml)
    obj = next(g for g in geoms if g.get("name") == "obj")
    assert obj.get("type") == "mesh"
    assert obj.get("mesh") == "mug_vis"
    assert abs(float(obj.get("mass")) - 0.25) < 1e-9
    # the visual mesh asset is declared
    assert any(m.get("name") == "mug_vis" for m in root.iter("mesh"))


def test_texture_produces_material_reference():
    asset = MeshAsset(name="shoe", visual_path="m.obj", mass=0.5,
                      texture_path="materials/textures/texture.png")
    root = ET.fromstring(build_mjcf(mesh=asset))
    assert any(t.get("name") == "shoe_tex" for t in root.iter("texture"))
    obj = next(g for g in root.iter("geom") if g.get("name") == "obj")
    assert obj.get("material") == "shoe_mat"


def test_collision_parts_split_mass_and_disable_visual_collision():
    asset = MeshAsset(name="cup", visual_path="m.obj", mass=0.30,
                      collision_paths=("c0.obj", "c1.obj", "c2.obj"))
    root, geoms = _object_geoms(build_mjcf(mesh=asset))
    vis = next(g for g in geoms if g.get("name") == "vis")
    assert vis.get("contype") == "0" and vis.get("conaffinity") == "0"  # render-only
    # render-only geom must be mass-less, or MuJoCo adds phantom default-density
    # mass on top of the measured mass (a real bug caught by the sandbox run).
    assert float(vis.get("mass")) == 0.0
    cols = [g for g in geoms if g.get("name", "").startswith("col")]
    assert len(cols) == 3
    # mass split evenly across parts -> sums to exactly the measured mass
    total = sum(float(g.get("mass")) for g in cols)
    assert abs(total - 0.30) < 1e-9


def test_primitive_path_unchanged():
    xml = build_mjcf("box", M.get("steel"))
    root, geoms = _object_geoms(xml)
    assert next(g for g in geoms if g.get("name") == "obj").get("type") == "box"


def test_requires_shape_or_mesh():
    try:
        build_mjcf()  # neither primitive args nor mesh
    except ValueError as exc:
        assert "mesh" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_tilt_ground_euler_applies_in_mesh_mode():
    asset = MeshAsset(name="x", visual_path="m.obj", mass=1.0)
    root = ET.fromstring(build_mesh_mjcf(asset, ground_euler=(0.3, 0.0, 0.0)))
    ground = next(g for g in root.iter("geom") if g.get("name") == "ground")
    assert abs(float(ground.get("euler").split()[0]) - 0.3) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
