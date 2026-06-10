"""Tests for the GSO generator's pure-Python core: object discovery, mass/category
parsing, the thinner measured-physics block, and split planning. No trimesh/mujoco.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.data import generate_gso as G  # noqa: E402
from pseudomarble.data import samples  # noqa: E402


def _make_gso_tree(root):
    # ObjectA: mass in model.sdf, category in metadata.pbtxt, a texture
    a = os.path.join(root, "ObjectA")
    os.makedirs(os.path.join(a, "meshes"))
    os.makedirs(os.path.join(a, "materials", "textures"))
    open(os.path.join(a, "meshes", "model.obj"), "w").write("o\n")
    open(os.path.join(a, "model.sdf"), "w").write(
        "<sdf><model><link><inertial><mass>0.25</mass></inertial>"
        "</link></model></sdf>")
    open(os.path.join(a, "metadata.pbtxt"), "w").write('categories: "Mug"\n')
    open(os.path.join(a, "materials", "textures", "texture.png"), "w").write("x")
    # ObjectB: mass in meta.json (grams), category in json, mesh at top level
    b = os.path.join(root, "ObjectB")
    os.makedirs(b)
    open(os.path.join(b, "model.obj"), "w").write("o\n")
    json.dump({"mass_grams": 500, "category": "shoe"},
              open(os.path.join(b, "meta.json"), "w"))
    # ObjectC: no mesh -> skipped
    os.makedirs(os.path.join(root, "ObjectC"))


def test_discovery_and_metadata_parsing():
    with tempfile.TemporaryDirectory() as root:
        _make_gso_tree(root)
        objs = G.discover_objects(root)
        ids = {o.object_id: o for o in objs}
        assert set(ids) == {"ObjectA", "ObjectB"}  # C skipped (no mesh)

        a = ids["ObjectA"]
        assert abs(a.mass_kg - 0.25) < 1e-9 and a.mass_source == "sdf"
        assert a.category == "mug"
        assert a.texture_path and a.texture_path.endswith("texture.png")
        assert a.mesh_path.endswith(os.path.join("meshes", "model.obj"))

        b = ids["ObjectB"]
        assert abs(b.mass_kg - 0.5) < 1e-9 and b.mass_source == "json"  # 500 g
        assert b.category == "shoe"


def test_unknown_mass_and_category():
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "Plain")
        os.makedirs(d)
        open(os.path.join(d, "model.obj"), "w").write("o\n")
        o = G.discover_objects(root)[0]
        assert o.mass_kg is None and o.mass_source == "unknown"
        assert o.category == "unknown"


def test_measured_physics_block_is_honest():
    block = samples.physics_labels_measured(mass_kg=0.25, density=900.0,
                                            mass_source="sdf")
    assert block["mass_kg"] == 0.25
    assert block["provenance"]["mass"] == "sdf"
    assert block["provenance"]["density"] == "derived"
    # friction/restitution are priors, flagged as assumed
    assert block["provenance"]["friction"] == "assumed"
    assert block["provenance"]["restitution"] == "assumed"
    # still loader-compatible (normalized present, in range)
    for v in block["normalized"].values():
        assert 0.0 <= v <= 1.0


def test_record_assembles_through_shared_contract():
    block = samples.physics_labels_measured(mass_kg=0.5, density=None)
    frames = [samples.build_frame(0, "view_000.png", [1, 0, 0], [0, 0, 0])]
    rec = samples.build_sample_record(
        scene_id="train_000000", split="train", shape="mesh", frames=frames,
        resolution=128, generator="gso", material_id="ObjectB",
        physics_block=block, behavior=[], material_truth_block={"object_id": "ObjectB"})
    assert rec["generator"] == "gso"
    assert rec["input"] == {"shape": "mesh", "material": "ObjectB"}
    assert rec["physics"]["mass_kg"] == 0.5
    assert rec["material_truth"]["object_id"] == "ObjectB"
    json.dumps(rec)  # serializable


def test_to_mesh_asset_carries_measured_mass():
    obj = G.GsoObject(object_id="My Mug!", directory="/x", mesh_path="/x/m.obj",
                      category="mug", mass_kg=0.25, mass_source="sdf",
                      texture_path="/x/t.png")
    asset = G.to_mesh_asset(obj, half_height=0.08, scale=1.0, friction=0.5)
    assert abs(asset.mass - 0.25) < 1e-9
    assert asset.texture_path == "/x/t.png"
    assert asset.name.replace("_", "").isalnum()  # sanitized for MJCF


def test_plan_split_uses_categories_when_known():
    objs = [
        G.GsoObject("a", "", "", "mug", 0.2, "sdf"),
        G.GsoObject("b", "", "", "mug", 0.2, "sdf"),
        G.GsoObject("c", "", "", "shoe", 0.5, "sdf"),
        G.GsoObject("d", "", "", "bottle", 0.3, "sdf"),
    ]
    sp = G.plan_split(objs, holdout_categories="shoe", holdout_frac=0.2, seed=0)
    assert set(sp.test_ids) == {"c"}
    assert "shoe" in sp.holdout_categories
    assert set(sp.train_ids).isdisjoint(sp.test_ids)


def test_build_collision_convex_hull_branch_needs_no_backend():
    # The convex_hull short-circuit must not import trimesh/coacd and must flag
    # that concavity is NOT preserved (a real cavity would be lost).
    parts, meta = G.build_collision("any.obj", "/tmp/scene", "convex_hull", 16)
    assert parts == ()
    assert meta["method"] == "convex_hull"
    assert meta["concavity_preserved"] is False


def test_plan_split_falls_back_to_objects_when_categories_unknown():
    objs = [G.GsoObject(f"o{i}", "", "", "unknown", None, "unknown") for i in range(10)]
    sp = G.plan_split(objs, holdout_categories="", holdout_frac=0.2, seed=1)
    assert len(sp.train_ids) + len(sp.test_ids) == 10
    assert sp.holdout_categories == ()  # object-level fallback, no categories held


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
