"""Tests for the shared sample.json contract (generator-agnostic schema)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.data import samples  # noqa: E402


def test_fibonacci_poses_count_and_radius():
    poses = samples.fibonacci_sphere_poses(16, 2.6)
    assert len(poses) == 16
    # Every camera sits ~on the requested sphere.
    for x, y, z in poses:
        r = (x * x + y * y + z * z) ** 0.5
        assert abs(r - 2.6) < 1e-6


def test_pose_to_az_el_dist_roundtrips_distance():
    for pos in samples.fibonacci_sphere_poses(8, 3.0):
        _, _, dist = samples.pose_to_az_el_dist(pos)
        assert abs(dist - 3.0) < 1e-6


def test_physics_labels_normalized_in_range():
    labels = samples.physics_labels("steel")
    assert labels["raw"]["density"] == 7850
    for v in labels["normalized"].values():
        assert 0.0 <= v <= 1.0
    assert "metal" in labels["tags"]


def test_sample_record_has_full_schema():
    frames = [samples.build_frame(0, "view_000.png", [1, 0, 0], [0, 0, 0])]
    rec = samples.build_sample_record(
        scene_id="train_000000", split="train", shape="sphere",
        material_name="rubber", frames=frames, resolution=128,
        generator="mujoco", trajectory=[{"frame": 1, "position": [0, 0, 1]}], fps=60,
    )
    assert rec["generator"] == "mujoco"
    assert rec["input"] == {"shape": "sphere", "material": "rubber"}
    assert rec["appearance"]["resolution"] == 128
    assert rec["appearance"]["frames"][0]["file"] == "view_000.png"
    assert set(rec["physics"]["raw"]) == {"density", "friction", "restitution"}
    assert rec["dynamics"]["fps"] == 60


def test_both_generators_produce_identical_schema_keys():
    # The contract is the point: blender and mujoco records are structurally equal.
    frames = [samples.build_frame(0, "v.png", [1, 0, 0], [0, 0, 0])]
    common = dict(
        scene_id="s", split="train", shape="box", material_name="wood",
        frames=frames, resolution=64, trajectory=[], fps=30,
    )
    blender = samples.build_sample_record(generator="blender", **common)
    mujoco = samples.build_sample_record(generator="mujoco", **common)

    def shape_of(d):
        if isinstance(d, dict):
            return {k: shape_of(v) for k, v in sorted(d.items())}
        if isinstance(d, list):
            return [shape_of(x) for x in d]
        return type(d).__name__

    bshape, mshape = shape_of(blender), shape_of(mujoco)
    bshape["generator"] = mshape["generator"] = "str"  # only this value differs
    assert bshape == mshape


def test_manifest_summarizes_scenes():
    frames = [samples.build_frame(0, "v.png", [1, 0, 0], [0, 0, 0])]
    scenes = [
        samples.build_sample_record(
            scene_id=f"train_{i:06d}", split="train", shape="box",
            material_name="wood", frames=frames, resolution=64, generator="mujoco",
        )
        for i in range(3)
    ]
    man = samples.build_manifest("mujoco", {"n_train": 3}, [["box", "ice"]], scenes)
    assert man["generator"] == "mujoco"
    assert len(man["scenes"]) == 3
    assert man["held_out_combinations"] == [["box", "ice"]]
    assert man["scenes"][0]["material"] == "wood"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
