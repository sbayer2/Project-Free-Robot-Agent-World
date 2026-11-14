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


def test_above_floor_layout_puts_every_camera_above_the_floor():
    """Regression pin for the camera bug: the legacy full-sphere layout put about
    3 of 8 MuJoCo views under the floor. Every 'above' camera must sit above the
    floor (z > 0), inside the elevation band, at the requested distance."""
    lookat = (0.0, 0.0, 0.1)
    for n in (1, 4, 8, 16):
        angles = samples.above_floor_camera_angles(n, 15.0, 65.0)
        assert len(angles) == n
        for az, elev in angles:
            assert 0.0 <= az < 360.0
            assert 15.0 - 1e-9 <= elev <= 65.0 + 1e-9
            x, y, z = samples.camera_world_position(lookat, az, elev, 2.6)
            assert z > 0.0, f"camera below the floor at elevation {elev}"
            d = ((x - lookat[0]) ** 2 + (y - lookat[1]) ** 2 + (z - lookat[2]) ** 2) ** 0.5
            assert abs(d - 2.6) < 1e-9


def test_above_floor_layout_spreads_views_around_the_object():
    angles = samples.above_floor_camera_angles(8)
    azimuths = sorted(az for az, _ in angles)
    # No two views stacked on the same side: every gap under 120 degrees.
    gaps = [b - a for a, b in zip(azimuths, azimuths[1:], strict=False)]
    gaps.append(360.0 - azimuths[-1] + azimuths[0])
    assert max(gaps) < 120.0
    assert len({round(az, 6) for az in azimuths}) == 8
    # Deterministic: the same call gives the same cameras.
    assert angles == samples.above_floor_camera_angles(8)


def test_above_floor_layout_rejects_bands_that_reach_the_floor():
    for lo, hi in ((0.0, 60.0), (-10.0, 60.0), (20.0, 90.0), (50.0, 40.0)):
        try:
            samples.above_floor_camera_angles(8, lo, hi)
        except ValueError:
            continue
        raise AssertionError(f"band ({lo}, {hi}) should be rejected")


def test_camera_world_position_matches_mujoco_free_camera_convention():
    """MuJoCo's free camera sits at lookat - distance * forward, with
    forward = (cos e cos a, cos e sin a, sin e) and e = -elevation_up."""
    import math
    lookat = (0.2, -0.1, 0.3)
    for az, up in ((0.0, 30.0), (137.5, 15.0), (275.0, 65.0)):
        a, e = math.radians(az), math.radians(-up)
        fwd = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
        want = tuple(c - 2.6 * f for c, f in zip(lookat, fwd, strict=True))
        got = samples.camera_world_position(lookat, az, up, 2.6)
        assert all(abs(g - w) < 1e-12 for g, w in zip(got, want, strict=True))


def test_legacy_sphere_layout_is_unchanged():
    """Older worlds must regenerate byte-identically: the legacy layout's first
    and last poses are pinned."""
    poses = samples.fibonacci_sphere_poses(8, 2.6)
    assert abs(poses[0][0]) < 1e-12 and abs(poses[0][1] - 2.6) < 1e-12 and abs(poses[0][2]) < 1e-12
    assert abs(poses[-1][1] + 2.6) < 1e-12
    # The legacy view #2 (the one rendered from under the floor in pm_big) is pinned too.
    assert [round(c, 2) for c in poses[2]] == [0.21, 1.11, -2.34]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
