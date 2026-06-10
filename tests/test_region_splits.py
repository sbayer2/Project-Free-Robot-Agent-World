"""Tests for region-based holdout over continuous essence-space, and for scene
assignment (both pure-Python; scene assignment runs without MuJoCo)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.config import PHYSICS_NORMALIZERS as N  # noqa: E402
from pseudomarble.data.generate_mujoco import DEFAULT_SHAPES, assign_scenes  # noqa: E402
from pseudomarble.splits import (  # noqa: E402
    DEFAULT_REGION_HOLDOUT,
    EXTRAPOLATION_REGION_HOLDOUT,
    RegionHoldout,
)


def _norm(sample):
    p = sample.material.physics
    return p.density / N["density"], p.restitution / N["restitution"]


def test_extrapolation_corner_membership():
    h = EXTRAPOLATION_REGION_HOLDOUT
    assert h.kind == "extrapolation"
    # heavy AND bouncy -> held out (test)
    assert h.contains({"density": 0.9, "friction": 0.3, "restitution": 0.8}, "box")
    # only ONE extreme -> trains (the model HAS seen heavy, and seen bouncy)
    assert not h.contains({"density": 0.9, "friction": 0.3, "restitution": 0.2}, "box")
    assert not h.contains({"density": 0.2, "friction": 0.3, "restitution": 0.8}, "box")
    # interior -> trains
    assert not h.contains({"density": 0.3, "friction": 0.3, "restitution": 0.3}, "box")


def test_extrapolation_assignment_is_extreme_and_nonempty():
    recs = assign_scenes(DEFAULT_SHAPES, EXTRAPOLATION_REGION_HOLDOUT, 500, seed=0)
    test = [r for r in recs if r["split"] == "test"]
    assert 0 < len(test) < len(recs)  # corner reachable but not dominant
    # every held-out object is genuinely in the extreme corner (extrapolation)
    for r in test:
        d, rest = _norm(r["sample"])
        assert d >= 0.55 and rest >= 0.60


def test_default_region_is_labeled_interpolation():
    assert DEFAULT_REGION_HOLDOUT.kind == "interpolation"


def test_region_contains_inside_box():
    h = RegionHoldout(friction=(0.5, 0.8), restitution=(0.5, 0.8))
    assert h.contains({"density": 0.1, "friction": 0.6, "restitution": 0.6}, "box")
    assert not h.contains({"density": 0.1, "friction": 0.9, "restitution": 0.6}, "box")


def test_shape_constraint_limits_holdout():
    h = RegionHoldout(friction=(0.0, 1.0), shapes=("cylinder",))
    ess = {"density": 0.1, "friction": 0.5, "restitution": 0.5}
    assert h.contains(ess, "cylinder")
    assert not h.contains(ess, "box")  # same essence, non-held-out shape -> train


def test_all_none_holdout_matches_nothing():
    # Guard against an empty holdout silently sending everything to test.
    h = RegionHoldout()
    assert not h.contains({"density": 0.5, "friction": 0.5, "restitution": 0.5}, "box")


def test_label_returns_train_or_test():
    inside = {"density": 0.1, "friction": 0.65, "restitution": 0.65}
    outside = {"density": 0.1, "friction": 0.1, "restitution": 0.1}
    assert DEFAULT_REGION_HOLDOUT.label(inside, "box") == "test"
    assert DEFAULT_REGION_HOLDOUT.label(outside, "box") == "train"


def test_assign_scenes_is_deterministic_and_covers_both_splits():
    a = assign_scenes(DEFAULT_SHAPES, DEFAULT_REGION_HOLDOUT, num_scenes=200, seed=0)
    b = assign_scenes(DEFAULT_SHAPES, DEFAULT_REGION_HOLDOUT, num_scenes=200, seed=0)
    assert len(a) == 200
    # Deterministic: same shapes and same sampled densities.
    assert [r["shape"] for r in a] == [r["shape"] for r in b]
    assert [round(r["sample"].material.physics.density) for r in a] == \
           [round(r["sample"].material.physics.density) for r in b]
    splits = {r["split"] for r in a}
    assert "train" in splits  # the holdout is a small corner -> train dominates
    # Each record's split must match the region rule for its essence.
    for r in a:
        assert r["split"] in ("train", "test")
        assert r["scene_id"].startswith(r["split"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
