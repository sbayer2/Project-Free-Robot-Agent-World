"""Tests for the held-out material x shape combination split (the honest test)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.splits import (  # noqa: E402
    assign_scene_materials,
    make_combination_split,
)

SHAPES = ["cube", "sphere", "cylinder", "cone", "torus", "cup"]
MATS = ["ceramic", "rubber", "steel", "wood", "foam", "glass", "plastic", "stone"]


def test_split_is_deterministic():
    a = make_combination_split(SHAPES, MATS, 0.2, seed=7)
    b = make_combination_split(SHAPES, MATS, 0.2, seed=7)
    assert a == b


def test_train_and_test_are_disjoint():
    s = make_combination_split(SHAPES, MATS, 0.25, seed=3)
    assert set(s.train_pairs).isdisjoint(set(s.test_pairs))


def test_every_shape_and_material_appears_in_train():
    s = make_combination_split(SHAPES, MATS, 0.3, seed=1)
    train_shapes = {sh for sh, _ in s.train_pairs}
    train_mats = {m for _, m in s.train_pairs}
    assert train_shapes == set(SHAPES), "a shape vanished from train"
    assert train_mats == set(MATS), "a material vanished from train"


def test_holdout_pairs_are_novel_combinations():
    # Held-out pairs must not appear in train, but their shapes/materials must.
    s = make_combination_split(SHAPES, MATS, 0.2, seed=11)
    for shape, mat in s.test_pairs:
        assert (shape, mat) not in set(s.train_pairs)
        assert any(shape == sh for sh, _ in s.train_pairs)
        assert any(mat == m for _, m in s.train_pairs)


def test_holdout_fraction_is_approximately_respected():
    s = make_combination_split(SHAPES, MATS, 0.2, seed=5)
    total = len(SHAPES) * len(MATS)
    # Coverage constraints can only reduce the holdout, never inflate it.
    assert 0 < len(s.test_pairs) <= round(total * 0.2)


def test_zero_holdout_keeps_everything_in_train():
    s = make_combination_split(SHAPES, MATS, 0.0, seed=0)
    assert len(s.test_pairs) == 0
    assert len(s.train_pairs) == len(SHAPES) * len(MATS)


def test_scene_assignment_covers_both_splits():
    s = make_combination_split(SHAPES, MATS, 0.25, seed=2)
    recs = assign_scene_materials(SHAPES, MATS, s, num_scenes=200, seed=2)
    assert len(recs) == 200
    splits = {r["split"] for r in recs}
    assert splits == {"train", "test"}
    # Every assigned pair must be drawn from the correct partition.
    train_set, test_set = set(s.train_pairs), set(s.test_pairs)
    for r in recs:
        pair = (r["shape"], r["material"])
        assert pair in (train_set if r["split"] == "train" else test_set)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
