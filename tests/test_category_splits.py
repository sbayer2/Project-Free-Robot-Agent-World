"""Tests for the real-object (GSO) generalization splits: hold out whole
categories (or random objects). Pure-Python."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.splits import (  # noqa: E402
    make_category_holdout,
    make_object_holdout,
)

OBJECTS = [
    ("o1", "mug"), ("o2", "mug"), ("o3", "shoe"),
    ("o4", "bottle"), ("o5", "bottle"), ("o6", "toy"),
]


def test_explicit_category_holdout():
    s = make_category_holdout(OBJECTS, holdout_categories=["bottle"], seed=0)
    assert set(s.test_ids) == {"o4", "o5"}
    assert set(s.train_ids) == {"o1", "o2", "o3", "o6"}
    assert set(s.train_ids).isdisjoint(s.test_ids)
    assert s.holdout_categories == ("bottle",)


def test_no_object_of_held_category_leaks_into_train():
    s = make_category_holdout(OBJECTS, holdout_frac=0.34, seed=3)
    held = set(s.holdout_categories)
    train_cats = {c for oid, c in OBJECTS if oid in set(s.train_ids)}
    assert held.isdisjoint(train_cats)  # whole categories held out
    assert len(s.train_ids) > 0


def test_split_is_deterministic():
    a = make_category_holdout(OBJECTS, holdout_frac=0.34, seed=7)
    b = make_category_holdout(OBJECTS, holdout_frac=0.34, seed=7)
    assert a == b


def test_never_empties_train_even_with_huge_frac():
    s = make_category_holdout(OBJECTS, holdout_frac=1.0, seed=0)
    assert len(s.train_ids) > 0  # coverage guard keeps >=1 category in train


def test_object_holdout_fallback():
    ids = [f"o{i}" for i in range(10)]
    s = make_object_holdout(ids, holdout_frac=0.3, seed=0)
    assert len(s.test_ids) == 3
    assert set(s.train_ids).isdisjoint(s.test_ids)
    assert len(s.train_ids) + len(s.test_ids) == 10


def test_empty_inputs_raise():
    for bad in (lambda: make_category_holdout([]), lambda: make_object_holdout([])):
        try:
            bad()
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
