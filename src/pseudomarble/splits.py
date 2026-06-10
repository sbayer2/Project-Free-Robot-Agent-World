"""The honest split — held-out *material x shape combinations*.

If we train and test on the same (shape, material) pairings, a shared-latent
model will of course recover the appearance->physics coupling we authored, and
"beat" two independent models. That demonstrates almost nothing: it rediscovers
a correlation we typed in by hand.

The defensible question is *generalization*: hold out specific (shape, material)
combinations entirely, then measure whether the model infers physics for an
appearance it never saw paired with that shape. If it interpolates the coupling,
that is a real result. If it only memorizes seen pairs, the "eigenvector is
learnable" claim fails — and that failure is worth knowing.

Constraints we enforce so the test is meaningful:
  * every shape appears in train (with at least one material),
  * every material appears in train (with at least one shape),
  * held-out pairs appear *only* in test (novel combinations),
  * the split is deterministic given a seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Pair = Tuple[str, str]  # (shape_id, material_name)
Interval = Tuple[float, float]


@dataclass(frozen=True)
class CombinationSplit:
    train_pairs: Tuple[Pair, ...]
    test_pairs: Tuple[Pair, ...]

    def summary(self) -> Dict[str, int]:
        return {
            "n_train": len(self.train_pairs),
            "n_test": len(self.test_pairs),
            "n_shapes": len({s for s, _ in self.train_pairs + self.test_pairs}),
            "n_materials": len({m for _, m in self.train_pairs + self.test_pairs}),
        }


def make_combination_split(
    shapes: Sequence[str],
    materials: Sequence[str],
    holdout_frac: float = 0.2,
    seed: int = 0,
) -> CombinationSplit:
    """Partition the full shape x material grid into train / test combinations.

    A ``holdout_frac`` fraction of the full grid is reserved for test as *novel
    combinations*, subject to the coverage constraints in the module docstring.
    """
    if not shapes:
        raise ValueError("need at least one shape")
    if not materials:
        raise ValueError("need at least one material")
    if not 0.0 <= holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in [0,1): {holdout_frac}")

    shapes = list(dict.fromkeys(shapes))  # de-dup, preserve order
    materials = list(dict.fromkeys(materials))

    full_grid: List[Pair] = [(s, m) for s in shapes for m in materials]
    rng = random.Random(seed)

    target_holdout = int(round(len(full_grid) * holdout_frac))
    if target_holdout == 0:
        return CombinationSplit(tuple(full_grid), tuple())

    # Greedily choose held-out pairs while preserving coverage: a pair may be
    # held out only if both its shape and material still have another pair left
    # in train afterwards.
    candidates = full_grid[:]
    rng.shuffle(candidates)

    shape_train_count = {s: len(materials) for s in shapes}
    material_train_count = {m: len(shapes) for m in materials}

    holdout: List[Pair] = []
    for (s, m) in candidates:
        if len(holdout) >= target_holdout:
            break
        if shape_train_count[s] > 1 and material_train_count[m] > 1:
            holdout.append((s, m))
            shape_train_count[s] -= 1
            material_train_count[m] -= 1

    holdout_set = set(holdout)
    train = [p for p in full_grid if p not in holdout_set]
    # Deterministic ordering for reproducible manifests.
    train.sort()
    holdout.sort()
    return CombinationSplit(tuple(train), tuple(holdout))


# --------------------------------------------------------------------------- #
# Continuous-material generalization: hold out a REGION of essence-space.
# --------------------------------------------------------------------------- #
# With continuously-sampled materials there is no finite grid to hold combos out
# of. Instead we reserve a *box* in normalized essence-space (and optionally only
# for certain shapes) for the test set. A material whose essence lands in the box
# goes to test; everything else trains. Two flavors:
#   * INTERPOLATION — an *interior* box, surrounded by training data on all sides.
#     A smooth model fills an interior hole almost trivially, so it's a weak test.
#   * EXTRAPOLATION — a *corner/extreme*, where for held-out points no training
#     object lies jointly beyond them, so the model must extrapolate the coupling
#     OUTSIDE the training manifold's convex hull. This is the test with teeth.
@dataclass(frozen=True)
class RegionHoldout:
    """A held-out box in normalized (density, friction, restitution) space.

    Each axis is an optional ``(min, max)`` interval on the *normalized* value
    (raw / config.PHYSICS_NORMALIZERS). ``None`` means that axis is unconstrained.
    If ``shapes`` is non-empty, only those shapes count as held out (so the same
    essence can be train-on-one-shape, test-on-another). ``kind`` is a label
    recorded in the manifest ("interpolation" | "extrapolation").
    """

    density: Optional[Interval] = None
    friction: Optional[Interval] = None
    restitution: Optional[Interval] = None
    shapes: Tuple[str, ...] = field(default_factory=tuple)
    kind: str = "region"

    def contains(self, essence_norm: Dict[str, float], shape: str) -> bool:
        if self.shapes and shape not in self.shapes:
            return False
        axes = (
            ("density", self.density),
            ("friction", self.friction),
            ("restitution", self.restitution),
        )
        constrained = False
        for key, interval in axes:
            if interval is None:
                continue
            constrained = True
            lo, hi = interval
            if not (lo <= essence_norm[key] <= hi):
                return False
        # An all-None holdout would match everything; require >=1 constraint.
        return constrained

    def label(self, essence_norm: Dict[str, float], shape: str) -> str:
        return "test" if self.contains(essence_norm, shape) else "train"


# INTERPOLATION (weak): a mid-grip, fairly-bouncy *interior* box.
DEFAULT_REGION_HOLDOUT = RegionHoldout(
    friction=(0.55, 0.80),
    restitution=(0.55, 0.80),
    kind="interpolation",
)

# EXTRAPOLATION (the test with teeth): the HEAVY *and* BOUNCY corner. Normalized
# essence maxes out near density~1.0 and restitution~0.85; this box sits at that
# upper corner, so training has heavy OR bouncy objects but never both-extreme
# together — predicting their behavior requires extrapolating the coupling beyond
# anything seen. ~3-4% of samples land here (a non-empty, non-dominant test set).
EXTRAPOLATION_REGION_HOLDOUT = RegionHoldout(
    density=(0.55, 1.01),
    restitution=(0.60, 0.90),
    kind="extrapolation",
)


def assign_scene_materials(
    shape_ids: Sequence[str],
    materials: Sequence[str],
    split: CombinationSplit,
    num_scenes: int,
    seed: int = 0,
) -> List[Dict[str, str]]:
    """Expand a combination split into concrete per-scene assignments.

    Returns a list of ``{"scene_id", "shape", "material", "split"}`` records,
    drawing train pairs for ~(1-holdout) of scenes and test pairs for the rest,
    so a single ``num_scenes`` run produces both partitions.
    """
    rng = random.Random(seed)
    records: List[Dict[str, str]] = []
    train = list(split.train_pairs)
    test = list(split.test_pairs)
    if not train:
        raise ValueError("split has no train pairs")

    n_test = (
        int(round(num_scenes * len(test) / max(1, len(train) + len(test))))
        if test
        else 0
    )
    n_train = num_scenes - n_test

    for i in range(n_train):
        s, m = rng.choice(train)
        records.append(
            {"scene_id": f"train_{i:06d}", "shape": s, "material": m, "split": "train"}
        )
    for i in range(n_test):
        s, m = rng.choice(test)
        records.append(
            {"scene_id": f"test_{i:06d}", "shape": s, "material": m, "split": "test"}
        )
    return records


# --------------------------------------------------------------------------- #
# Real-object generalization: hold out whole CATEGORIES (or objects).
# --------------------------------------------------------------------------- #
# For scanned objects (GSO) the material/appearance is baked per object — you
# cannot freely recombine shape x material, so the synthetic combination/region
# splits do not transfer. The honest generalization test is whether the model
# infers physical behavior for object *categories* (or objects) it never trained
# on. We hold out entire categories so no object of a held-out kind leaks into
# train; if categories are unknown/degenerate, fall back to a random object split.
@dataclass(frozen=True)
class CategorySplit:
    train_ids: Tuple[str, ...]
    test_ids: Tuple[str, ...]
    holdout_categories: Tuple[str, ...]

    def summary(self) -> Dict[str, int]:
        return {"n_train": len(self.train_ids), "n_test": len(self.test_ids),
                "n_holdout_categories": len(self.holdout_categories)}


def make_category_holdout(
    objects: Sequence[Tuple[str, str]],
    holdout_categories: Optional[Sequence[str]] = None,
    holdout_frac: float = 0.2,
    seed: int = 0,
) -> CategorySplit:
    """Split ``objects`` (list of ``(object_id, category)``) by holding out whole
    categories. If ``holdout_categories`` is given, those go to test; otherwise a
    ``holdout_frac`` fraction of the categories is chosen at random. Every held-out
    category's objects go to test; all others train. Coverage guard: never hold
    out *all* categories (training set must be non-empty)."""
    if not objects:
        raise ValueError("need at least one object")
    cats = list(dict.fromkeys(c for _, c in objects))  # stable unique categories
    rng = random.Random(seed)

    if holdout_categories is not None:
        held = [c for c in cats if c in set(holdout_categories)]
    else:
        n = min(len(cats) - 1, int(round(len(cats) * holdout_frac)))
        shuffled = cats[:]
        rng.shuffle(shuffled)
        held = shuffled[:max(0, n)]
    held_set = set(held)
    if len(held_set) >= len(cats):  # would empty the train set
        held_set = set(list(held_set)[:-1])

    train, test = [], []
    for oid, cat in objects:
        (test if cat in held_set else train).append(oid)
    if not train:
        raise ValueError("holdout emptied the train set; reduce holdout_categories")
    return CategorySplit(tuple(train), tuple(test), tuple(sorted(held_set)))


def make_object_holdout(object_ids: Sequence[str], holdout_frac: float = 0.2,
                        seed: int = 0) -> CategorySplit:
    """Fallback when categories are unknown: hold out a random fraction of objects
    (weaker than a category split — same object *kind* may appear in both)."""
    if not object_ids:
        raise ValueError("need at least one object")
    ids = list(dict.fromkeys(object_ids))
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_test = min(len(ids) - 1, int(round(len(ids) * holdout_frac)))
    test = set(shuffled[:n_test])
    train = tuple(i for i in ids if i not in test)
    return CategorySplit(train, tuple(shuffled[:n_test]), ())
