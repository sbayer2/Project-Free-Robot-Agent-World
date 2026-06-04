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
# goes to test; everything else trains. This tests true interpolation /
# extrapolation: predicting behavior for essences never seen during training.
@dataclass(frozen=True)
class RegionHoldout:
    """A held-out box in normalized (density, friction, restitution) space.

    Each axis is an optional ``(min, max)`` interval on the *normalized* value
    (raw / config.PHYSICS_NORMALIZERS). ``None`` means that axis is unconstrained.
    If ``shapes`` is non-empty, only those shapes count as held out (so the same
    essence can be train-on-one-shape, test-on-another).
    """

    density: Optional[Interval] = None
    friction: Optional[Interval] = None
    restitution: Optional[Interval] = None
    shapes: Tuple[str, ...] = field(default_factory=tuple)

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


# A mid-grip, fairly-bouncy corner of essence-space, held out by default. Forces
# the model to extrapolate behavior for materials it never trained on.
DEFAULT_REGION_HOLDOUT = RegionHoldout(
    friction=(0.55, 0.80),
    restitution=(0.55, 0.80),
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
