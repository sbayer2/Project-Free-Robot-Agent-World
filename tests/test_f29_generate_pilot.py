"""Tests for the F29 pilot dataset generator (scripts/f29_generate_pilot.py).

Pure-Python + scipy (hull tests skip cleanly where scipy is absent); no
MuJoCo, no filesystem beyond nothing. Pins the frozen section-3 filter
(top-alpha product tail AND outside the training hull), the arm dials
against runs/f27/launch.sh, the pilot seed convention, and — the load-
bearing invariant — the MATCHED-ARMS property: one seed per alpha plus a
physics-only filter must give all four arms the identical physics answer
key, differing only in appearance.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from f29_generate_pilot import (  # noqa: E402
    APPEARANCE_NOISE,
    ARM_DIALS,
    MAX_CANDIDATES,
    PILOT_SEED_BASE,
    normalize_phys,
    physics_rows,
    pilot_assignments,
    pilot_seed,
    product_threshold,
)

scipy_spatial = pytest.importorskip("scipy.spatial")


def test_frozen_constants_pinned():
    """Arm dials pinned to runs/f27/launch.sh; seed family; noise."""
    assert ARM_DIALS == {"ctrl": (0.0, 1.0), "base": (1.0, 1.0),
                         "g2": (1.0, 2.0), "loud": (1.0, 8.0)}
    assert PILOT_SEED_BASE == 2900
    assert pilot_seed(0.05) == 2905
    assert pilot_seed(0.20) == 2920
    assert pilot_seed(0.40) == 2940
    assert APPEARANCE_NOISE == 0.07
    assert MAX_CANDIDATES == 200_000


def test_product_threshold_quantile():
    """(1 - alpha) quantile of density x restitution, hand-checkable."""
    # products 1..100 exactly (density k, restitution 1, friction irrelevant)
    train = [(float(k), 0.5, 1.0) for k in range(1, 101)]
    t20 = product_threshold(train, 0.20)
    assert 80.0 <= t20 <= 81.0        # numpy linear quantile of 1..100 at q=0.8
    t05 = product_threshold(train, 0.05)
    assert 95.0 <= t05 <= 96.0


def _cube_train():
    """A unit-cube cloud of training physics in RAW units scaled so the
    normalized points span [0, 1]^3 exactly (normalize_phys divides by the
    PHYSICS_NORMALIZERS)."""
    from pseudomarble.config import PHYSICS_NORMALIZERS as N
    pts = []
    for a in (0.0, 1.0):
        for b in (0.0, 1.0):
            for c in (0.0, 1.0):
                pts.append((a * N["density"], b * N["friction"],
                            c * N["restitution"]))
    # add an interior point so Delaunay is non-degenerate under jitter
    pts.append((0.5 * N["density"], 0.5 * N["friction"], 0.5 * N["restitution"]))
    return pts


def test_hull_containment_inside_and_outside():
    from f29_generate_pilot import build_hull, passes_filter

    from pseudomarble.config import PHYSICS_NORMALIZERS as N
    train = _cube_train()
    hull = build_hull(train)
    threshold = -1.0  # product test passes trivially; isolate the hull check
    inside = (0.5 * N["density"], 0.5 * N["friction"], 0.5 * N["restitution"])
    outside = (1.5 * N["density"], 0.5 * N["friction"], 1.5 * N["restitution"])
    assert passes_filter(inside, threshold, hull) is False   # in hull -> reject
    assert passes_filter(outside, threshold, hull) is True   # out -> accept


def test_filter_requires_both_conditions():
    from f29_generate_pilot import build_hull, passes_filter

    from pseudomarble.config import PHYSICS_NORMALIZERS as N
    train = _cube_train()
    hull = build_hull(train)
    # outside the hull but BELOW the product threshold -> reject
    outside_low = (1.5 * N["density"], 0.5 * N["friction"], 0.0)
    big_threshold = 1e9
    assert passes_filter(outside_low, -1.0, hull) is True
    assert passes_filter(outside_low, big_threshold, hull) is False


def test_normalize_phys_uses_config_normalizers():
    from pseudomarble.config import PHYSICS_NORMALIZERS as N
    d, f, r = normalize_phys((N["density"], N["friction"], N["restitution"]))
    assert (d, f, r) == (1.0, 1.0, 1.0)


def _loose_train():
    """Training cloud occupying only the low corner of material space so a
    real MaterialSampler stream has plenty of acceptable (tail, outside-hull)
    candidates -- keeps pilot_assignments() fast in tests."""
    import random

    from pseudomarble.config import PHYSICS_NORMALIZERS as N
    rng = random.Random(0)
    return [(rng.uniform(0.02, 0.10) * N["density"],
             rng.uniform(0.02, 0.10) * N["friction"],
             rng.uniform(0.02, 0.10) * N["restitution"]) for _ in range(50)]


def test_pilot_assignments_matched_arms_property():
    """THE apparatus invariant: same alpha => identical physics sequence
    across all four arms (filter reads physics only; appearance dials do
    not perturb the material stream)."""
    train = _loose_train()
    per_arm = {arm: pilot_assignments(arm, 0.20, train, n_scenes=12)[0]
               for arm in ARM_DIALS}
    ref = physics_rows(per_arm["ctrl"])
    for arm in ("base", "g2", "loud"):
        assert physics_rows(per_arm[arm]) == ref, f"{arm} physics diverged"
    # ...and the appearance DOES differ between ctrl (alpha=0) and loud (g=8)
    a_ctrl = per_arm["ctrl"][0]["sample"].material.visual
    a_loud = per_arm["loud"][0]["sample"].material.visual
    assert a_ctrl != a_loud


def test_pilot_assignments_split_ids_and_determinism():
    train = _loose_train()
    a1, s1 = pilot_assignments("loud", 0.20, train, n_scenes=8)
    a2, _s2 = pilot_assignments("loud", 0.20, train, n_scenes=8)
    assert [x["scene_id"] for x in a1] == [f"test_{i:06d}" for i in range(8)]
    assert all(x["split"] == "test" for x in a1)
    assert all(x["shape"] == "box" for x in a1)
    assert physics_rows(a1) == physics_rows(a2)  # deterministic
    assert s1["n_drawn"] >= 8 and 0 < s1["acceptance_rate"] <= 1.0


def test_pilot_assignments_accepted_candidates_actually_pass_filter():
    from f29_generate_pilot import build_hull, passes_filter
    train = _loose_train()
    assignments, stats = pilot_assignments("base", 0.05, train, n_scenes=8)
    hull = build_hull(train)
    for row in physics_rows(assignments):
        assert passes_filter(row, stats["threshold"], hull)


def test_pilot_assignments_thin_tail_fails_loudly(monkeypatch):
    """A filter nothing can pass must raise, not spin forever."""
    import f29_generate_pilot as g
    monkeypatch.setattr(g, "MAX_CANDIDATES", 200)
    train = _loose_train()
    monkeypatch.setattr(g, "product_threshold", lambda *_: float("inf"))
    with pytest.raises(RuntimeError, match="too thin"):
        g.pilot_assignments("ctrl", 0.05, train, n_scenes=8)
