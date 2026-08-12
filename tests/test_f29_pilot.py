"""Tests for the F29 pilot driver's pure-Python geometry (scripts/f29_pilot.py).

Pins the frozen prereg constants (docs/HELD_OUT_TRANSFER.md sections 3, 4)
against the module and exercises the alpha* rule end-to-end. Numpy-only,
no MLX, no filesystem -- runs in any session.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from f29_pilot import (  # noqa: E402
    ACCEPTABLE_BAND,
    JOINT_CHECKPOINTS,
    N_SCENES_PER_ARM,
    PILOT_ALPHAS,
    PILOT_ARM,
    PILOT_DATA,
    TARGET_DROP,
    alpha_tag,
    combined_drop,
    drop_ratio,
    pick_alpha,
    pilot_summary,
    within_acceptable_band,
)


def test_frozen_prereg_constants_pinned():
    """Section-3/4 commitments. Any edit to these fails here."""
    assert PILOT_ALPHAS == (0.05, 0.20, 0.40)
    assert TARGET_DROP == 0.5
    assert ACCEPTABLE_BAND == (0.25, 0.75)
    assert PILOT_ARM == "loud"
    assert N_SCENES_PER_ARM == 128
    assert JOINT_CHECKPOINTS == "runs/f27b/{arm}_s{seed}/model.safetensors"
    assert PILOT_DATA == "data/pm_f29_pilot_a{tag}_{arm}"


def test_alpha_tag_matches_dataset_naming_convention():
    """PILOT_DATA + alpha_tag together produce runs/data path names."""
    assert alpha_tag(0.05) == "5"
    assert alpha_tag(0.20) == "20"
    assert alpha_tag(0.40) == "40"
    for a in PILOT_ALPHAS:
        path = PILOT_DATA.format(tag=alpha_tag(a), arm="loud")
        assert path.startswith("data/pm_f29_pilot_a")
        assert path.endswith("_loud")
    with pytest.raises(ValueError):
        alpha_tag(0.055)  # non-percent
    with pytest.raises(ValueError):
        alpha_tag(0.0)
    with pytest.raises(ValueError):
        alpha_tag(1.0)


def test_drop_ratio_reference_cases():
    assert drop_ratio(5.0, 5.0) == pytest.approx(0.0)     # no drop
    assert drop_ratio(5.0, 2.5) == pytest.approx(0.5)     # exactly half
    assert drop_ratio(5.0, 1.0) == pytest.approx(0.8)     # heavy drop
    assert drop_ratio(5.0, 6.0) == pytest.approx(-0.2)    # held-out better (negative)
    assert math.isnan(drop_ratio(0.0, 0.0))               # undefined
    assert math.isnan(drop_ratio(-0.1, 1.0))              # invalid train gain


def test_combined_drop_averages_students():
    """Two students at 0.4 and 0.6 drops -> combined 0.5."""
    j = drop_ratio(5.0, 3.0)
    d = drop_ratio(4.0, 1.6)
    assert j == pytest.approx(0.4)
    assert d == pytest.approx(0.6)
    assert combined_drop(5.0, 3.0, 4.0, 1.6) == pytest.approx(0.5)


def test_combined_drop_nan_when_either_student_invalid():
    assert math.isnan(combined_drop(0.0, 1.0, 4.0, 2.0))    # joint invalid
    assert math.isnan(combined_drop(5.0, 3.0, -1.0, 1.0))   # disjoint invalid


def test_pick_alpha_closest_to_half():
    # drops 0.10 / 0.48 / 0.75 -> distances 0.40 / 0.02 / 0.25 -> alpha 0.20 wins
    assert pick_alpha(PILOT_ALPHAS, [0.10, 0.48, 0.75]) == pytest.approx(0.20)
    # exactly-on-target wins even against a close runner-up
    assert pick_alpha(PILOT_ALPHAS, [0.50, 0.51, 0.49]) == pytest.approx(0.05)


def test_pick_alpha_tiebreak_prefers_smaller_alpha():
    """Ties -> less aggression wins (more measurable transfer)."""
    # 0.40 and 0.60 are equidistant from 0.5 -> smaller alpha 0.05
    assert pick_alpha(PILOT_ALPHAS, [0.40, 0.60, 0.20]) == pytest.approx(0.05)


def test_pick_alpha_rejects_nan_options():
    # 0.05 unmeasurable (NaN) -> next best is 0.40 at drop 0.55 (dist 0.05)
    assert pick_alpha(PILOT_ALPHAS, [float("nan"), 0.10, 0.55]) == pytest.approx(0.40)
    with pytest.raises(ValueError):
        pick_alpha(PILOT_ALPHAS, [float("nan")] * 3)


def test_pick_alpha_length_mismatch_errors():
    with pytest.raises(ValueError):
        pick_alpha((0.1, 0.2), [0.5])


def test_within_acceptable_band_boundaries():
    assert within_acceptable_band(0.25) is True    # inclusive
    assert within_acceptable_band(0.75) is True    # inclusive
    assert within_acceptable_band(0.5) is True
    assert within_acceptable_band(0.24) is False
    assert within_acceptable_band(0.76) is False


def test_pilot_summary_full_schema():
    """The dict layout Amendment 1 will pin. Any schema drift fails here."""
    per_alpha = [
        {"joint_train": 5.67, "joint_held": 5.10, "disjoint_train": 4.80,
         "disjoint_held": 4.30, "joint_r_held": 0.99, "disjoint_r_held": 0.99,
         "extrapolation_frac": 1.0},                            # small drops
        {"joint_train": 5.67, "joint_held": 3.00, "disjoint_train": 4.80,
         "disjoint_held": 2.30, "joint_r_held": 0.99, "disjoint_r_held": 0.99,
         "extrapolation_frac": 1.0},                            # ~half drops
        {"joint_train": 5.67, "joint_held": 1.20, "disjoint_train": 4.80,
         "disjoint_held": 0.90, "joint_r_held": 0.99, "disjoint_r_held": 0.99,
         "extrapolation_frac": 1.0},                            # near-collapse
    ]
    r = pilot_summary(PILOT_ALPHAS, per_alpha)
    assert r["target_drop"] == 0.5
    assert r["acceptable_band"] == [0.25, 0.75]
    assert r["pilot_arm"] == "loud"
    assert r["alphas_tested"] == list(PILOT_ALPHAS)
    assert len(r["per_alpha"]) == 3
    row = r["per_alpha"][1]  # the ~half-drop row
    for k in ("alpha", "joint_train", "joint_held", "disjoint_train",
              "disjoint_held", "joint_drop", "disjoint_drop", "combined_drop",
              "joint_r_held", "disjoint_r_held", "extrapolation_frac"):
        assert k in row, f"missing schema key {k}"
    # alpha* = 0.20 (its combined_drop ~0.71/2 + 0.52/2 = ~0.5 mid-band)
    assert r["alpha_star"] == pytest.approx(0.20)
    assert r["within_acceptable_band"] is True


def test_pilot_summary_flags_out_of_band_miss():
    """A pilot whose alpha* misses the band gets flagged (Amendment 1 must
    explicitly disclose; the alpha* is not overridden)."""
    per_alpha = [
        {"joint_train": 5.67, "joint_held": 5.60, "disjoint_train": 4.80,
         "disjoint_held": 4.75},  # tiny drop
        {"joint_train": 5.67, "joint_held": 5.50, "disjoint_train": 4.80,
         "disjoint_held": 4.60},  # still tiny
        {"joint_train": 5.67, "joint_held": 5.40, "disjoint_train": 4.80,
         "disjoint_held": 4.50},  # still tiny
    ]
    r = pilot_summary(PILOT_ALPHAS, per_alpha)
    # all three combined_drops well below 0.25 -> whichever wins is out-of-band
    assert r["within_acceptable_band"] is False
    assert r["combined_drop_at_alpha_star"] < 0.25
