"""Pins for the R1 framing rule's floor (scripts/f35_r1_floor.py;
docs/CROSS_INTERACTION.md §10.2a): admissibility, placement, rule 4's
out-of-view treatment, the tie rule and the split."""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import f35_r1_floor as R  # noqa: E402


def test_admissible_rungs_stop_at_two_pixels_per_body():
    assert [w for w in R.LADDER if R.admissible("kick", w)] == [0.5, 1, 2, 4, 8]
    assert [w for w in R.LADDER if R.admissible("strike", w)] == [0.5, 1, 2, 4]


def test_window_placement():
    assert np.allclose(R.window(("trail", "ground"), 2.0), (-0.3, 1.7, -0.3, 1.7))
    assert np.allclose(R.window(("centre", "ground"), 1.0), (-0.5, 0.5, -0.15, 0.85))
    assert np.allclose(R.window(("trail", "top"), 1.0), (-0.15, 0.85, -0.85, 0.15))


def test_out_of_view_is_constant_zero_and_noiseless():
    uv = np.array([[[0.5, 0.1], [3.0, 0.1], [-2.0, 5.0]]])        # in, right, out
    up = np.tile([0.0, 0.0, 1.0], (1, 3, 1))
    win = R.window(("trail", "ground"), 1.0)
    rng = np.random.default_rng(0)
    X = R.framed(uv, up, win, 1.0, rng)
    assert X[0, :, 5].tolist() == [1.0, 0.0, 0.0]
    # both coordinates constant out of view: the uncrossed edge's coordinate
    # must not survive (a box leaving sideways would still report its height)
    assert (X[0, 1:, :2] == 0).all()
    assert (X[0, 1:, 2:5] == 0).all()                              # up zeroed
    assert not np.allclose(X[0, 0, :2], [0.5, 0.1])                # in view: noisy
    assert abs(np.linalg.norm(X[0, 0, 2:5]) - 1.0) < 1e-12         # rotation keeps |up|


def test_noise_scales_with_window():
    uv = np.zeros((1, 4000, 2)) + 0.3
    up = np.tile([0.0, 0.0, 1.0], (1, 4000, 1))
    for w in (1.0, 4.0):
        X = R.framed(uv, up, R.window(("centre", "ground"), w), w, np.random.default_rng(1))
        sd = (X[0, :, 0] - 0.3).std()
        assert abs(sd / (0.5 * w / 96) - 1.0) < 0.05


def test_tie_rule_takes_the_largest_tied_rung():
    assert R.choose_w({0.5: 0.90, 1.0: 0.895, 2.0: 0.85}) == 1.0
    assert R.choose_w({0.5: 0.90, 1.0: 0.80}) == 0.5


def test_split_is_fixed_and_disjoint():
    tr, te = R.split(512)
    assert (len(tr), len(te)) == (410, 102)
    assert not set(tr) & set(te)
    assert (R.split(512)[0] == tr).all()



def test_skill_vs_train_variance_stays_defined_for_a_saturated_field():
    Y = np.array([[-2.0, 0.1], [-2.0, 0.3], [-2.0, 0.2]])        # field 0 constant
    P = np.array([[-2.0, 0.1], [-1.9, 0.2], [-2.0, 0.2]])
    sk = R.skill(Y, P, ref_var=np.array([1.0, 0.04]))
    assert abs(sk[0] - (1 - 0.01 / 3)) < 1e-12 and np.isfinite(sk).all()
