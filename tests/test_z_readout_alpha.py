"""Checks for the readout-strength diagnostic (scripts/z_readout_alpha.py)."""

import os
import sys

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
diag = pytest.importorskip("z_readout_alpha")
probe = pytest.importorskip("probe_appearance")


def test_std_rank_counts_collinear_and_dead_dims():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 5))
    z = np.concatenate([base, base[:, :2] * 3.0 + 1.0, np.zeros((200, 3))], 1)
    assert diag.std_rank(z) == (7, 5)


def test_folds_match_the_primary_readout():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(120, 6))
    y = z @ rng.normal(size=6) + 0.1 * rng.normal(size=120)
    sweep, _ = diag.alpha_sweep(z, y, diag.folds_for(len(y)), alphas=(1e-3,))
    assert sweep["0.001"] == pytest.approx(probe.kfold_r2(z, y[:, None])[0], abs=1e-9)


def test_signal_in_a_near_null_direction_is_lost_to_strong_ridge():
    rng = np.random.default_rng(2)
    a = rng.normal(size=400)
    y = rng.normal(size=400)
    # two dims that nearly cancel: their difference carries y at 1 % of their spread
    z = np.stack([a, a + 0.01 * y, rng.normal(size=400)], 1)
    sweep, _ = diag.alpha_sweep(z, y, diag.folds_for(len(y)), alphas=(1e-3, 100.0))
    assert sweep["0.001"] > 0.9
    assert sweep["100"] < 0.5
