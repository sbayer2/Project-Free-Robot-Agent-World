"""Tests for the F31 trajectory-oracle pilot (scripts/f31_trajectory_oracle.py)."""

import os
import sys

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import f31_trajectory_oracle as T  # noqa: E402


def _traj(n=125, dt=1 / 63, v=0.0, stop_at=None):
    """Object sliding along +x at speed v, stopping after frame ``stop_at``."""
    out, x = [], 0.0
    for i in range(n):
        out.append({"t": i * dt, "pos": [x, 0.0, 0.15], "up": [0.0, 0.0, 1.0]})
        if stop_at is None or i < stop_at:
            x += v * dt
    return out


def test_resample_is_relative_to_first_frame_and_fixed_length():
    tr = _traj(v=1.0)
    R = T.resample(tr, 16)
    assert R.shape == (16, 6)
    assert np.allclose(R[0], [0, 0, 0, 0, 0, 1])      # pos relative, up untouched
    assert R[-1, 0] == pytest.approx(tr[-1]["pos"][0])


def test_resample_rejects_degenerate_trajectory():
    with pytest.raises(ValueError):
        T.resample(_traj(n=1), 4)


def test_motion_fraction_stationary_moving_and_stopping():
    assert T.motion_fraction(_traj(v=0.0)) == 0.0
    assert T.motion_fraction(_traj(v=1.0)) == pytest.approx(1.0)
    half = T.motion_fraction(_traj(v=1.0, stop_at=62))
    assert 0.45 < half < 0.55


def test_standardize_uses_train_stats_and_drops_constant_columns():
    Ytr = np.array([[1.0, 5.0, 0.0], [3.0, 5.0, 2.0]])
    Yte = np.array([[2.0, 5.0, 4.0]])
    a, b = T.standardize(Ytr, Yte)
    assert a.shape == (2, 2) and b.shape == (1, 2)    # constant middle column gone
    assert np.allclose(a.mean(0), 0.0)
    assert np.allclose(b, [[0.0, 3.0]])               # scaled by TRAIN sd, not test


def test_inverse_r2_recovers_a_linear_code_and_rejects_noise():
    rng = np.random.default_rng(0)
    E = rng.uniform(size=(300, 3))
    X_good = E @ rng.normal(size=(3, 8))
    X_noise = rng.normal(size=(300, 8))
    good = T.inverse_r2(X_good[:240], E[:240], X_good[240:], E[240:])
    bad = T.inverse_r2(X_noise[:240], E[:240], X_noise[240:], E[240:])
    assert min(good.values()) > 0.95
    assert max(bad.values()) < 0.2


def test_iid_split_only_draws_from_train_region():
    sp = np.array(["train"] * 50 + ["test"] * 10)
    tr, te = T.iid_split(sp, seed=0)
    assert not (tr & te).any()
    assert (tr | te).sum() == 50 and not (tr | te)[50:].any()
