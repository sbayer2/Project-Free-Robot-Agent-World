"""Checks for the per-view readout (scripts/perview_readout.py)."""

import os
import sys

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
pv = pytest.importorskip("perview_readout")
oracle = pytest.importorskip("oracle_ceiling")


def test_block_features_are_block_means_per_view():
    imgs = np.random.default_rng(0).random((2, 3, 32, 32, 3)).astype(np.float32)
    F = pv.block_features(imgs, g=16)
    assert F.shape == (2, 3, 16 * 16 * 3)
    expect = imgs[1, 2, 2:4, 6:8, :].mean(axis=(0, 1))
    assert np.allclose(F[1, 2].reshape(16, 16, 3)[1, 3], expect, atol=1e-6)


@pytest.mark.parametrize("p", [6, 300])
def test_dual_ridge_equals_the_primal_estimator(p):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, p)) * rng.uniform(0.1, 5.0, size=p) + 3.0
    Y = np.stack([X[:, 0] - X[:, 1], rng.normal(size=80)], 1)
    path = pv.RidgePath(X[:60], Y[:60])
    for alpha in (1e-1, 10.0, 1e3):
        assert np.allclose(path.predict(X[60:], alpha),
                           oracle.ridge_fit_predict(X[:60], Y[:60], X[60:], alpha=alpha),
                           atol=1e-6)


def test_nested_matches_the_single_target_readout():
    diag = pytest.importorskip("z_readout_alpha")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 8))
    y = X[:, 0] - 0.5 * X[:, 3] + rng.normal(size=150)
    grid = (1e-1, 1.0, 10.0, 1e2)
    out = pv.nested(X, y[:, None], grid=grid)
    r2, picks = diag.nested_grid(X, y, alphas=grid)
    assert out["r2"][0] == pytest.approx(r2, abs=1e-8)
    assert [p[0] for p in out["picks"]] == picks
    assert out["best_of_grid"][0] >= out["r2"][0] - 1e-12


def test_readout_sets():
    F = np.zeros((4, 16, 5))
    sets = pv.readout_sets(F)
    assert len(sets) == 18
    assert sets["view 3"].shape == (4, 5) and sets["mean"].shape == (4, 5)
    assert sets["concat"].shape == (4, 80)
