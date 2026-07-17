"""Tests for the F18 oracle-ceiling harness (scripts/oracle_ceiling.py).

Encodes two design decisions that were learned the hard way:
  * the quadratic expansion must REFUSE to build a Gram matrix that would blow
    up memory (a 492-col one-hot expanded to ~124k features and 51 GB RSS), and
  * the bit curve must skip cluster counts without training support rather than
    silently reporting a memorization score.

Numpy-only; no MLX/MuJoCo. Runs standalone or under pytest.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oracle_ceiling import (  # noqa: E402
    MAX_QUAD_FEATURES,
    best_oracle,
    gain,
    quadratic,
    quantize,
    ridge_fit_predict,
)


def test_quadratic_refuses_to_build_an_enormous_gram_matrix():
    """The F18 bug: quadratic() on a wide one-hot block asked for a 123 GB solve."""
    wide = np.zeros((10, 500))
    with pytest.raises(ValueError, match="quad=False"):
        quadratic(wide)


def test_quadratic_expands_small_inputs():
    X = np.arange(12, dtype=float).reshape(4, 3)
    Q = quadratic(X)
    # 3 linear + 6 unique pairwise (incl. squares).
    assert Q.shape == (4, 9)
    assert np.allclose(Q[:, :3], X)


def test_quad_feature_cap_is_the_documented_bound():
    n = 10
    assert quadratic(np.zeros((2, n))).shape[1] == n * (n + 3) // 2 <= MAX_QUAD_FEATURES


def test_best_oracle_quad_false_skips_the_quadratic_arm():
    """quad=False must stay cheap AND still return a usable prediction."""
    rng = np.random.default_rng(0)
    X = np.eye(40)[rng.integers(0, 40, 80)]  # 40-col one-hot -> 860 quad features
    Y = rng.normal(size=(80, 3))
    g, which, P = best_oracle(X[:60], Y[:60], X[60:], Y[60:], quad=False)
    assert which != "ridge-quad"
    assert P.shape == (20, 3)
    assert np.isfinite(g)


def test_ridge_recovers_an_exact_linear_map():
    """Sanity: the oracle's regressor must fit what it is given the answers for."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    W = rng.normal(size=(4, 2))
    Y = X @ W + 5.0
    P = ridge_fit_predict(X, Y, X, alpha=1e-9)
    assert np.allclose(P, Y, atol=1e-4)


def test_gain_is_one_when_the_oracle_predicts_the_train_mean():
    """gain = MSE(predict train-mean) / MSE(oracle); predicting the mean scores 1."""
    rng = np.random.default_rng(2)
    Ytr = rng.normal(size=(50, 3))
    Yte = rng.normal(size=(10, 3))
    P = np.repeat(Ytr.mean(0)[None], 10, axis=0)
    assert gain(Ytr, Yte, P) == pytest.approx(1.0)


def test_gain_exceeds_one_for_a_real_predictor():
    rng = np.random.default_rng(3)
    Ytr = rng.normal(size=(50, 2))
    Yte = rng.normal(size=(10, 2))
    assert gain(Ytr, Yte, Yte + 1e-3) > 1.0


def test_quantize_is_a_one_hot_capped_at_the_cluster_count():
    """The oracle's analogue of F17's FSQ: essence capped at log2(n) bits."""
    rng = np.random.default_rng(4)
    E = rng.normal(size=(60, 3))
    oh = quantize(E, E[:40], n_clusters=9)
    assert oh.shape == (60, 9)
    assert np.all(oh.sum(1) == 1)
    assert set(np.unique(oh)) <= {0.0, 1.0}


def test_quantize_never_exceeds_available_train_points():
    E = np.random.default_rng(5).normal(size=(20, 3))
    assert quantize(E, E[:8], n_clusters=81).shape[1] == 8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
