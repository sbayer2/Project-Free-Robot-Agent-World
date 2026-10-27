"""Shape and invariance checks for the cross-view pooling pilot's features."""

import os
import sys

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
pilot = pytest.importorskip("crossview_pilot")


def _imgs(seed=0, b=4, v=8, hw=8):
    return np.random.default_rng(seed).random((b, v, hw, hw, 3)).astype(np.float32)


def test_per_view_shape():
    assert pilot.per_view(_imgs()).shape == (4, 8, 13)


def test_dft_magnitude_ignores_cyclic_view_shift():
    f = pilot.per_view(_imgs())
    a = np.abs(np.fft.rfft(f, axis=1))
    b = np.abs(np.fft.rfft(np.roll(f, 3, axis=1), axis=1))
    assert np.allclose(a, b, atol=1e-5)


def test_sorted_and_stats_sets_are_order_free():
    imgs = _imgs()
    shuffled = imgs[:, np.random.default_rng(1).permutation(imgs.shape[1])]
    s1, s2 = pilot.feature_sets(imgs), pilot.feature_sets(shuffled)
    for key in ("C mean + std across views", "G sorted across views",
                "H mean + |dev| quantiles"):
        assert np.allclose(s1[key], s2[key], atol=1e-5), key


def test_nested_r2_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    Y = np.stack([X @ rng.normal(size=5), rng.normal(size=120)], 1)
    r2 = pilot.nested_r2(X, Y)
    assert r2[0] > 0.95 and r2[1] < 0.1
