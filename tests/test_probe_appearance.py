"""Tests for the F19 z->appearance_params probe (scripts/probe_appearance.py).

Numpy-only core (kfold R^2, retention framing, pixel features, appearance-row
parsing); the MLX encoding path is exercised on the Mac. Encodes the metric
correction from the F19 amendment: retention is z_trained/z_untrained, and it is
only defined where the random encoder itself decodes the channel.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from probe_appearance import (  # noqa: E402
    APPEARANCE_CHANNELS,
    appearance_row,
    kfold_r2,
    pixel_features,
)


def test_kfold_r2_is_high_for_a_linear_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    W = rng.normal(size=(5, 2))
    Y = X @ W  # exactly linear -> R^2 ~ 1
    r2 = kfold_r2(X, Y)
    assert r2.shape == (2,)
    assert np.all(r2 > 0.95)


def test_kfold_r2_is_near_zero_for_pure_noise():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5))
    Y = rng.normal(size=(200, 2))  # unrelated -> out-of-fold R^2 ~ 0 or below
    r2 = kfold_r2(X, Y)
    assert np.all(r2 < 0.3)


def test_appearance_row_is_eight_channels_in_order():
    mt = {"appearance_params": {"base_color": [0.1, 0.2, 0.3, 1.0], "roughness": 0.4,
                                "metallic": 0.5, "transmission": 0.6, "ior": 1.5}}
    row = appearance_row(mt)
    assert row == [0.1, 0.2, 0.3, 1.0, 0.4, 0.5, 0.6, 1.5]
    assert len(row) == len(APPEARANCE_CHANNELS)


def test_pixel_features_shape_and_finiteness():
    rng = np.random.default_rng(2)
    imgs = rng.random((6, 8, 32, 32, 3)).astype("float32")  # (B, views, H, W, C)
    feats = pixel_features(imgs)
    # 3 channel-mean + 3 channel-std + 8*8*3 downsample = 198.
    assert feats.shape == (6, 3 + 3 + 8 * 8 * 3)
    assert np.all(np.isfinite(feats))


def test_retention_is_the_ratio_not_the_subtractive_baseline():
    """The F19 correction: a random projection is near-lossless, so retention =
    z_trained/z_untrained, NOT (trained-untrained)/(pixels-untrained)."""
    tr = np.array([0.77, 0.27])
    un = np.array([0.91, 0.53])
    present = un > 0.1
    retention = np.where(present, tr / np.where(present, un, 1.0), np.nan)
    assert retention[0] == pytest.approx(0.846, abs=1e-2)
    assert retention[1] == pytest.approx(0.509, abs=1e-2)


def test_retention_undefined_when_channel_absent_from_random_encoder():
    tr = np.array([-0.31])
    un = np.array([-0.44])  # not linearly present -> retention n/a
    present = un > 0.1
    retention = np.where(present, tr / np.where(present, un, 1.0), np.nan)
    assert np.isnan(retention[0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
