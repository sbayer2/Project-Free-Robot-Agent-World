"""Tests for scripts/f31_precision_diag.py.

The band decomposition must be an exact rewrite of the ridge the F31a readout
uses (all components at 1e-3 reproduces kfold_r2's fixed alpha; all at 1e-5 the
direct ridge at 1e-5), and components below the std_rank threshold must be
labelled separately. TF32 rounding must keep 10 stored mantissa bits.
"""

from __future__ import annotations

import os
import sys

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import f31_precision_diag as P  # noqa: E402
from oracle_ceiling import ridge_fit_predict  # noqa: E402
from probe_appearance import kfold_r2  # noqa: E402


def _data(seed=0, n=200, p=40):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, p)) * np.logspace(0, -3, p)  # graded column scales
    z[:, 5] = 2.0                                           # a dead column
    z[:, 6] = z[:, 7] * (1 + 1e-9)                          # a near-duplicate
    y = z[:, 0] + 50 * z[:, 30] + 0.3 * rng.normal(size=n)
    return z, y


def _oof(z, y, alpha, k=5, seed=0):
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), k)
    pred = np.zeros_like(y)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        pred[te] = ridge_fit_predict(z[tr], y[tr, None], z[te], alpha=alpha)[:, 0]
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def test_all_fixed_weights_reproduce_kfold_r2_and_direct_low_alpha():
    z, y = _data()
    res = P.band_decompose(z, y)
    E = np.column_stack([y, y, y])
    assert abs(res["fixed"] - kfold_r2(z, E)[1]) < 1e-8
    assert abs(res["lo"] - _oof(z, y, P.A_LO)) < 1e-6


def test_band_gains_cover_total_and_counts_match_active_dims():
    z, y = _data(1)
    res = P.band_decompose(z, y)
    assert abs(sum(res["gain"].values()) - res["total_gain"]) < 0.05
    assert abs(res["gain"]["B1"]) < 0.01  # unshrunk directions cannot gain much
    assert abs(sum(res["components_per_fold"].values()) - (z.shape[1] - 1)) < 1e-9


def test_below_rank_components_are_split_off_b4():
    # f = s^2/(s^2 + 1e-3): 0.05 -> 0.71 (B2), 0.02 -> 0.29 (B3); the rank
    # threshold is 1e-6 * 100 = 1e-4, so 1e-3 stays in B4 and 1e-5 drops below.
    s = np.array([100.0, 1.0, 0.05, 0.02, 1e-3, 1e-5])
    assert P.bands_of(s).tolist() == [0, 0, 1, 2, 3, 4]


def test_tf32_keeps_ten_mantissa_bits():
    torch = pytest.importorskip("torch")
    x = torch.tensor([1.0, 1.0 + 2 ** -11, 1.0 + 2 ** -12, -3.14159, 0.0])
    got = P.tf32(x)
    assert got[0] == 1.0 and got[1] == 1.0 + 2 ** -10 and got[2] == 1.0 and got[4] == 0.0
    assert abs(float(got[3]) + 3.14159) <= 3.14159 * 2 ** -11
    r = torch.randn(10000)
    rel = ((P.tf32(r) - r).abs() / r.abs()).max()
    assert float(rel) <= 2 ** -11 + 1e-7
