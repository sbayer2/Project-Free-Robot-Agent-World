"""Tests for scripts/specular_view_diag.py.

The fast tests pin the readout that the first cloud run got wrong (an
alpha * n penalty floor that shrank away friction's faint directions): the
penalty is unscaled alpha * I on standardized columns, alpha = 0 is the
minimum-norm least-squares fit, and the nested pick is made on inner folds of
the outer training rows only.

The slow test (opt-in: PM_SLOW=1, needs mujoco + an offscreen GL) re-renders
the 512 loud-world materials and checks the Mac's Step 8 friction values on
its 3.9.0 photos to within 0.03 (16-view mean 0.612 / 0.597, concat
0.760 / 0.616, old / above-floor layout). The above layout is skipped where
samples.above_floor_camera_angles (PR #48) is absent.
"""

from __future__ import annotations

import os
import sys

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import specular_view_diag as S  # noqa: E402


def _std(x):
    sd = x.std(0)
    return (x - x.mean(0)) / np.where(sd > 1e-12, sd, 1.0)


def test_ridge_path_matches_primal_and_dual_with_unscaled_penalty():
    rng = np.random.default_rng(0)
    for n, p in ((60, 20), (40, 150)):  # n > p (primal) and p > n (dual)
        x, y, xt = rng.normal(size=(n, p)), rng.normal(size=n), rng.normal(size=(7, p))
        xs, ym = _std(x), y.mean()
        mu, sd = x.mean(0), x.std(0)
        xts = (xt - mu) / sd
        for a in (1e-3, 0.5, 30.0):
            w = np.linalg.solve(xs.T @ xs + a * np.eye(p), xs.T @ (y - ym))
            wd = xs.T @ np.linalg.solve(xs @ xs.T + a * np.eye(n), y - ym)
            got = S.ridge_path(x, y, xt, [a])[0]
            assert np.allclose(got, xts @ w + ym, atol=1e-8)
            assert np.allclose(got, xts @ wd + ym, atol=1e-8)


def test_alpha_zero_is_minimum_norm_least_squares():
    rng = np.random.default_rng(1)
    x, y, xt = rng.normal(size=(30, 80)), rng.normal(size=30), rng.normal(size=(5, 80))
    xs = _std(x)
    w = np.linalg.lstsq(xs, y - y.mean(), rcond=None)[0]
    xts = (xt - x.mean(0)) / x.std(0)
    assert np.allclose(S.ridge_path(x, y, xt, [0.0])[0], xts @ w + y.mean(), atol=1e-8)


def test_constant_columns_do_not_blow_up():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 6))
    x[:, 2] = 3.0
    y = x[:, 0] + 0.1 * rng.normal(size=50)
    pred = S.ridge_path(x, y, x[:5], [0.0, 1.0])
    assert np.isfinite(pred).all()


def test_penalty_is_not_scaled_by_n():
    """The original bug: alpha * n made the weakest penalty ~400x too strong.
    A faint (low-variance) column carrying the signal must survive alpha = 1e-3."""
    rng = np.random.default_rng(3)
    n = 400
    big = rng.normal(size=(n, 50))
    faint = 0.01 * rng.normal(size=n)
    x = np.column_stack([big, faint + 1e-4 * big[:, :1].ravel()])
    y = faint / faint.std()
    res = S.cv_readout(x, y, alphas=[1e-3], fold_seeds=(0,))
    assert res["nested"] > 0.9


def test_nested_pick_is_interior_and_beats_extremes_on_noisy_target():
    rng = np.random.default_rng(4)
    n, p = 300, 120
    x = rng.normal(size=(n, p))
    beta = rng.normal(size=p) / np.sqrt(p)
    y = x @ beta + 1.0 * rng.normal(size=n)
    alphas = (0.0, 1e-3, 1e-1, 10.0, 100.0, 1e3, 1e5)
    res = S.cv_readout(x, y, alphas=alphas, fold_seeds=(0, 1))
    assert len(res["picks"]) == 2 * 5
    assert all(a not in (0.0, 1e5) for a in res["picks"])
    assert res["nested"] > max(res["by_alpha"][0], res["by_alpha"][-1])
    assert res["best_of_grid"] >= res["nested"] - 0.05


def test_null_target_reads_about_zero_or_below():
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(200, 40)), rng.normal(size=200)
    assert S.cv_readout(x, y, fold_seeds=(0,))["nested"] < 0.05


def test_block_features_shape_and_means():
    img = np.arange(8 * 8 * 3, dtype=float).reshape(8, 8, 3)
    f = S.block_features(img, 4)
    assert f.shape == (4 * 4 * 3,)
    assert np.isclose(f[0], img[:2, :2, 0].mean())


def test_coarsen_equals_direct_block_means():
    rng = np.random.default_rng(6)
    img = rng.random((128, 128, 3))
    f16 = S.block_features(img, 16)
    assert np.allclose(S.coarsen(f16, 16, 8), S.block_features(img, 8))
    stack = np.stack([f16, f16 * 0.5])[None]  # leading (views, scenes) dims
    assert S.coarsen(stack, 16, 8).shape == (1, 2, 8 * 8 * 3)


def test_sphere_layout_matches_render_views_driving():
    from pseudomarble.data import samples

    ang = S.layout_angles("sphere")
    assert len(ang) == 16
    want = samples.pose_to_az_el_dist(samples.fibonacci_sphere_poses(16, 2.6)[0])[:2]
    assert ang[0] == want
    # views 0 and 15 are the mirror-image level poles (az +-90, el 0)
    assert abs(abs(ang[0][0]) - 90) < 1e-9 and abs(ang[0][1]) < 1e-9
    assert abs(abs(ang[15][0]) - 90) < 1e-9 and abs(ang[15][1]) < 1e-9


# The Mac's Step 8 nested friction values on its 3.9.0 photos (runs/postfix/
# perview.json at 16x16; perview_block8.json at 8x8). The 8x8 cells are the
# regime CROSSVIEW_POOLING section 1 rests on, since the trained encoder pools
# each view over H x W before the view mean.
STEP8 = {16: {"sphere": {"mean": 0.612, "concat": 0.760},
              "above": {"mean": 0.597, "concat": 0.616}},
         8: {"sphere": {"mean": 0.107, "concat": 0.717},
             "above": {"mean": 0.005, "concat": 0.630}}}


@pytest.mark.skipif(os.environ.get("PM_SLOW") != "1", reason="slow: set PM_SLOW=1")
@pytest.mark.parametrize("layout", ["sphere", "above"])
def test_reproduces_step8_friction_within_003(layout):
    pytest.importorskip("mujoco")
    from pseudomarble.data import samples

    if layout == "above" and not hasattr(samples, "above_floor_camera_angles"):
        pytest.skip("above layout lands with PR #48")
    mats = S.sample_materials(512, 1234, 1.0, 8.0)
    f16 = S.render_features(mats, {layout: S.layout_angles(layout)})[layout]
    fr = S.targets(mats)["friction"]
    alphas = tuple(10.0 ** k for k in range(-3, 7))  # the Mac's Step 8 grid
    for block, feats in ((16, f16), (8, S.coarsen(f16, 16, 8))):
        xm = feats.mean(0)
        xc = feats.transpose(1, 0, 2).reshape(feats.shape[1], -1)
        got = {"mean": S.cv_readout(xm, fr, alphas=alphas)["nested"],
               "concat": S.cv_readout(xc, fr, alphas=alphas)["nested"]}
        print(f"[step8] {layout} {block}x{block} {got}")
        for pool, want in STEP8[block][layout].items():
            assert abs(got[pool] - want) <= 0.03, (layout, block, pool, got[pool], want)
    # The 8x8 gap (concat minus mean) is what section 1 rests on; at 16x16 the
    # above-layout gap (0.019) sits inside the tolerance, at 8x8 it doesn't.
    want_gap = STEP8[8][layout]["concat"] - STEP8[8][layout]["mean"]
    assert got["concat"] - got["mean"] >= want_gap - 0.06
