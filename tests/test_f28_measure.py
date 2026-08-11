"""Tests for the F28 directional-coherence driver's geometry
(scripts/f28_measure.py).

Numpy-only core; the MLX encode/Jacobian layer is exercised on the Mac (the
vjp test skips cleanly where MLX is absent). Encodes the preregistration's
section 2.3 requirement: a synthetic-case proof that ridge-probe directions
(already standardized-X) and raw-z Jacobians (times train-split std) land in
the SAME coordinate system before any cosine is taken -- with anisotropic
scales chosen so that forgetting or inverting the std conversion fails the
test. Also pins the committed section-3 pairing table and the report schema
the frozen verdicts script reads.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from f28_measure import (  # noqa: E402
    BEHAVIOR_FIELDS,
    CHANNELS,
    ESSENCE_AXES,
    MATCHED_BEHAVIOR,
    MATCHED_ESSENCE,
    SIGNS_BEHAVIOR,
    SIGNS_ESSENCE,
    alignment_matrix,
    arm_record,
    channel_targets,
    directional_D,
    probe_directions,
    ridge_weights,
    stitch_predict,
    transfer_directions,
    unit_rows,
    variance_weighted_r2,
)
from oracle_ceiling import ridge_fit_predict  # noqa: E402

from pseudomarble.probes import behavior_field_names  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_committed_pairing_table_is_the_frozen_one():
    """The section-3 commitment as amended, pinned: any edit to the pairing
    must fail here. Amendment 3 (2026-08-11) dropped roughness<->friction as
    VOID-BY-LEGIBILITY; the two surviving cells are metallic<->density (+)
    and value<->restitution (-), friction kept as mismatched diagnostic."""
    assert CHANNELS == ("metallic", "value")
    assert ESSENCE_AXES == ("density", "friction", "restitution")
    assert MATCHED_ESSENCE == ((0, 0), (1, 2))
    assert SIGNS_ESSENCE == (1.0, -1.0)
    assert BEHAVIOR_FIELDS == ("push.path_length", "drop.n_bounces")
    assert MATCHED_BEHAVIOR == ((0, 0), (1, 1))
    assert SIGNS_BEHAVIOR == (-1.0, -1.0)
    names = behavior_field_names()
    for f in BEHAVIOR_FIELDS:  # addressed by name; must exist in the vector
        assert f in names


def test_channel_targets_pick_the_committed_columns():
    Ya = np.zeros((2, 8), np.float32)
    Ya[0] = [0.2, 0.7, 0.4, 1.0, 0.31, 0.62, 0.0, 0.5]  # RGB max = 0.7 (G)
    Ya[1] = [0.9, 0.1, 0.1, 1.0, 0.11, 0.22, 0.0, 0.5]  # RGB max = 0.9 (R)
    Y = channel_targets(Ya)
    assert Y.shape == (2, 2)
    np.testing.assert_allclose(Y[:, 0], [0.62, 0.22])  # metallic = col 5
    np.testing.assert_allclose(Y[:, 1], [0.7, 0.9])    # value = max(RGB)


def test_ridge_weights_match_ridge_fit_predict():
    """The extracted weights must reproduce oracle_ceiling.ridge_fit_predict
    exactly -- the prereg names that function as the probe."""
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.normal(size=(60, 9)), rng.normal(size=(25, 9))
    Ytr = rng.normal(size=(60, 3))
    W, mu, sd = ridge_weights(Xtr, Ytr)
    A = np.concatenate([(Xte - mu) / sd, np.ones((len(Xte), 1))], axis=1)
    np.testing.assert_allclose(A @ W, ridge_fit_predict(Xtr, Ytr, Xte),
                               atol=1e-10)


def test_standardized_coordinate_composition_on_synthetic_case():
    """THE section-2.3 required test. Anisotropic z (per-dim scales spanning
    four orders of magnitude), an exact linear channel y = z @ a, and a linear
    'head' with Jacobian row b. The probe direction comes out in
    standardized-X coordinates (prop. to a*sd); the raw Jacobian must be
    multiplied by sd to land in the same system. The mean cosine must equal
    the analytic cos(a*sd, b*sd) -- and must NOT equal the raw cos(a, b), so
    dropping or inverting the conversion fails loudly."""
    rng = np.random.default_rng(1)
    L, n = 12, 400
    sd_true = 10.0 ** rng.uniform(-2, 2, size=L)
    Z = rng.normal(size=(n, L)) * sd_true + rng.normal(size=L)
    a, b = rng.normal(size=L), rng.normal(size=L)
    y = Z @ a  # exact linear channel, no noise

    dirs, _mu, sd = probe_directions(Z, y[:, None])
    np.testing.assert_allclose(sd, Z.std(0), rtol=1e-12)
    jac_raw = np.tile(b, (n, 1))[None]  # (P=1, B, L): linear head, constant rows
    M = alignment_matrix(dirs, jac_raw, sd)

    expected = _cos(a * sd, b * sd)
    assert abs(M[0, 0] - expected) < 1e-4
    # teeth: with these scales the raw-coordinate cosine is a different number,
    # so a driver that skips (or inverts) the std conversion cannot pass.
    assert abs(expected - _cos(a, b)) > 0.05
    assert abs(expected - _cos(a / sd_true, b * sd_true)) > 0.05


def test_directional_D_signs_and_mismatched_cells():
    """2x3 alignment matrix, Amendment-3 matched cells (0,0) and (1,2):
    D averages those two with committed signs; the other four cells feed
    mismatched_abs only."""
    M = np.array([[0.5, 0.9, -0.9],
                  [0.9, -0.9, -0.6]])
    D, mism = directional_D(M, ((0, 0), (1, 2)), (1.0, -1.0))
    assert abs(D - (0.5 + 0.6) / 2) < 1e-12
    assert abs(mism - 0.9) < 1e-12  # four unmatched cells, all |0.9|


def test_stitch_transfer_matches_analytic_composition():
    """Linear world: z_ro = z_bo @ T exactly, channel y = z_ro @ a. The
    transferred direction must equal the gradient of y w.r.t. standardized
    z_bo, i.e. unit((T @ a) * sd_bo), and the stitch R^2 must be ~1."""
    rng = np.random.default_rng(2)
    n, L_bo, L_ro = 300, 10, 7
    Zbo = rng.normal(size=(n, L_bo)) * 10.0 ** rng.uniform(-1, 1, size=L_bo)
    T = rng.normal(size=(L_bo, L_ro))
    Zro = Zbo @ T
    a = rng.normal(size=L_ro)
    tr, te = np.arange(n) < 200, np.arange(n) >= 200

    dirs_ro, _mu_ro, sd_ro = probe_directions(Zro[tr], (Zro @ a)[tr, None])
    W_st, mu_bo, sd_bo = ridge_weights(Zbo[tr], Zro[tr])
    r2 = variance_weighted_r2(Zro[te], stitch_predict(W_st, mu_bo, sd_bo, Zbo[te]))
    assert r2 > 0.999

    got = transfer_directions(dirs_ro, sd_ro, W_st)[0]
    want = unit_rows((T @ a) * sd_bo)
    assert abs(abs(_cos(got, want)) - 1.0) < 1e-3
    assert _cos(got, want) > 0  # orientation preserved, not just the axis


def test_variance_weighted_r2_bounds():
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(50, 4))
    assert variance_weighted_r2(Y, Y) == pytest.approx(1.0)
    assert variance_weighted_r2(Y, np.tile(Y.mean(0), (50, 1))) == pytest.approx(0.0)


def test_report_schema_feeds_the_frozen_verdicts_script(tmp_path):
    """A synthetic report assembled with the driver's own arm_record() must run
    through scripts/f28_verdicts.py (frozen one commit earlier) end to end."""
    gates = {"latent_pr_min": 24.0, "probe_r2_min_trained": 0.55,
             "probe_r2_min_untrained": 0.30, "jac_norm_median_min": 5e-3}
    r = {"ctrl": 0.008, "base": 0.263, "g2": 0.809, "loud": 0.992}
    mk = {
        "ctrl": ([0.01, -0.02, 0.015], 0.01),
        "base": ([0.05, 0.08, 0.06], 0.04),
        "g2": ([0.22, 0.25, 0.20], 0.15),
        "loud": ([0.30, 0.34, 0.31], 0.24),
    }
    arms = {tag: arm_record(r[tag], dict(gates), ld, [x / 2 for x in ld], 0.08,
                            {"D": dj, "stitch_r2": 0.9, "probe_r2_min": 0.5})
            for tag, (ld, dj) in mk.items()}
    rep = tmp_path / "f28_report.json"
    rep.write_text(json.dumps({"apparatus": {"r": r}, "arms": arms}))

    out = subprocess.run(
        [sys.executable, os.path.join("scripts", "f28_verdicts.py"),
         "--report", str(rep)],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert "GATES: ALL PASS" in out.stdout
    assert "H1" in out.stdout and "CLEAN" in out.stdout
    assert "H2" in out.stdout and "CONTENT" in out.stdout
    assert "H4" in out.stdout


def test_head_jacobian_rows_exact_on_linear_head():
    """MLX-only: on a purely linear head the batched one-hot-cotangent vjp
    rows are known analytically -- every scene's row for output p must equal
    column p of the weight matrix. This pins the machinery head_jacobian_rows
    adds (batching, cotangent construction, stacking) with no autodiff
    subtlety in the way. Skips where MLX is unavailable (Linux sessions)."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from f28_measure import head_jacobian_rows

    rng = np.random.default_rng(4)
    Wm = rng.normal(size=(64, 3)).astype(np.float32)
    z = rng.normal(size=(5, 64)).astype(np.float32)
    head = lambda zz: zz @ mx.array(Wm)  # noqa: E731

    jac = head_jacobian_rows(head, z, (0, 1, 2), 3)
    assert jac.shape == (3, 5, 64)
    # Tolerance is set by the platform, not the math: MLX GPU matmul carries
    # ~1e-3 relative error (reduced-precision accumulation; a plain one-hot
    # row-copy matmul shows the same error). Negligible against the frozen
    # +/-0.05 decision bands; an indexing/cotangent bug is order-1 here.
    for p in range(3):
        np.testing.assert_allclose(jac[p], np.tile(Wm[:, p], (5, 1)),
                                   rtol=5e-3, atol=1e-3)


def test_head_jacobian_rows_consistent_with_jvp_on_real_head():
    """MLX-only: on the real (piecewise-linear ReLU) essence head, verify the
    vjp rows against forward-mode jvp: for a random tangent v,
    row(p, i) . v must equal the jvp's output (i, p). Both sides are the same
    Jacobian under the same subgradient convention, so this holds even at ReLU
    kinks -- where a finite-difference check does not (the kink is why this
    test is jvp-based). Agreement is bounded by the platform's ~1e-3
    relative matmul error (see the linear-head test), not by the math."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from f28_measure import head_jacobian_rows

    from pseudomarble.config import ModelConfig
    from pseudomarble.models.mlx_net import build_model

    mx.random.seed(7)
    m = build_model(ModelConfig(image_size=32))
    rng = np.random.default_rng(5)
    z = rng.normal(size=(5, 256)).astype(np.float32)
    tang = rng.normal(size=(5, 256)).astype(np.float32)

    jac = head_jacobian_rows(m.essence_from_z, z, (0, 1, 2), 3)
    assert jac.shape == (3, 5, 256)

    _, jvps = mx.jvp(m.essence_from_z, [mx.array(z)], [mx.array(tang)])
    jv = np.array(jvps[0].tolist(), np.float32)  # (5, 3): row i = J_i @ v_i
    for p in range(3):
        got = np.einsum("bl,bl->b", jac[p].astype(np.float64),
                        tang.astype(np.float64))
        np.testing.assert_allclose(got, jv[:, p], rtol=2e-2, atol=2e-2)
