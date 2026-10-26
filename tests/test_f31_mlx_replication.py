"""Numpy-only cores of the F31a MLX replication tools and the tilt-bug oracle
(scripts/f31_measure_mlx.py, scripts/f31_head_diag_mlx.py,
scripts/tilt_bug_oracle.py). The MLX encode paths run on the Mac only.

Pins: held-out gain matches train_torch's definition (predicting the train
mean scores exactly 1); the logged per-epoch gain feeds f31_measure's escape
rule; the trajectory block slices match probes.trajectory_vector's layout;
a constant head reads as collapsed (gain 1, prediction std 0); and the
pre-fix target rebuild swaps only the tilt block.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import f31_head_diag_mlx as H  # noqa: E402
import f31_measure_mlx as M  # noqa: E402
import tilt_bug_oracle as T  # noqa: E402
from f31_measure import escape_epoch  # noqa: E402

from pseudomarble.probes import (  # noqa: E402
    OUTCOME_FIELDS,
    PROBE_ORDER,
    behavior_vector,
    trajectory_dim,
)


def _splits(n_train: int, n_test: int) -> np.ndarray:
    return np.array(["train"] * n_train + ["test"] * n_test)


def test_heldout_gain_train_mean_is_one_and_perfect_is_inf():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(30, 4))
    s = _splits(20, 10)
    mean_pred = np.tile(Y[s == "train"].mean(0), (30, 1))
    assert np.isclose(M.heldout_gain(Y, mean_pred, s), 1.0)
    assert M.heldout_gain(Y, Y.copy(), s) == float("inf")


def test_logged_gains_drive_the_escape_rule():
    hist = [{"epoch": 0, "behavior_mse": 0.10}, {"epoch": 1, "behavior_mse": 0.06},
            {"epoch": 2, "behavior_mse": 0.04}]
    rows = M.logged_gains(hist, baseline=0.10)
    assert [round(r["heldout_gain"], 3) for r in rows] == [1.0, 1.667, 2.5]
    assert escape_epoch(rows) == 2
    assert escape_epoch(M.logged_gains(hist[:2], baseline=0.10)) is None


def test_probe_blocks_match_trajectory_layout():
    b = H.probe_blocks(("drop", "tilt"), 16, True)
    assert b["drop"] == slice(0, 48) and b["tilt"] == slice(48, 96)
    assert b["all"].stop == trajectory_dim(16, ("drop", "tilt"), pos_only=True)
    assert H.probe_blocks(("drop",), 4, False)["all"].stop == trajectory_dim(4, ("drop",))


def test_constant_head_reads_as_collapsed():
    rng = np.random.default_rng(1)
    target = rng.normal(size=(50, 96))
    mu = target.mean(0)
    blocks = H.probe_blocks(("drop", "tilt"), 16, True)
    rep = H.block_report(target, np.tile(mu, (50, 1)), mu, blocks)
    assert np.isclose(rep["all_gain"], 1.0) and np.isclose(rep["tilt_pred_std"], 0.0)
    assert rep["tilt_target_std"] > 0.5


def test_dead_fraction():
    h = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    assert np.isclose(H.dead_fraction(h), 1 / 3)


def test_explained_share_and_tilt_block():
    rng = np.random.default_rng(2)
    x = rng.uniform(size=200)
    Y = np.stack([2 * x, rng.normal(size=200)], 1)
    X = np.stack([np.ones_like(x), x], 1)
    share = T.explained_share(Y, X)
    assert 0.0 < share < 1.0
    assert np.isclose(T.explained_share(Y[:, :1], X), 1.0)


def _outcome(v: float) -> dict:
    return {k: (False if k == "toppled" else v) for k in OUTCOME_FIELDS}


def test_prefix_targets_swaps_only_tilt(tmp_path):
    audit = {}
    for i, sid in enumerate(("train_000000", "test_000001")):
        probes = [{"probe": p, "outcome": _outcome(1.0 + i)} for p in PROBE_ORDER]
        (tmp_path / sid).mkdir()
        (tmp_path / sid / "sample.json").write_text(json.dumps({"behavior": {"probes": probes}}))
        audit[sid] = {"old": {"slid_distance": 0.004, "path_length": 0.01}}
    old = T.prefix_targets(str(tmp_path), audit)
    for i, sid in enumerate(sorted(audit)):
        rec = json.loads((tmp_path / sid / "sample.json").read_text())
        new = np.array(behavior_vector(rec["behavior"]["probes"], normalize=True))
        diff = np.flatnonzero(~np.isclose(old[i], new))
        tilt = PROBE_ORDER.index("tilt") * len(OUTCOME_FIELDS)
        changed = {OUTCOME_FIELDS[j - tilt] for j in diff}
        assert diff.min() >= tilt and diff.max() < tilt + len(OUTCOME_FIELDS)
        assert changed == {"slid_distance", "path_length"}
