"""Tests for the F31a trajectory-head collapse diagnostic (scripts/f31_head_diag.py)."""

import os
import sys

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import f31_head_diag as D  # noqa: E402


def test_probe_blocks_match_trajectory_vector_layout():
    from pseudomarble.probes import trajectory_dim

    b = D.probe_blocks(("drop", "tilt"), 16, pos_only=True)
    assert b["drop"] == slice(0, 48) and b["tilt"] == slice(48, 96)
    assert b["all"].stop == trajectory_dim(16, ("drop", "tilt"), pos_only=True)
    assert D.probe_blocks(("tilt",), 8, pos_only=False)["tilt"] == slice(0, 48)


def test_collapsed_head_reads_gain_one_and_zero_spread():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(200, 96))
    blocks = D.probe_blocks(("drop", "tilt"), 16, pos_only=True)
    mean = t.mean(0)
    r = D.block_report(t, np.tile(mean, (200, 1)), mean, blocks)
    assert abs(r["all_gain"] - 1.0) < 1e-9
    assert r["tilt_pred_std"] < 1e-12 and r["tilt_target_std"] > 0.9


def test_learning_head_reads_high_gain_and_matched_spread():
    rng = np.random.default_rng(1)
    t = rng.normal(size=(200, 96))
    blocks = D.probe_blocks(("drop", "tilt"), 16, pos_only=True)
    r = D.block_report(t, t + 0.1 * rng.normal(size=t.shape), t.mean(0), blocks)
    assert r["drop_gain"] > 50
    assert abs(r["drop_pred_std"] - r["drop_target_std"]) < 0.05


def test_offset_constant_scores_below_one():
    # the drop-block signature on the torch runs: constant, but off the train mean
    t = np.random.default_rng(2).normal(size=(100, 48))
    blocks = {"drop": slice(0, 48)}
    r = D.block_report(t, np.tile(t.mean(0) + 1.0, (100, 1)), t.mean(0), blocks)
    assert r["drop_gain"] < 0.6
