"""Tests for the CPU torch trainer (models/train_torch.py)."""

import pytest

from pseudomarble.models import train_torch as T


def test_batches_cover_every_index_once_and_shuffle_is_seeded():
    b = T.batches(10, 4, shuffle=True, seed=3)
    flat = [i for chunk in b for i in chunk]
    assert sorted(flat) == list(range(10))
    assert [len(c) for c in b] == [4, 4, 2]
    assert b == T.batches(10, 4, shuffle=True, seed=3)
    assert T.batches(10, 4, shuffle=False, seed=0)[0] == [0, 1, 2, 3]


def test_trainer_shares_train_py_flags():
    # One parser for both trainers: a flag cannot mean different things.
    from pseudomarble.models import train
    assert T.parse_args is train.parse_args and T.make_config is train.make_config


def test_heldout_gain_and_participation_ratio():
    torch = pytest.importorskip("torch")
    tr = torch.tensor([[0.0], [2.0]])
    te = torch.tensor([[1.0], [3.0]])
    # baseline MSE vs train-mean 1.0: ((1-1)^2 + (3-1)^2)/2 = 2; model MSE 0.25
    assert T.heldout_gain(tr, te, te.clone() + 0.5) == pytest.approx(2.0 / 0.25)
    assert T.heldout_gain(tr, te, te.clone()) == float("inf")
    z = torch.zeros(8, 4)
    z[:, 0] = torch.arange(8.0)
    assert T.participation_ratio(z) == pytest.approx(1.0)      # one live dim
    assert T.participation_ratio(torch.randn(500, 4)) > 3.5    # four live dims
