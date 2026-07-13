"""Tests for the FSQ latent bottleneck (F17): config plumbing, the ternary
code contract in the numpy mirror, and the straight-through gradient in torch.

numpy tests need numpy; torch tests skip cleanly when torch is absent.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

np = pytest.importorskip("numpy")

from dataclasses import replace  # noqa: E402

from pseudomarble.config import ModelConfig  # noqa: E402
from pseudomarble.models.numpy_net import NumpyModel  # noqa: E402
from pseudomarble.models.train import parse_args  # noqa: E402

TINY = replace(ModelConfig(), conv_channels=(4,), latent_dim=16, image_size=8,
               render_seed=4, render_channels=4, behavior_head_width=8,
               essence_head_width=8)


def _images(b=3, n=2, hw=8):
    rng = np.random.default_rng(0)
    return rng.random((b, n, hw, hw, 3), dtype=np.float32)


def test_default_is_off_and_identical_shapes():
    assert ModelConfig().latent_trits == 0
    out = NumpyModel(TINY, seed=0)(_images())
    assert "code" not in out
    assert out["z"].shape == (3, 16)


def test_numpy_code_is_ternary_and_bounded():
    cfg = replace(TINY, latent_trits=4)
    out = NumpyModel(cfg, seed=0)(_images(b=6))
    code = out["code"]
    assert code.shape == (6, 4)
    assert set(np.unique(code)) <= {-1.0, 0.0, 1.0}
    assert out["z"].shape == (6, 16)          # heads still see latent_dim
    assert out["behavior"].shape[1] == cfg.behavior_dim


def test_information_bound_distinct_codes():
    # k trits admit at most 3^k distinct codes; with k=1 that's 3.
    cfg = replace(TINY, latent_trits=1)
    out = NumpyModel(cfg, seed=0)(_images(b=32))
    distinct = {tuple(row) for row in out["code"]}
    assert len(distinct) <= 3


def test_train_and_coherence_args_plumb():
    args = parse_args(["--data", "d", "--latent-trits", "6"])
    assert args.latent_trits == 6
    assert parse_args(["--data", "d"]).latent_trits is None


def test_torch_straight_through_gradient_flows():
    torch = pytest.importorskip("torch")
    from pseudomarble.models.torch_net import build_model

    cfg = replace(TINY, latent_trits=4)
    model = build_model(cfg)
    imgs = torch.rand(2, 2, 8, 8, 3)
    out = model(imgs)
    assert set(torch.unique(out["code"]).tolist()) <= {-1.0, 0.0, 1.0}
    # Rounding has zero gradient a.e.; the STE must still pass gradient back
    # through the bottleneck projection.
    out["behavior"].sum().backward()
    g = model.bottleneck_down.weight.grad
    assert g is not None and float(g.abs().sum()) > 0.0
