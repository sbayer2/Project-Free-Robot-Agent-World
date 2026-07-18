"""Tests for the NumPy forward-reference model (the in-sandbox MLX stand-in).

Skips cleanly if numpy is unavailable; otherwise it actually runs the full
encode -> heads -> loss pipeline, validating the architecture composes and the
shapes flow exactly as the MLX model will on the Mac.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False

from pseudomarble.config import ModelConfig  # noqa: E402
from pseudomarble.models import losses  # noqa: E402

# A small architecture so the test is fast. image_size=16 keeps the render
# decoder cheap (16 = render_seed(4) * 2^2), which matters for the python-loop
# numpy conv.
SMALL = replace(
    ModelConfig(),
    conv_channels=(4, 8),
    latent_dim=16,
    behavior_head_width=16,
    essence_head_width=8,
    image_size=16,
)


def _skip_if_no_numpy():
    if not _HAVE_NUMPY:  # pragma: no cover
        print("numpy unavailable; skipping")
        return True
    return False


def test_forward_shapes():
    if _skip_if_no_numpy():
        return
    from pseudomarble.models.numpy_net import NumpyModel

    model = NumpyModel(SMALL, seed=0)
    images = np.random.default_rng(1).random((2, 3, 16, 16, 3))  # B=2, N=3 views
    out = model(images)
    assert out["z"].shape == (2, SMALL.latent_dim)
    assert out["behavior"].shape == (2, SMALL.behavior_dim)
    assert out["essence"].shape == (2, SMALL.essence_dim)
    assert out["render"].shape == (2, SMALL.image_size, SMALL.image_size, 3)


def test_appearance_head_gated_off_by_default():
    """F20: with appearance_weight=0 (default) the head does not exist and the
    output has no 'appearance' key — the architecture is byte-identical, so old
    checkpoints still load."""
    if _skip_if_no_numpy():
        return
    from pseudomarble.models.numpy_net import NumpyModel

    out = NumpyModel(SMALL, seed=0)(np.random.default_rng(1).random((2, 3, 16, 16, 3)))
    assert "appearance" not in out
    assert not hasattr(NumpyModel(SMALL, seed=0), "Wa1")


def test_appearance_head_on_when_weighted():
    """F20: appearance_weight>0 adds the z->appearance_params head, output (B, 8)."""
    if _skip_if_no_numpy():
        return
    from pseudomarble.models.numpy_net import NumpyModel

    cfg = replace(SMALL, appearance_weight=1.0)
    out = NumpyModel(cfg, seed=0)(np.random.default_rng(1).random((2, 3, 16, 16, 3)))
    assert out["appearance"].shape == (2, cfg.appearance_dim)
    assert cfg.appearance_dim == 8


def test_appearance_loss_term_added_only_when_weighted():
    """The pure-Python loss reference gains the appearance term iff weight>0."""
    if _skip_if_no_numpy():
        return
    pred = [[0.0] * 8, [0.0] * 8]
    tgt = [[1.0] * 8, [1.0] * 8]
    common = dict(behavior_pred=[[0.0]], behavior_target=[[0.0]],
                  essence_pred=[[0.0]], essence_target=[[0.0]])
    off = losses.combined_loss(**common, appearance_pred=pred, appearance_target=tgt,
                               appearance_weight=0.0)
    on = losses.combined_loss(**common, appearance_pred=pred, appearance_target=tgt,
                              appearance_weight=2.0)
    assert "appearance" not in off
    assert on["appearance"] == 1.0            # MSE of 0 vs 1
    assert on["total"] == off["total"] + 2.0  # weight * appearance MSE


def test_forward_is_deterministic():
    if _skip_if_no_numpy():
        return
    from pseudomarble.models.numpy_net import NumpyModel

    images = np.random.default_rng(2).random((2, 2, 16, 16, 3))
    a = NumpyModel(SMALL, seed=5)(images)["behavior"]
    b = NumpyModel(SMALL, seed=5)(images)["behavior"]
    assert np.allclose(a, b)


def test_pipeline_composes_with_loss():
    if _skip_if_no_numpy():
        return
    from pseudomarble.models.numpy_net import NumpyModel

    model = NumpyModel(SMALL, seed=0)
    images = np.random.default_rng(3).random((4, 3, 16, 16, 3))
    out = model(images)
    render_target = images.mean(axis=1)  # mean view
    result = losses.combined_loss(
        out["behavior"].tolist(), np.zeros((4, SMALL.behavior_dim)).tolist(),
        out["essence"].tolist(), np.zeros((4, SMALL.essence_dim)).tolist(),
        essence_weight=SMALL.essence_weight,
        render_pred=out["render"].tolist(), render_target=render_target.tolist(),
        render_weight=SMALL.render_weight,
    )
    assert result["total"] >= 0.0
    assert result["behavior"] >= 0.0
    assert result["render"] >= 0.0


def test_too_small_image_raises():
    if _skip_if_no_numpy():
        return
    from pseudomarble.models.numpy_net import NumpyModel

    model = NumpyModel(SMALL, seed=0)
    tiny = np.random.default_rng(4).random((1, 1, 4, 4, 3))  # can't survive 2 strided convs
    try:
        model(tiny)
    except ValueError as exc:
        assert "too small" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for too-small image")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
