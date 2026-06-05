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

# A small architecture so the test is fast (3 conv layers wouldn't fit 16x16).
SMALL = replace(
    ModelConfig(),
    conv_channels=(4, 8),
    latent_dim=16,
    behavior_head_width=16,
    essence_head_width=8,
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
    behavior_target = np.zeros((4, SMALL.behavior_dim))
    essence_target = np.zeros((4, SMALL.essence_dim))
    result = losses.combined_loss(
        out["behavior"].tolist(), behavior_target.tolist(),
        out["essence"].tolist(), essence_target.tolist(),
        essence_weight=SMALL.essence_weight,
    )
    assert result["total"] >= 0.0
    assert result["behavior"] >= 0.0


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
