"""Tests for the PyTorch CPU backend (the in-sandbox trainable stand-in).

Skips cleanly without torch; otherwise it runs a real optimizer loop and asserts
the loss drops — verifying gradients flow through encoder + heads before any Mac
time is spent. Also checks output shapes match the numpy reference.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

from pseudomarble.config import ModelConfig  # noqa: E402

SMALL = replace(ModelConfig(), conv_channels=(4, 8), latent_dim=16,
                behavior_head_width=16, essence_head_width=8, image_size=16)


def _skip():
    if not _HAVE_TORCH:  # pragma: no cover
        print("torch unavailable; skipping")
        return True
    return False


def test_forward_shapes():
    if _skip():
        return
    from pseudomarble.models.torch_net import build_model

    model = build_model(SMALL)
    images = torch.rand(2, 3, 16, 16, 3)  # B=2, N=3 views, NHWC
    out = model(images)
    assert out["z"].shape == (2, SMALL.latent_dim)
    assert out["behavior"].shape == (2, SMALL.behavior_dim)
    assert out["essence"].shape == (2, SMALL.essence_dim)
    assert tuple(out["render"].shape) == (2, SMALL.image_size, SMALL.image_size, 3)


def test_training_reduces_loss():
    if _skip():
        return
    from pseudomarble.models.torch_net import overfit_smoke

    torch.manual_seed(0)
    images = torch.rand(4, 2, 16, 16, 3)
    behavior_t = torch.rand(4, SMALL.behavior_dim)
    essence_t = torch.rand(4, SMALL.essence_dim)
    hist = overfit_smoke(SMALL, images, behavior_t, essence_t, steps=150, lr=1e-3)
    assert hist[-1] < hist[0], "loss did not decrease"
    assert hist[-1] < 0.5 * hist[0], f"loss barely moved: {hist[0]:.3f} -> {hist[-1]:.3f}"


def test_init_is_deterministic():
    if _skip():
        return
    from pseudomarble.models.torch_net import overfit_smoke

    images = torch.rand(3, 2, 16, 16, 3)
    bt, et = torch.rand(3, SMALL.behavior_dim), torch.rand(3, SMALL.essence_dim)
    a = overfit_smoke(SMALL, images, bt, et, steps=3, seed=7)
    b = overfit_smoke(SMALL, images, bt, et, steps=3, seed=7)
    assert abs(a[0] - b[0]) < 1e-9


def test_shapes_match_numpy_backend():
    if _skip():
        return
    try:
        import numpy as np
        from pseudomarble.models.numpy_net import NumpyModel
    except Exception:  # pragma: no cover
        print("numpy unavailable; skipping cross-backend check")
        return
    from pseudomarble.models.torch_net import build_model

    images = np.random.default_rng(0).random((2, 2, 16, 16, 3)).astype("float32")
    n_out = NumpyModel(SMALL, seed=0)(images)
    t_out = build_model(SMALL)(torch.from_numpy(images))
    assert tuple(t_out["behavior"].shape) == n_out["behavior"].shape
    assert tuple(t_out["essence"].shape) == n_out["essence"].shape
    assert tuple(t_out["render"].shape) == n_out["render"].shape


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
