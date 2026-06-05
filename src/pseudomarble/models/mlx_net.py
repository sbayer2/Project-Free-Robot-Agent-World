"""MLX encoder + behavior/essence heads — the trainable model (runs on the Mac).

Mirrors ``models/numpy_net.py`` exactly (same layers from the same ``ModelConfig``),
but uses ``mlx.nn`` so it is differentiable and trains on Apple-silicon/Metal.
MLX is import-guarded: this module imports anywhere, but the network classes only
exist when ``mlx`` is installed, and ``build_model`` raises a clear error
otherwise. This keeps the package importable in CI / Linux sessions.

    from pseudomarble.models.mlx_net import build_model, loss_fn
    model = build_model(ModelConfig())
    out = model(images)            # {"z", "behavior", "essence"}
"""

from __future__ import annotations

from typing import Dict

from pseudomarble.config import ModelConfig

try:
    import mlx.core as mx  # type: ignore
    import mlx.nn as nn  # type: ignore

    _HAVE_MLX = True
except Exception:  # pragma: no cover - depends on runtime
    mx = None  # type: ignore
    nn = None  # type: ignore
    _HAVE_MLX = False


def _require_mlx() -> None:
    if not _HAVE_MLX:
        raise RuntimeError(
            "MLX is not available in this session (Apple-silicon only). The model "
            "is authored and shape-tested here; run training on the Mac with "
            "`pip install -e \".[mlx]\"`. For an in-session forward check use "
            "models.numpy_net.NumpyModel."
        )


if _HAVE_MLX:

    class Encoder(nn.Module):
        """Multi-view CNN -> shared latent z."""

        def __init__(self, cfg: ModelConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.convs = []
            cin = cfg.in_channels
            for cout in cfg.conv_channels:
                self.convs.append(nn.Conv2d(cin, cout, cfg.conv_kernel,
                                            stride=cfg.conv_stride))
                cin = cout
            self.proj = nn.Linear(cin, cfg.latent_dim)

        def __call__(self, images):
            # images: (B, N, H, W, C); fold views into the batch for the CNN.
            B, N = images.shape[0], images.shape[1]
            x = images.reshape(B * N, *images.shape[2:])
            for conv in self.convs:
                x = nn.relu(conv(x))
            x = x.mean(axis=(1, 2))                  # global average pool over H, W
            z = nn.relu(self.proj(x))                # per-view latent
            return z.reshape(B, N, -1).mean(axis=1)  # pool over the N views

    class MLP(nn.Module):
        def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_hidden)
            self.fc2 = nn.Linear(d_hidden, d_out)

        def __call__(self, x):
            return self.fc2(nn.relu(self.fc1(x)))

    class PseudoMarbleModel(nn.Module):
        """Encoder + behavior head (+ auxiliary essence head)."""

        def __init__(self, cfg: ModelConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.encoder = Encoder(cfg)
            self.behavior = MLP(cfg.latent_dim, cfg.behavior_head_width, cfg.behavior_dim)
            self.essence = MLP(cfg.latent_dim, cfg.essence_head_width, cfg.essence_dim)

        def __call__(self, images) -> Dict:
            z = self.encoder(images)
            return {"z": z, "behavior": self.behavior(z), "essence": self.essence(z)}


def build_model(cfg: ModelConfig = ModelConfig()):
    """Construct the MLX model (raises a clear error if MLX is unavailable)."""
    _require_mlx()
    return PseudoMarbleModel(cfg)


def loss_fn(model, batch: Dict, cfg: ModelConfig):
    """Total loss = behavior MSE + essence_weight * essence MSE (mirrors losses.py).

    ``batch`` holds mlx arrays: ``images`` (B,N,H,W,C), ``behavior`` (B,behavior_dim),
    ``essence`` (B,essence_dim).
    """
    _require_mlx()
    out = model(batch["images"])
    b = mx.mean((out["behavior"] - batch["behavior"]) ** 2)
    e = mx.mean((out["essence"] - batch["essence"]) ** 2)
    return b + cfg.essence_weight * e
