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
            from pseudomarble.config import num_upsample_steps
            self.cfg = cfg
            self.encoder = Encoder(cfg)
            if cfg.latent_trits > 0:
                # FSQ bottleneck: latent_dim -> k ternary dims -> latent_dim.
                self.bottleneck_down = nn.Linear(cfg.latent_dim, cfg.latent_trits)
                self.bottleneck_up = nn.Linear(cfg.latent_trits, cfg.latent_dim)
            self.behavior = MLP(cfg.latent_dim, cfg.behavior_head_width, cfg.behavior_dim)
            self.essence = MLP(cfg.latent_dim, cfg.essence_head_width, cfg.essence_dim)
            if cfg.appearance_weight > 0:  # F20 aux; gated to keep default identical
                self.appearance = MLP(cfg.latent_dim, cfg.appearance_head_width,
                                      cfg.appearance_dim)

            # Render decoder: z -> seed map -> (upsample + conv)*k -> RGB (NHWC).
            ch, s = cfg.render_channels, cfg.render_seed
            self._seed_ch, self._seed_s = ch, s
            self.seed = nn.Linear(cfg.latent_dim, ch * s * s)
            self.dec_convs = [nn.Conv2d(ch, ch, 3, padding=1)
                              for _ in range(num_upsample_steps(cfg))]
            self.dec_final = nn.Conv2d(ch, 3, 3, padding=1)

        def decode(self, z):
            x = nn.relu(self.seed(z)).reshape(z.shape[0], self._seed_s, self._seed_s,
                                              self._seed_ch)
            for conv in self.dec_convs:
                x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)  # nearest 2x upsample
                x = nn.relu(conv(x))
            return mx.sigmoid(self.dec_final(x))

        def _fsq(self, h):
            """3-level FSQ: round(tanh(h)) in {-1,0,1}, straight-through grad."""
            c = mx.tanh(h)
            return c + mx.stop_gradient(mx.round(c) - c)

        def bottleneck(self, z):
            """(code, expanded z). Identity (None, z) when the bottleneck is off."""
            if self.cfg.latent_trits <= 0:
                return None, z
            code = self._fsq(self.bottleneck_down(z))
            return code, nn.relu(self.bottleneck_up(code))

        def __call__(self, images) -> Dict:
            code, z = self.bottleneck(self.encoder(images))
            out = {"z": z, "behavior": self.behavior(z),
                   "essence": self.essence(z), "render": self.decode(z)}
            if self.cfg.appearance_weight > 0:
                out["appearance"] = self.appearance(z)
            if code is not None:
                out["code"] = code
            return out

        # Convenience accessors used by the coherence harness (decode = render).
        def behavior_from_z(self, z):
            return self.behavior(z)

        def essence_from_z(self, z):
            return self.essence(z)

        def appearance_from_z(self, z):
            return self.appearance(z)

        def render_from_z(self, z):
            return self.decode(z)


def build_model(cfg: ModelConfig = ModelConfig()):
    """Construct the MLX model (raises a clear error if MLX is unavailable)."""
    _require_mlx()
    return PseudoMarbleModel(cfg)


def loss_fn(model, batch: Dict, cfg: ModelConfig):
    """Total = behavior_weight*behavior MSE + essence_weight*essence MSE
    + render_weight*recon MSE.

    ``batch`` holds mlx arrays: ``images`` (B,N,H,W,C), ``behavior`` (B,behavior_dim),
    ``essence`` (B,essence_dim). The render target is the mean over the N views
    (the pose-averaged canonical appearance); input H,W must equal cfg.image_size
    so it matches the decoder output. Per-head weights let one config train the
    shared model (all on) or the independent baseline's render-only / behavior-only
    models (zero the other heads).
    """
    _require_mlx()
    out = model(batch["images"])
    b = mx.mean((out["behavior"] - batch["behavior"]) ** 2)
    e = mx.mean((out["essence"] - batch["essence"]) ** 2)
    render_target = mx.mean(batch["images"], axis=1)  # (B, H, W, C)
    r = mx.mean((out["render"] - render_target) ** 2)
    total = cfg.behavior_weight * b + cfg.essence_weight * e + cfg.render_weight * r
    if cfg.appearance_weight > 0:  # F20 aux: force z to retain the material channels
        a = mx.mean((out["appearance"] - batch["appearance"]) ** 2)
        total = total + cfg.appearance_weight * a
    return total
