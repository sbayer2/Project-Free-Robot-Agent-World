"""PyTorch CPU backend — the in-sandbox *trainable* stand-in.

``numpy_net`` proves the forward pass; this proves **training works** — that
gradients flow through the encoder + heads and the loss actually decreases —
without needing the Mac. It mirrors ``mlx_net`` / ``numpy_net`` (same architecture
from the same ``ModelConfig``); it is NOT the canonical trainer (that's
``train.py`` on MLX/Metal), it's a CPU de-risking tool.

PyTorch uses NCHW; the external interface stays NHWC (B, N, H, W, C) like the
other backends, so inputs/outputs match. Torch is import-guarded.

Quick in-session smoke (overfits a tiny synthetic batch; loss should plummet)::

    python -m pseudomarble.models.torch_net
"""

from __future__ import annotations

from typing import Dict, List

from pseudomarble.config import ModelConfig

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore

    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAVE_TORCH = False


def _require_torch() -> None:
    if not _HAVE_TORCH:
        raise RuntimeError("torch is not installed; `pip install torch` (CPU is fine)")


if _HAVE_TORCH:

    class _MLP(nn.Module):
        def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_hidden)
            self.fc2 = nn.Linear(d_hidden, d_out)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    class TorchModel(nn.Module):
        """Encoder + behavior head (+ aux essence head), mirroring mlx_net."""

        def __init__(self, cfg: ModelConfig = ModelConfig()) -> None:
            super().__init__()
            self.cfg = cfg
            convs = []
            cin = cfg.in_channels
            for cout in cfg.conv_channels:
                convs.append(nn.Conv2d(cin, cout, cfg.conv_kernel, stride=cfg.conv_stride))
                cin = cout
            self.convs = nn.ModuleList(convs)
            self.proj = nn.Linear(cin, cfg.latent_dim)
            if cfg.latent_trits > 0:  # FSQ bottleneck (mirrors mlx_net)
                self.bottleneck_down = nn.Linear(cfg.latent_dim, cfg.latent_trits)
                self.bottleneck_up = nn.Linear(cfg.latent_trits, cfg.latent_dim)
            self.behavior = _MLP(cfg.latent_dim, cfg.behavior_head_width, cfg.behavior_dim)
            self.essence = _MLP(cfg.latent_dim, cfg.essence_head_width, cfg.essence_dim)
            if cfg.appearance_weight > 0:  # F20 aux head (mirrors mlx_net; gated)
                self.appearance = _MLP(cfg.latent_dim, cfg.appearance_head_width,
                                       cfg.appearance_dim)

            # Render decoder: z -> seed map -> (upsample + conv)*k -> RGB.
            from pseudomarble.config import num_upsample_steps
            ch, s = cfg.render_channels, cfg.render_seed
            self._seed_ch, self._seed_s = ch, s
            self.seed = nn.Linear(cfg.latent_dim, ch * s * s)
            self.dec_convs = nn.ModuleList(
                [nn.Conv2d(ch, ch, 3, padding=1) for _ in range(num_upsample_steps(cfg))]
            )
            self.dec_final = nn.Conv2d(ch, 3, 3, padding=1)

        def encode(self, images):
            # images: (B, N, H, W, C) NHWC -> (B*N, C, H, W) NCHW for conv.
            B, N = images.shape[0], images.shape[1]
            x = images.reshape(B * N, *images.shape[2:]).permute(0, 3, 1, 2)
            for conv in self.convs:
                x = torch.relu(conv(x))
            x = x.mean(dim=(2, 3))                     # global average pool
            z = torch.relu(self.proj(x))              # per-view latent
            return z.reshape(B, N, -1).mean(dim=1)    # pool over views

        def decode(self, z):
            # z -> (B, ch, s, s) NCHW -> upsample/conv -> (B, S, S, 3) NHWC.
            x = torch.relu(self.seed(z)).reshape(z.shape[0], self._seed_ch,
                                                 self._seed_s, self._seed_s)
            for conv in self.dec_convs:
                x = torch.relu(conv(F.interpolate(x, scale_factor=2, mode="nearest")))
            x = torch.sigmoid(self.dec_final(x))
            return x.permute(0, 2, 3, 1)

        def bottleneck(self, z):
            """(code, expanded z); identity when off. 3-level FSQ with a
            straight-through gradient (mirrors mlx_net)."""
            if self.cfg.latent_trits <= 0:
                return None, z
            c = torch.tanh(self.bottleneck_down(z))
            code = c + (torch.round(c) - c).detach()   # STE
            return code, torch.relu(self.bottleneck_up(code))

        def forward(self, images) -> Dict:
            code, z = self.bottleneck(self.encode(images))
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
    _require_torch()
    return TorchModel(cfg)


def loss_fn(out: Dict, behavior_t, essence_t, cfg: ModelConfig, render_t=None,
            appearance_t=None):
    """behavior_weight*behavior MSE + essence_weight*essence MSE
    (+ render_weight*recon MSE) (+ appearance_weight*appearance MSE).

    ``render_t`` is the mean-view target (B, image_size, image_size, 3); when
    omitted the render term is skipped (e.g. behavior-only checks). ``appearance_t``
    is the 8-dim material-channel target (F20); added only when appearance_weight>0
    and the target is supplied. Per-head weights mirror cfg so this matches the MLX
    objective exactly."""
    _require_torch()
    b = torch.mean((out["behavior"] - behavior_t) ** 2)
    e = torch.mean((out["essence"] - essence_t) ** 2)
    loss = cfg.behavior_weight * b + cfg.essence_weight * e
    if render_t is not None:
        loss = loss + cfg.render_weight * torch.mean((out["render"] - render_t) ** 2)
    if cfg.appearance_weight > 0 and appearance_t is not None:
        loss = loss + cfg.appearance_weight * torch.mean(
            (out["appearance"] - appearance_t) ** 2)
    return loss


def overfit_smoke(cfg: ModelConfig, images, behavior_t, essence_t,
                  steps: int = 200, lr: float = 1e-3, seed: int = 0) -> List[float]:
    """Train on one fixed batch (incl. the render head); returns loss per step.
    Loss should fall sharply if gradients flow through the whole network."""
    _require_torch()
    torch.manual_seed(seed)
    model = TorchModel(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    render_t = images.mean(dim=1)  # mean over the N views = canonical appearance
    history: List[float] = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(images), behavior_t, essence_t, cfg, render_t)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return history


def _demo() -> None:  # pragma: no cover - manual smoke
    from dataclasses import replace
    _require_torch()
    cfg = replace(ModelConfig(), conv_channels=(8, 16), latent_dim=32,
                  behavior_head_width=32, essence_head_width=16, image_size=32)
    torch.manual_seed(0)
    images = torch.rand(4, 3, 32, 32, 3)               # B=4, N=3 views (32=4*2^3)
    behavior_t = torch.rand(4, cfg.behavior_dim)
    essence_t = torch.rand(4, cfg.essence_dim)
    hist = overfit_smoke(cfg, images, behavior_t, essence_t, steps=200)
    print(f"[torch smoke] loss {hist[0]:.4f} -> {hist[-1]:.5f} "
          f"(x{hist[0] / max(1e-9, hist[-1]):.1f} reduction)")


if __name__ == "__main__":
    _demo()
