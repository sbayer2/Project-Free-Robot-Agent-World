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
            self.behavior = _MLP(cfg.latent_dim, cfg.behavior_head_width, cfg.behavior_dim)
            self.essence = _MLP(cfg.latent_dim, cfg.essence_head_width, cfg.essence_dim)

        def encode(self, images):
            # images: (B, N, H, W, C) NHWC -> (B*N, C, H, W) NCHW for conv.
            B, N = images.shape[0], images.shape[1]
            x = images.reshape(B * N, *images.shape[2:]).permute(0, 3, 1, 2)
            for conv in self.convs:
                x = torch.relu(conv(x))
            x = x.mean(dim=(2, 3))                     # global average pool
            z = torch.relu(self.proj(x))              # per-view latent
            return z.reshape(B, N, -1).mean(dim=1)    # pool over views

        def forward(self, images) -> Dict:
            z = self.encode(images)
            return {"z": z, "behavior": self.behavior(z), "essence": self.essence(z)}


def build_model(cfg: ModelConfig = ModelConfig()):
    _require_torch()
    return TorchModel(cfg)


def loss_fn(out: Dict, behavior_t, essence_t, essence_weight: float):
    _require_torch()
    b = torch.mean((out["behavior"] - behavior_t) ** 2)
    e = torch.mean((out["essence"] - essence_t) ** 2)
    return b + essence_weight * e


def overfit_smoke(cfg: ModelConfig, images, behavior_t, essence_t,
                  steps: int = 200, lr: float = 1e-3, seed: int = 0) -> List[float]:
    """Train on one fixed batch; returns the loss per step. Loss should fall
    sharply if gradients flow correctly through the whole network."""
    _require_torch()
    torch.manual_seed(seed)
    model = TorchModel(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: List[float] = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(images), behavior_t, essence_t, cfg.essence_weight)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return history


def _demo() -> None:  # pragma: no cover - manual smoke
    from dataclasses import replace
    _require_torch()
    cfg = replace(ModelConfig(), conv_channels=(8, 16), latent_dim=32,
                  behavior_head_width=32, essence_head_width=16)
    torch.manual_seed(0)
    images = torch.rand(4, 3, 24, 24, 3)               # B=4, N=3 views
    behavior_t = torch.rand(4, cfg.behavior_dim)
    essence_t = torch.rand(4, cfg.essence_dim)
    hist = overfit_smoke(cfg, images, behavior_t, essence_t, steps=200)
    print(f"[torch smoke] loss {hist[0]:.4f} -> {hist[-1]:.5f} "
          f"(x{hist[0] / max(1e-9, hist[-1]):.1f} reduction)")


if __name__ == "__main__":
    _demo()
