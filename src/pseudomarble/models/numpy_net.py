"""NumPy forward-reference of the encoder + behavior/essence heads.

MLX has no working runtime on a plain-Linux session (the pip wheel ships without
``libmlx.so``), so this NumPy implementation is the in-sandbox stand-in: it builds
the *same* architecture as ``models/mlx_net.py`` from the same ``ModelConfig`` and
runs a real forward pass, so the data->encode->heads->loss pipeline and all the
shape arithmetic are exercised and tested in any session. It is **forward only**
(no autodiff); training happens with MLX on the Mac.

Architecture (shared with mlx_net):
    images (B, N, H, W, C)
      -> per-view conv stack (Conv2d + ReLU, strided)   # N views folded into batch
      -> global average pool over H, W
      -> Linear -> ReLU                                  # per-view latent
      -> mean over the N views                           # the shared latent z (B, D)
      -> behavior head: Linear -> ReLU -> Linear  (B, behavior_dim)
      -> essence  head: Linear -> ReLU -> Linear  (B, essence_dim)
"""

from __future__ import annotations

from typing import Dict

from pseudomarble.config import ModelConfig


def _np():
    import numpy as np  # imported lazily so the package stays dep-light
    return np


def _he(shape, rng):
    np = _np()
    fan_in = 1
    for s in shape[:-1]:
        fan_in *= s
    return (rng.standard_normal(shape) * np.sqrt(2.0 / max(1, fan_in))).astype("float32")


def _conv2d(x, w, b, stride, pad=0):
    """NHWC convolution. x:(N,H,W,Cin) w:(KH,KW,Cin,Cout) -> (N,OH,OW,Cout)."""
    np = _np()
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    N, H, W, _ = x.shape
    KH, KW, _, Cout = w.shape
    OH = (H - KH) // stride + 1
    OW = (W - KW) // stride + 1
    if OH <= 0 or OW <= 0:
        raise ValueError(
            f"input {H}x{W} too small for kernel {KH}x{KW} stride {stride}; "
            "use a larger image_size or fewer/larger-stride conv layers"
        )
    wflat = w.reshape(-1, Cout)
    out = np.empty((N, OH, OW, Cout), dtype="float32")
    for i in range(OH):
        hs = i * stride
        for j in range(OW):
            ws = j * stride
            patch = x[:, hs:hs + KH, ws:ws + KW, :].reshape(N, -1)
            out[:, i, j, :] = patch @ wflat + b
    return out


def _upsample2(x):
    """Nearest-neighbour 2x upsample of an NHWC tensor."""
    np = _np()
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def _sigmoid(x):
    np = _np()
    return 1.0 / (1.0 + np.exp(-x))


class NumpyModel:
    """Forward-only NumPy model mirroring the MLX network's architecture."""

    def __init__(self, cfg: ModelConfig = ModelConfig(), seed: int = 0) -> None:
        np = _np()
        from pseudomarble.config import num_upsample_steps
        self.cfg = cfg
        rng = np.random.default_rng(seed)

        self.convs = []
        cin = cfg.in_channels
        for cout in cfg.conv_channels:
            w = _he((cfg.conv_kernel, cfg.conv_kernel, cin, cout), rng)
            self.convs.append((w, np.zeros((cout,), "float32")))
            cin = cout

        self.Wp, self.bp = _he((cin, cfg.latent_dim), rng), np.zeros((cfg.latent_dim,), "float32")
        if cfg.latent_trits > 0:  # FSQ bottleneck (mirrors mlx_net)
            self.Wd, self.bd = _he((cfg.latent_dim, cfg.latent_trits), rng), np.zeros((cfg.latent_trits,), "float32")
            self.Wu, self.bu = _he((cfg.latent_trits, cfg.latent_dim), rng), np.zeros((cfg.latent_dim,), "float32")
        self.Wb1, self.bb1 = _he((cfg.latent_dim, cfg.behavior_head_width), rng), np.zeros((cfg.behavior_head_width,), "float32")
        self.Wb2, self.bb2 = _he((cfg.behavior_head_width, cfg.behavior_dim), rng), np.zeros((cfg.behavior_dim,), "float32")
        self.We1, self.be1 = _he((cfg.latent_dim, cfg.essence_head_width), rng), np.zeros((cfg.essence_head_width,), "float32")
        self.We2, self.be2 = _he((cfg.essence_head_width, cfg.essence_dim), rng), np.zeros((cfg.essence_dim,), "float32")
        if cfg.appearance_weight > 0:  # F20 aux head (mirrors mlx_net; gated)
            self.Wa1, self.ba1 = _he((cfg.latent_dim, cfg.appearance_head_width), rng), np.zeros((cfg.appearance_head_width,), "float32")
            self.Wa2, self.ba2 = _he((cfg.appearance_head_width, cfg.appearance_dim), rng), np.zeros((cfg.appearance_dim,), "float32")

        # Render decoder: z -> seed feature map -> (upsample + conv)*k -> RGB.
        ch, s = cfg.render_channels, cfg.render_seed
        self.n_up = num_upsample_steps(cfg)
        self.Ws, self.bs = _he((cfg.latent_dim, ch * s * s), rng), np.zeros((ch * s * s,), "float32")
        self.dec = [(_he((3, 3, ch, ch), rng), np.zeros((ch,), "float32")) for _ in range(self.n_up)]
        self.Wf, self.bf = _he((3, 3, ch, 3), rng), np.zeros((3,), "float32")

    def encode(self, images):
        """images: (B, N, H, W, C) -> z: (B, latent_dim)."""
        np = _np()
        x = np.asarray(images, dtype="float32")
        B, N = x.shape[0], x.shape[1]
        x = x.reshape(B * N, *x.shape[2:])
        for w, b in self.convs:
            x = np.maximum(_conv2d(x, w, b, self.cfg.conv_stride), 0.0)
        x = x.mean(axis=(1, 2))                       # global average pool
        z = np.maximum(x @ self.Wp + self.bp, 0.0)    # per-view latent
        return z.reshape(B, N, -1).mean(axis=1)       # pool over views

    def decode(self, z):
        """z: (B, latent_dim) -> reconstructed canonical image (B, S, S, 3)."""
        np = _np()
        ch, s = self.cfg.render_channels, self.cfg.render_seed
        x = np.maximum(z @ self.Ws + self.bs, 0.0).reshape(z.shape[0], s, s, ch)
        for w, b in self.dec:
            x = np.maximum(_conv2d(_upsample2(x), w, b, stride=1, pad=1), 0.0)
        return _sigmoid(_conv2d(x, self.Wf, self.bf, stride=1, pad=1))

    def behavior_from_z(self, z):
        np = _np()
        return np.maximum(z @ self.Wb1 + self.bb1, 0.0) @ self.Wb2 + self.bb2

    def essence_from_z(self, z):
        np = _np()
        return np.maximum(z @ self.We1 + self.be1, 0.0) @ self.We2 + self.be2

    def appearance_from_z(self, z):
        np = _np()
        return np.maximum(z @ self.Wa1 + self.ba1, 0.0) @ self.Wa2 + self.ba2

    def bottleneck(self, z):
        """(code, expanded z); identity when off. Forward-only: no gradient
        trick needed — code = round(tanh(.)) in {-1,0,1} (mirrors mlx_net)."""
        np = _np()
        if self.cfg.latent_trits <= 0:
            return None, z
        code = np.round(np.tanh(z @ self.Wd + self.bd))
        return code, np.maximum(code @ self.Wu + self.bu, 0.0)

    def __call__(self, images) -> Dict:
        code, z = self.bottleneck(self.encode(images))
        out = {"z": z, "behavior": self.behavior_from_z(z),
               "essence": self.essence_from_z(z), "render": self.decode(z)}
        if self.cfg.appearance_weight > 0:
            out["appearance"] = self.appearance_from_z(z)
        if code is not None:
            out["code"] = code
        return out
