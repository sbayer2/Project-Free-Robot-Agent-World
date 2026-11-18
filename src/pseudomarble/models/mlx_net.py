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

        def attach_pool(self) -> None:
            """Build the cross-view pooling layer (none for "mean"). Called by
            PseudoMarbleModel after its heads, so turning pooling on leaves the
            initial CNN and head weights of a given seed unchanged."""
            from pseudomarble.config import ATTN_POOLS, check_view_pool
            cfg = self.cfg
            check_view_pool(cfg)
            if cfg.view_pool in ("stats", "stats_ctrl"):
                self.pool = nn.Linear(2 * cfg.latent_dim, cfg.latent_dim)
            elif cfg.view_pool in ATTN_POOLS:
                self.mixer = ViewMixer(cfg)

        def views(self, images):
            """images (B, N, H, W, C) -> per-view latents v (B, N, latent_dim)."""
            B, N = images.shape[0], images.shape[1]
            x = images.reshape(B * N, *images.shape[2:])  # fold views into the batch
            for conv in self.convs:
                x = nn.relu(conv(x))
            x = x.mean(axis=(1, 2))                  # global average pool over H, W
            v = nn.relu(self.proj(x))                # per-view latent
            return v.reshape(B, N, -1)

        def pool_views(self, v):
            """Per-view latents (B, N, L) -> z (B, L), by cfg.view_pool. Mirrors
            numpy_net.view_pool_forward."""
            from pseudomarble.config import VIEW_STD_EPS
            kind = self.cfg.view_pool
            if kind == "mean":
                return v.mean(axis=1)                # pool over the N views
            if kind in ("stats", "stats_ctrl"):
                m = v.mean(axis=1)
                second = (mx.sqrt(mx.mean((v - m[:, None, :]) ** 2, axis=1) + VIEW_STD_EPS)
                          if kind == "stats" else m)
                return nn.relu(self.pool(mx.concatenate([m, second], axis=-1)))
            return self.mixer(v)

        def __call__(self, images):
            return self.pool_views(self.views(images))

    class ViewMixer(nn.Module):
        """view_pool "attn": one pre-LayerNorm transformer block over the N view
        tokens v_i + E d_i (d_i = camera i's unit direction), ReLU FFN, residual,
        then the mean over tokens. "attn_ctrl" adds a self-only attention mask:
        the same parameters and depth, with nothing passed between views."""

        def __init__(self, cfg: ModelConfig) -> None:
            super().__init__()
            from pseudomarble.config import VIEW_LN_EPS, VIEW_MASK_NEG
            L, F = cfg.latent_dim, cfg.view_attn_ffn
            self._heads = cfg.view_attn_heads
            self.dir_embed = nn.Linear(3, L)
            self.ln1 = nn.LayerNorm(L, eps=VIEW_LN_EPS)
            self.q_proj = nn.Linear(L, L)
            self.k_proj = nn.Linear(L, L)
            self.v_proj = nn.Linear(L, L)
            self.out_proj = nn.Linear(L, L)
            self.ln2 = nn.LayerNorm(L, eps=VIEW_LN_EPS)
            self.ff1 = nn.Linear(L, F)
            self.ff2 = nn.Linear(F, L)
            # Underscored arrays are not parameters: never trained, never saved.
            self._dirs = mx.array([list(d) for d in cfg.view_dirs], dtype=mx.float32)
            n = len(cfg.view_dirs)
            self._mask = (mx.where(mx.eye(n) > 0, 0.0, VIEW_MASK_NEG).astype(mx.float32)
                          if cfg.view_pool == "attn_ctrl" else None)

        def __call__(self, v):
            B, N, L = v.shape
            if N != self._dirs.shape[0]:
                raise ValueError(f"{N} views but {self._dirs.shape[0]} view_dirs")
            H = self._heads
            dh = L // H

            def split(t):  # (B, N, L) -> (B, H, N, dh)
                return t.reshape(B, N, H, dh).transpose(0, 2, 1, 3)

            x = v + self.dir_embed(self._dirs)[None]
            h = self.ln1(x)
            q, k, vv = split(self.q_proj(h)), split(self.k_proj(h)), split(self.v_proj(h))
            s = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / (dh ** 0.5))
            if self._mask is not None:
                s = s + self._mask
            o = (mx.softmax(s, axis=-1) @ vv).transpose(0, 2, 1, 3).reshape(B, N, L)
            x = x + self.out_proj(o)
            x = x + self.ff2(nn.relu(self.ff1(self.ln2(x))))
            return x.mean(axis=1)

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
            if cfg.trajectory_weight > 0:  # F31 head; gated to keep default identical
                from pseudomarble.config import trajectory_target_dim
                self.trajectory = MLP(cfg.latent_dim, cfg.trajectory_head_width,
                                      trajectory_target_dim(cfg))

            # Render decoder: z -> seed map -> (upsample + conv)*k -> RGB (NHWC).
            ch, s = cfg.render_channels, cfg.render_seed
            self._seed_ch, self._seed_s = ch, s
            self.seed = nn.Linear(cfg.latent_dim, ch * s * s)
            self.dec_convs = [nn.Conv2d(ch, ch, 3, padding=1)
                              for _ in range(num_upsample_steps(cfg))]
            self.dec_final = nn.Conv2d(ch, 3, 3, padding=1)
            self.encoder.attach_pool()  # last: see Encoder.attach_pool

        def encode_views(self, images):
            """Per-view latents v (B, N, latent_dim) before pooling (D1 diagnostic)."""
            return self.encoder.views(images)

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
            if self.cfg.trajectory_weight > 0:
                out["trajectory"] = self.trajectory(z)
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

        def trajectory_from_z(self, z):
            return self.trajectory(z)

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
    if cfg.trajectory_weight > 0:  # F31: predict the recorded probe paths
        t = mx.mean((out["trajectory"] - batch["trajectory"]) ** 2)
        total = total + cfg.trajectory_weight * t
    if cfg.coherence_weight > 0:  # F25 objective: train the heads to move together
        total = total + cfg.coherence_weight * coherence_term(model, out, cfg)
    return total


def coherence_term(model, out: Dict, cfg: ModelConfig):
    """1 - Pearson(|render response|, |behavior response|) under shared latent
    perturbations -- the differentiable analogue of models/coherence.py's metric
    (losses.coherence_alignment_loss is the framework-free reference).

    Perturbs each sample's z along cfg.coherence_dirs fresh random directions
    (step cfg.coherence_eps) and correlates the two heads' per-(sample, dir)
    response magnitudes over the B*D points. Adds no parameters; the extra cost
    is D batched decoder/head forwards per step.
    """
    z = out["z"]                                   # (B, L)
    B, L = z.shape
    D = cfg.coherence_dirs
    u = mx.random.normal((D, L))
    u = u / mx.maximum(mx.sqrt(mx.sum(u * u, axis=1, keepdims=True)), 1e-8)
    u = mx.stop_gradient(u)
    zp = (z[:, None, :] + cfg.coherence_eps * u[None, :, :]).reshape(B * D, L)

    d_render = model.decode(zp) - mx.repeat(out["render"], D, axis=0)
    d_behav = model.behavior_from_z(zp) - mx.repeat(out["behavior"], D, axis=0)
    r_mag = mx.mean(mx.abs(d_render.reshape(B * D, -1)), axis=1)   # (B*D,)
    b_mag = mx.mean(mx.abs(d_behav.reshape(B * D, -1)), axis=1)

    rc = r_mag - mx.mean(r_mag)
    bc = b_mag - mx.mean(b_mag)
    denom = mx.sqrt(mx.sum(rc * rc) * mx.sum(bc * bc))
    corr = mx.sum(rc * bc) / mx.maximum(denom, 1e-12)
    return 1.0 - corr
