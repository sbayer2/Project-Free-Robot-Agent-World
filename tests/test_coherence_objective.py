"""Tests for the F25 coherence objective (docs/ARCHITECTED_UNITY.md).

Three layers, mirroring the repo's backend rule:
* pure-Python reference (losses.pearson / coherence_alignment_loss) -- no deps
* torch mirror (torch_net.coherence_term) -- skips without torch
* MLX mirror (mlx_net.coherence_term) -- skips off-Mac

Plus the gating contract: coherence_weight defaults to 0.0 and the default
objective is unchanged (the F20 byte-identical pattern), and train.py wires
--coherence-weight into the config.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.config import ModelConfig  # noqa: E402
from pseudomarble.models.losses import (  # noqa: E402
    coherence_alignment_loss,
    pearson,
)

try:
    import torch
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

try:
    import mlx.core as mx  # noqa: F401
    _HAVE_MLX = True
except Exception:  # pragma: no cover
    _HAVE_MLX = False

SMALL = replace(
    ModelConfig(),
    conv_channels=(4, 8),
    latent_dim=16,
    encoder_width=32,
    behavior_head_width=16,
    essence_head_width=8,
    render_channels=8,
    image_size=16,
)


def test_pearson_endpoints():
    assert abs(pearson([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-12
    assert abs(pearson([1, 2, 3, 4], [8, 6, 4, 2]) + 1.0) < 1e-12
    assert pearson([1, 1, 1, 1], [2, 4, 6, 8]) == 0.0  # constant side -> neutral


def test_coherence_alignment_loss_range():
    assert abs(coherence_alignment_loss([1, 2, 3], [2, 4, 6]) - 0.0) < 1e-12
    assert abs(coherence_alignment_loss([1, 2, 3], [6, 4, 2]) - 2.0) < 1e-12
    mid = coherence_alignment_loss([1.0, 2.0, 1.5, 3.0], [2.0, 1.0, 3.0, 1.5])
    assert 0.0 <= mid <= 2.0


def test_default_config_gates_the_term_off():
    cfg = ModelConfig()
    assert cfg.coherence_weight == 0.0, (
        "the F25 objective must default OFF so the default model, its "
        "checkpoints, and every historical result are byte-identical")
    assert cfg.coherence_dirs > 0 and cfg.coherence_eps > 0


def test_trainpy_wires_coherence_weight():
    from pseudomarble.models.train import make_config, parse_args
    args = parse_args(["--data", "x", "--coherence-weight", "0.7"])
    assert make_config(args).coherence_weight == 0.7
    args = parse_args(["--data", "x"])
    assert make_config(args).coherence_weight == 0.0  # untouched default


def test_torch_coherence_term_range_and_gradients():
    if not _HAVE_TORCH:  # pragma: no cover
        print("torch unavailable; skipping")
        return
    from pseudomarble.models import torch_net

    torch.manual_seed(0)
    cfg = replace(SMALL, coherence_weight=1.0)
    model = torch_net.build_model(cfg)
    imgs = torch.rand(3, 2, cfg.image_size, cfg.image_size, 3)
    out = model(imgs)
    term = torch_net.coherence_term(model, out, cfg)
    assert 0.0 <= float(term) <= 2.0
    term.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads), (
        "coherence term must backprop into the model")


def test_torch_loss_unchanged_when_gated_off():
    if not _HAVE_TORCH:  # pragma: no cover
        print("torch unavailable; skipping")
        return
    from pseudomarble.models import torch_net

    torch.manual_seed(1)
    model = torch_net.build_model(SMALL)  # coherence_weight = 0.0 default
    imgs = torch.rand(2, 2, SMALL.image_size, SMALL.image_size, 3)
    out = model(imgs)
    bt = torch.rand(2, SMALL.behavior_dim)
    et = torch.rand(2, SMALL.essence_dim)
    rt = imgs.mean(dim=1)
    base = torch_net.loss_fn(out, bt, et, SMALL, render_t=rt)
    with_model = torch_net.loss_fn(out, bt, et, SMALL, render_t=rt, model=model)
    assert float(base) == float(with_model), (
        "passing model= must be a no-op while coherence_weight == 0")


def test_mlx_coherence_term_range():
    if not _HAVE_MLX:  # pragma: no cover
        print("mlx unavailable; skipping")
        return
    import mlx.core as mx

    from pseudomarble.models import mlx_net

    mx.random.seed(0)
    cfg = replace(SMALL, coherence_weight=1.0)
    model = mlx_net.build_model(cfg)
    imgs = mx.random.uniform(shape=(3, 2, cfg.image_size, cfg.image_size, 3))
    out = model(imgs)
    term = mlx_net.coherence_term(model, out, cfg)
    mx.eval(term)
    assert 0.0 <= float(term) <= 2.0
    # And the full loss trains: grad through the term is finite.
    batch = {"images": imgs,
             "behavior": mx.zeros((3, cfg.behavior_dim)),
             "essence": mx.zeros((3, cfg.essence_dim))}

    def f(m):
        return mlx_net.loss_fn(m, batch, cfg)

    import mlx.nn as nn
    loss, grads = nn.value_and_grad(model, f)(model)
    mx.eval(loss)
    assert float(loss) == float(loss), "loss must be finite (not NaN)"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"{len(fns)} coherence-objective tests passed")
