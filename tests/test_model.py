"""Tests for model config, the MLX import guard, and the train CLI (pure-Python).

These run in any session: they check the config matches the behavior target, that
the MLX path either builds (on Apple silicon) or fails with a clear message
(elsewhere), and that the training arg parser behaves.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataclasses import replace  # noqa: E402

from pseudomarble import probes  # noqa: E402
from pseudomarble.config import ModelConfig, num_upsample_steps  # noqa: E402
from pseudomarble.models import mlx_net, train  # noqa: E402


def test_behavior_dim_matches_probes():
    # The model's behavior head must match the flattened probe target exactly.
    assert ModelConfig().behavior_dim == probes.BEHAVIOR_DIM


def test_num_upsample_steps_and_validation():
    assert num_upsample_steps(replace(ModelConfig(), image_size=128, render_seed=4)) == 5
    assert num_upsample_steps(replace(ModelConfig(), image_size=16, render_seed=4)) == 2
    try:
        num_upsample_steps(replace(ModelConfig(), image_size=96, render_seed=4))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-power-of-two image_size")


def test_mlx_builds_or_guards_clearly():
    if mlx_net._HAVE_MLX:  # on the Mac: it should actually build
        model = mlx_net.build_model(ModelConfig())
        assert model is not None
    else:  # elsewhere: a clear, actionable error
        try:
            mlx_net.build_model(ModelConfig())
        except RuntimeError as exc:
            assert "MLX" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected RuntimeError without MLX")


def test_train_args_defaults_and_overrides():
    ns = train.parse_args([])
    assert ns.data == "data/pseudo_marble"
    assert ns.epochs == 30
    assert ns.batch_size == 16

    ns2 = train.parse_args(["--epochs", "5", "--latent-dim", "64", "--out", "runs/x"])
    assert ns2.epochs == 5
    cfg = train.make_config(ns2)
    assert cfg.latent_dim == 64  # override applied


def test_make_config_default_keeps_model_config():
    ns = train.parse_args([])
    assert train.make_config(ns).latent_dim == ModelConfig().latent_dim
    # default per-head weights match the full shared objective
    assert train.make_config(ns).behavior_weight == 1.0
    assert train.make_config(ns).render_weight == ModelConfig().render_weight


def test_make_config_per_head_weight_overrides():
    # render-only: behavior + essence heads off
    ro = train.make_config(train.parse_args(
        ["--behavior-weight", "0", "--essence-weight", "0", "--render-weight", "1"]))
    assert (ro.behavior_weight, ro.essence_weight, ro.render_weight) == (0.0, 0.0, 1.0)
    # behavior-only: render head off
    bo = train.make_config(train.parse_args(
        ["--render-weight", "0", "--essence-weight", "0"]))
    assert (bo.render_weight, bo.essence_weight, bo.behavior_weight) == (0.0, 0.0, 1.0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
