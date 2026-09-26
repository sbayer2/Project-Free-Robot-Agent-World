"""F31 trajectory target + head: pure-Python target, dataset hook, gated heads.

The head mirrors the F20 appearance head's contract: off by default (no
parameters, no output key, default model byte-identical), on when
trajectory_weight > 0, same output shape in every backend.
"""

import json
from dataclasses import replace

import pytest

from pseudomarble import probes as P
from pseudomarble.config import ModelConfig
from pseudomarble.models import train

SMALL = replace(ModelConfig(), conv_channels=(4, 8), latent_dim=16,
                behavior_head_width=16, essence_head_width=8, image_size=16,
                trajectory_frames=4, trajectory_head_width=8)


def _traj(n=10, dx=0.0, up=(0.0, 0.0, 1.0)):
    return [{"t": i * 0.02, "pos": [i * dx, 0.0, 0.15], "up": list(up)}
            for i in range(n)]


def _records(dx=0.1):
    return [{"probe": k, "outcome": {}, "trajectory": _traj(dx=dx)}
            for k in ("push", "drop", "tilt")]            # on-disk order shuffled


def test_trajectory_vector_layout_scale_and_order():
    v = P.trajectory_vector(_records(dx=0.1), frames=4)
    assert len(v) == P.trajectory_dim(4) == 3 * 4 * 6
    first = v[:6]                                          # drop, frame 0
    assert first == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    last_drop = v[3 * 6:4 * 6]                             # drop, final frame (index 9)
    assert last_drop[0] == pytest.approx(0.9 / P.TRAJECTORY_POS_SCALE)


def test_trajectory_vector_refuses_missing_paths():
    recs = [{"probe": k, "outcome": {}} for k in P.PROBE_ORDER]
    with pytest.raises(ValueError, match="keep-trajectory"):
        P.trajectory_vector(recs, frames=4)
    with pytest.raises(ValueError):
        P.trajectory_vector(_records(), frames=1)


def test_dataset_adds_trajectory_only_on_request(tmp_path):
    from pseudomarble.data.dataset import PseudoMarbleDataset

    scene = {"split": "train", "input": {"shape": "box"},
             "behavior": {"probes": _records()},
             "physics": {"normalized": {}}, "material_truth": {}}
    (tmp_path / "s0").mkdir()
    (tmp_path / "s0" / "sample.json").write_text(json.dumps(scene))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"scenes": [{"scene_id": "s0", "split": "train"}]}))
    ds = PseudoMarbleDataset(str(tmp_path))
    assert "trajectory" not in next(ds.iter_batches(1))
    b = next(ds.iter_batches(1, trajectory_frames=4))
    assert len(b["trajectory"][0]) == P.trajectory_dim(4)


def test_cli_flags_reach_the_config_and_gate_loading():
    cfg = train.make_config(train.parse_args(["--trajectory-weight", "0.5",
                                              "--trajectory-frames", "8"]))
    assert (cfg.trajectory_weight, cfg.trajectory_frames) == (0.5, 8)
    assert train.trajectory_frames_for(cfg) == 8
    assert train.trajectory_frames_for(ModelConfig()) == 0   # off by default


def test_numpy_head_gated_and_shaped():
    np = pytest.importorskip("numpy")
    from pseudomarble.models.numpy_net import NumpyModel

    x = np.random.default_rng(1).random((2, 3, 16, 16, 3))
    off = NumpyModel(SMALL, seed=0)
    assert "trajectory" not in off(x) and not hasattr(off, "Wt1")
    on = NumpyModel(replace(SMALL, trajectory_weight=1.0), seed=0)
    assert on(x)["trajectory"].shape == (2, P.trajectory_dim(4))


def test_torch_head_gated_shaped_and_in_the_loss():
    torch = pytest.importorskip("torch")
    from pseudomarble.models.torch_net import TorchModel, loss_fn

    x = torch.rand(2, 3, 16, 16, 3)
    off = TorchModel(SMALL)
    assert "trajectory" not in off(x) and not hasattr(off, "trajectory")
    cfg = replace(SMALL, trajectory_weight=2.0)
    m = TorchModel(cfg)
    out = m(x)
    assert out["trajectory"].shape == (2, P.trajectory_dim(4))
    beh, ess = torch.zeros(2, cfg.behavior_dim), torch.zeros(2, cfg.essence_dim)
    tgt = out["trajectory"].detach() + 1.0                 # MSE exactly 1
    base = loss_fn(out, beh, ess, cfg)
    with_t = loss_fn(out, beh, ess, cfg, trajectory_t=tgt)
    assert float(with_t - base) == pytest.approx(2.0, rel=1e-5)
