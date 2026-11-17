"""Cross-view pooling (ModelConfig.view_pool; docs/CROSSVIEW_POOLING.md §3).

The numpy forward (numpy_net.view_pool_forward) is the reference. The mlx and
torch poolers are checked against it with the same weights, for every arm.
The checks also cover: the default "mean" model is unchanged; the matched
controls pass nothing between views; the std has a finite gradient at zero
spread; camera directions come from the stored frames, never from a pose
function.
"""

import json
import math
import os
from dataclasses import replace

import pytest

from pseudomarble.config import VIEW_POOLS, ModelConfig, check_view_pool
from pseudomarble.data import samples
from pseudomarble.data.dataset import PseudoMarbleDataset
from pseudomarble.models import train

np = pytest.importorskip("numpy")
from pseudomarble.models import numpy_net  # noqa: E402

N_VIEWS = 6
SMALL = replace(ModelConfig(), conv_channels=(4, 8), latent_dim=16,
                behavior_head_width=16, essence_head_width=8, image_size=16,
                view_attn_heads=4, view_attn_ffn=24)


def _dirs(n=N_VIEWS):
    look = (0.0, 0.0, 0.15)
    out = []
    for az, el in samples.above_floor_camera_angles(n):
        p = samples.camera_world_position(look, az, el, 2.6)
        d = [a - b for a, b in zip(p, look, strict=True)]
        norm = math.sqrt(sum(x * x for x in d))
        out.append(tuple(x / norm for x in d))
    return tuple(out)


def _cfg(pool):
    return replace(SMALL, view_pool=pool, view_dirs=_dirs() if "attn" in pool else ())


def _v(seed=0, b=3):
    return np.random.default_rng(seed).normal(size=(b, N_VIEWS, SMALL.latent_dim)) \
        .astype("float32")


def _np_params(pool, seed=0):
    p = numpy_net._init_view_pool(_cfg(pool), np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 100)   # non-trivial LayerNorm affine, too
    return {k: (w + 0.1 * rng.normal(size=w.shape).astype("float32"),
                b + 0.1 * rng.normal(size=b.shape).astype("float32"))
            for k, (w, b) in p.items()}


# -- config ------------------------------------------------------------------ #
def test_check_view_pool_rejects_what_cannot_build():
    check_view_pool(ModelConfig())
    with pytest.raises(ValueError, match="unknown"):
        check_view_pool(replace(SMALL, view_pool="fnet"))
    with pytest.raises(ValueError, match="view_dirs"):
        check_view_pool(replace(SMALL, view_pool="attn"))
    with pytest.raises(ValueError, match="divisible"):
        check_view_pool(replace(_cfg("attn"), view_attn_heads=3))
    with pytest.raises(ValueError, match="6 directions for 5 views"):
        check_view_pool(_cfg("attn_ctrl"), num_views=5)


def test_parameter_counts_match_the_prereg():
    """§3: P1/P1' add 131 k parameters, P2/P2' about 0.53 M, P0 none."""
    from pseudomarble.config import ATTN_POOLS
    cfg = ModelConfig()
    L, F = cfg.latent_dim, cfg.view_attn_ffn
    want = {"mean": 0, "stats": 2 * L * L + L, "stats_ctrl": 2 * L * L + L}
    attn = (3 * L + L) + 4 * (L * L + L) + (L * F + F) + (F * L + L) + 2 * (2 * L)
    for pool in VIEW_POOLS:
        c = replace(cfg, view_pool=pool, view_dirs=_dirs(16) if pool in ATTN_POOLS else ())
        p = numpy_net._init_view_pool(c, np.random.default_rng(0))
        n = sum(w.size + b.size for w, b in p.values())
        assert n == want.get(pool, attn), pool
    assert want["stats"] == 131_328 and attn == 528_128


# -- the default model is unchanged ----------------------------------------- #
def test_mean_pool_is_the_plain_view_average_and_draws_nothing():
    x = np.random.default_rng(1).random((2, N_VIEWS, 16, 16, 3))
    m = numpy_net.NumpyModel(SMALL, seed=0)
    assert m.pool == {}
    assert np.allclose(m.encode(x), m.encode_views(x).mean(axis=1), atol=0)


def test_turning_pooling_on_leaves_the_other_initial_weights_unchanged():
    base = numpy_net.NumpyModel(SMALL, seed=3)
    for pool in VIEW_POOLS[1:]:
        m = numpy_net.NumpyModel(_cfg(pool), seed=3)
        for name in ("Wp", "Wb1", "We2", "Ws", "Wf"):
            assert np.array_equal(getattr(m, name), getattr(base, name)), (pool, name)


# -- what the matched controls can and cannot see --------------------------- #
@pytest.mark.parametrize("pool,separable", [("mean", True), ("stats_ctrl", True),
                                            ("attn_ctrl", True), ("stats", False),
                                            ("attn", False)])
def test_controls_pass_nothing_between_views(pool, separable):
    """Without mixing, z is a mean of per-view terms, so changing view 0 moves z
    by an amount that does not depend on the other views. The stats arms end in
    a ReLU; a large bias keeps it in its linear range, so the test reads the
    pooled features themselves (stats_ctrl: linear in the mean; stats: the std
    couples the views)."""
    cfg, p = _cfg(pool), _np_params(pool)
    if pool in ("stats", "stats_ctrl"):
        W, b0 = p["pool"]
        p = {"pool": (W, b0 + 100.0)}

    def inner(v):
        return numpy_net.view_pool_forward(v, p, cfg)
    a, b = _v(1, b=1), _v(2, b=1)
    b[:, 0] = a[:, 0]                            # same view 0, different others
    a2, b2 = a.copy(), b.copy()
    bump = np.random.default_rng(9).normal(size=SMALL.latent_dim).astype("float32")
    a2[:, 0] += bump
    b2[:, 0] += bump
    same = np.allclose(inner(a2) - inner(a), inner(b2) - inner(b), atol=1e-4)
    assert same == separable


@pytest.mark.parametrize("pool", ["attn", "attn_ctrl"])
def test_attn_arms_pair_each_view_with_its_own_direction(pool):
    """Reordering views together with their directions leaves z unchanged (the
    block is permutation-equivariant and z is the token mean); reordering the
    views alone does not, so each token carries its own camera's direction."""
    cfg, p = _cfg(pool), _np_params(pool)
    v = _v(4, b=2)
    order = [0, 5, 3, 1, 4, 2]
    z = numpy_net.view_pool_forward(v, p, cfg)
    moved = replace(cfg, view_dirs=tuple(cfg.view_dirs[i] for i in order))
    assert np.allclose(numpy_net.view_pool_forward(v[:, order], p, moved), z, atol=1e-5)
    assert not np.allclose(numpy_net.view_pool_forward(v[:, order], p, cfg), z, atol=1e-3)


# -- cross-backend agreement ------------------------------------------------ #
def _to_mlx(model, pool, p):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    enc = model.encoder
    if pool in ("stats", "stats_ctrl"):
        w, b = p["pool"]
        enc.pool.update({"weight": mx.array(w.T), "bias": mx.array(b)})
        return
    flat = []
    for k, (w, b) in p.items():
        if k.startswith("ln"):
            flat += [(f"{k}.weight", mx.array(w)), (f"{k}.bias", mx.array(b))]
        else:
            flat += [(f"{k}.weight", mx.array(w.T)), (f"{k}.bias", mx.array(b))]
    enc.mixer.update(tree_unflatten(flat))


@pytest.mark.parametrize("pool", VIEW_POOLS)
def test_mlx_pooler_matches_the_numpy_reference(pool):
    """Exact on MLX's CPU stream. On the Metal GPU a float32 matmul carries about
    1e-3 relative error on this Mac (a plain a @ b shows it too), so the GPU
    check is loose: it guards the wiring, the CPU check the arithmetic."""
    from pseudomarble.models import mlx_net
    if not mlx_net._HAVE_MLX:
        pytest.skip("MLX runs on the Mac only")
    import mlx.core as mx
    cfg, p = _cfg(pool), _np_params(pool)
    model = mlx_net.build_model(cfg)
    if p:
        _to_mlx(model, pool, p)
    v = _v(5)
    ref = numpy_net.view_pool_forward(v, p, cfg)
    assert ref.dtype == np.float32
    with mx.stream(mx.cpu):
        got = np.array(model.encoder.pool_views(mx.array(v)))
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-5)
    gpu = np.array(model.encoder.pool_views(mx.array(v)))
    assert np.allclose(gpu, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("pool", VIEW_POOLS)
def test_torch_pooler_matches_the_numpy_reference(pool):
    torch = pytest.importorskip("torch")
    from pseudomarble.models.torch_net import TorchModel
    cfg, p = _cfg(pool), _np_params(pool)
    model = TorchModel(cfg)
    with torch.no_grad():
        if pool in ("stats", "stats_ctrl"):
            model.pool.weight.copy_(torch.from_numpy(p["pool"][0].T.copy()))
            model.pool.bias.copy_(torch.from_numpy(p["pool"][1]))
        elif p:
            for k, (w, b) in p.items():
                mod = getattr(model.mixer, k)
                mod.weight.copy_(torch.from_numpy(w if k.startswith("ln") else w.T.copy()))
                mod.bias.copy_(torch.from_numpy(b))
        v = _v(5)
        got = model.pool_views(torch.from_numpy(v)).numpy()
    assert np.allclose(got, numpy_net.view_pool_forward(v, p, cfg), atol=1e-4)


@pytest.mark.parametrize("pool", VIEW_POOLS)
def test_mlx_model_runs_saves_and_reloads_every_arm(pool, tmp_path):
    from pseudomarble.models import mlx_net
    if not mlx_net._HAVE_MLX:
        pytest.skip("MLX runs on the Mac only")
    import mlx.core as mx
    from mlx.utils import tree_flatten
    cfg = _cfg(pool)
    model = mlx_net.build_model(cfg)
    x = mx.array(np.random.default_rng(6).random((2, N_VIEWS, 16, 16, 3)).astype("float32"))
    out = model(x)
    assert out["z"].shape == (2, SMALL.latent_dim)
    assert model.encode_views(x).shape == (2, N_VIEWS, SMALL.latent_dim)
    keys = {k for k, _ in tree_flatten(model.parameters())}
    assert not any("_dirs" in k or "_mask" in k for k in keys)   # not trained, not saved
    assert any(k.startswith("encoder.pool.") or k.startswith("encoder.mixer.")
               for k in keys) == (pool != "mean")
    path = str(tmp_path / "m.safetensors")
    model.save_weights(path)
    again = mlx_net.build_model(cfg)
    again.load_weights(path)
    assert np.allclose(np.array(again(x)["z"]), np.array(out["z"]), atol=1e-6)


def test_mlx_mean_model_parameter_tree_is_unchanged():
    """No new keys for the default: old checkpoints still load strictly."""
    from pseudomarble.models import mlx_net
    if not mlx_net._HAVE_MLX:
        pytest.skip("MLX runs on the Mac only")
    from mlx.utils import tree_flatten
    keys = [k for k, _ in tree_flatten(mlx_net.build_model(SMALL).parameters())]
    assert [k for k in keys if k.startswith("encoder.")] == [
        "encoder.convs.0.weight", "encoder.convs.0.bias",
        "encoder.convs.1.weight", "encoder.convs.1.bias",
        "encoder.proj.weight", "encoder.proj.bias"]


def test_std_has_a_finite_gradient_when_views_agree():
    """Dead ReLU units give v_i identical across views; sqrt(var + eps) keeps
    the gradient finite there (plain sqrt(var) would be infinite)."""
    from pseudomarble.models import mlx_net
    if not mlx_net._HAVE_MLX:
        pytest.skip("MLX runs on the Mac only")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    model = mlx_net.build_model(_cfg("stats"))
    v = _v(7)
    v[:, :, 3] = 0.0                                  # a dead unit in every view
    v[:, :, 5] = 1.25                                 # a constant one
    v = mx.array(v)

    def f(m, v):
        return mx.sum(m.encoder.pool_views(v))
    _, grads = nn.value_and_grad(model, f)(model, v)
    gv = mx.grad(lambda v: f(model, v))(v)
    assert all(np.isfinite(np.array(g)).all() for _, g in tree_flatten(grads))
    assert np.isfinite(np.array(gv)).all()


# -- camera directions from the stored frames -------------------------------- #
def _world(root, n_scenes=3, layout="above", perturb=None):
    look = [0.0, 0.0, 0.15]
    if layout == "above":
        pos = [samples.camera_world_position(look, az, el, 2.6)
               for az, el in samples.above_floor_camera_angles(N_VIEWS)]
    else:
        pos = samples.fibonacci_sphere_poses(N_VIEWS, 2.6)
    scenes = []
    for s in range(n_scenes):
        sid = f"s{s}"
        frames = [samples.build_frame(i, f"view_{i:03d}.png", p, look)
                  for i, p in enumerate(pos)]
        if perturb is not None and s == perturb:
            frames[2]["position"][0] += 0.05
        rec = {"split": "train" if s else "test", "input": {"shape": "box"},
               "appearance": {"resolution": 16, "frames": frames},
               "behavior": {"probes": []}, "physics": {"normalized": {}},
               "material_truth": {}}
        os.makedirs(os.path.join(root, sid))
        with open(os.path.join(root, sid, "sample.json"), "w") as fh:
            json.dump(rec, fh)
        scenes.append({"scene_id": sid, "split": rec["split"]})
    with open(os.path.join(root, "manifest.json"), "w") as fh:
        json.dump({"scenes": scenes}, fh)
    return pos, look


def test_camera_directions_are_read_from_the_stored_frames(tmp_path):
    pos, look = _world(str(tmp_path))
    d = PseudoMarbleDataset(str(tmp_path)).camera_directions()
    assert len(d) == N_VIEWS
    for u, p in zip(d, pos, strict=True):
        w = [a - b for a, b in zip(p, look, strict=True)]
        n = math.sqrt(sum(x * x for x in w))
        assert u == pytest.approx([x / n for x in w], abs=1e-12)
    el = [math.degrees(math.asin(u[2])) for u in d]
    assert el[0] == pytest.approx(15.0) and el[-1] == pytest.approx(65.0)
    assert el == sorted(el)                          # 15 -> 65 in view order
    # Not the legacy pose function's directions, which reach below the floor.
    legacy = [[x / 2.6 for x in p] for p in samples.fibonacci_sphere_poses(N_VIEWS, 2.6)]
    assert min(u[2] for u in legacy) < 0 < min(u[2] for u in d)


def test_with_view_dirs_sets_attn_arms_only_and_caps_views(tmp_path):
    _world(str(tmp_path))
    full = PseudoMarbleDataset(str(tmp_path)).camera_directions()
    cfg = train.with_view_dirs(replace(SMALL, view_pool="attn"), str(tmp_path))
    assert [list(u) for u in cfg.view_dirs] == full
    capped = train.with_view_dirs(replace(SMALL, view_pool="attn_ctrl"), str(tmp_path), 4)
    assert [list(u) for u in capped.view_dirs] == full[:4]
    for pool in ("mean", "stats", "stats_ctrl"):
        assert train.with_view_dirs(replace(SMALL, view_pool=pool), str(tmp_path)).view_dirs == ()


def test_scenes_with_different_cameras_are_refused(tmp_path):
    _world(str(tmp_path), perturb=1)
    with pytest.raises(ValueError, match="different cameras"):
        PseudoMarbleDataset(str(tmp_path)).camera_directions()


def test_cli_view_pool_reaches_the_config():
    assert train.make_config(train.parse_args([])).view_pool == "mean"
    cfg = train.make_config(train.parse_args(["--view-pool", "attn_ctrl"]))
    assert cfg.view_pool == "attn_ctrl" and cfg.view_dirs == ()
    with pytest.raises(SystemExit):
        train.parse_args(["--view-pool", "fnet"])


def test_a_real_above_floor_world_if_present():
    """On the Mac's generated world: every scene agrees, the directions equal
    the stored positions, and every camera sits 15-65 degrees above the
    horizon. Set PM_CAM_WORLD to point at it; skipped where it is absent."""
    root = os.environ.get("PM_CAM_WORLD", "data/pm_f31fix_loud_cam")
    if not os.path.exists(os.path.join(root, "manifest.json")):
        pytest.skip(f"no world at {root}")
    ds = PseudoMarbleDataset(root)
    d = ds.camera_directions()
    assert len(d) == 16
    for f, u in zip(ds[0].record["appearance"]["frames"], d, strict=True):
        w = [a - b for a, b in zip(f["position"], f["look_at"], strict=True)]
        n = math.sqrt(sum(x * x for x in w))
        assert u == pytest.approx([x / n for x in w], abs=1e-12)
    el = [math.degrees(math.asin(u[2])) for u in d]
    assert el[0] == pytest.approx(15.0, abs=1e-6) and el[-1] == pytest.approx(65.0, abs=1e-6)


@pytest.mark.parametrize("pool", VIEW_POOLS)
def test_every_arm_trains_on_a_fixed_batch_in_mlx(pool):
    """Gradients reach the pooling layer and the loss falls, on random images
    (no world is touched: the prereg's only pre-freeze training is the ctrl
    stability check)."""
    from pseudomarble.models import mlx_net
    if not mlx_net._HAVE_MLX:
        pytest.skip("MLX runs on the Mac only")
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    mx.random.seed(0)
    cfg = _cfg(pool)
    model = mlx_net.build_model(cfg)
    rng = np.random.default_rng(8)
    batch = {"images": mx.array(rng.random((4, N_VIEWS, 16, 16, 3)).astype("float32")),
             "behavior": mx.array(rng.random((4, cfg.behavior_dim)).astype("float32")),
             "essence": mx.array(rng.random((4, cfg.essence_dim)).astype("float32"))}
    opt = optim.AdamW(learning_rate=1e-3)
    step = nn.value_and_grad(model, lambda m: mlx_net.loss_fn(m, batch, cfg))
    losses = []
    for i in range(40):
        loss, grads = step(model)
        if i == 0 and pool != "mean":
            g = dict(tree_flatten(grads))
            new = [k for k in g if ".pool." in k or ".mixer." in k]
            assert new and any(float(mx.abs(g[k]).max()) > 0 for k in new)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        losses.append(float(loss))
    assert all(math.isfinite(x) for x in losses) and losses[-1] < 0.5 * losses[0]
