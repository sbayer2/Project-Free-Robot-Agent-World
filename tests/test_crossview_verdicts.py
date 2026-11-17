"""The pooling verdicts (scripts/crossview_verdicts.py) and readouts
(scripts/crossview_measure.py): each frozen rule of docs/CROSSVIEW_POOLING.md
§4-§5 on synthetic reports, and the per-view constructions D1 reads."""

import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
V = pytest.importorskip("crossview_verdicts")
M = pytest.importorskip("crossview_measure")

N_LOUD, N_CTRL = 20, 3


def _rep(arm, nested, fixed=None, heldout=None, ctrl=False, pr=40.0, rank=100, d1=None):
    rows = {}
    for i, x in enumerate(nested):
        rows[f"runs/{arm}_s{i}"] = {
            "view_pool": V.VIEW_POOL[arm], "friction_nested": x,
            "z_r2_fixed": {"density": 0.0, "friction": (fixed or nested)[i],
                           "restitution": 0.98},
            "z_r2_heldout": {"friction": (heldout or [0.5] * len(nested))[i]},
            "std_rank": rank, "latent_pr": pr, "heldout_gain": 1.0,
            **({"d1": d1[i]} if d1 else {})}
    return {"runs": rows}


def _spread(mean, sd=0.05, n=N_LOUD, seed=0):
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    return list(mean + sd * z)


def _reps(**over):
    """Every arm at the P0 baseline, no effect anywhere, unless overridden."""
    d1 = [{"concat": 0.45, "C mean + std": 0.40, "D |DFT over view index|": 0.42,
           "G sorted across views": 0.44, "H mean + |dev| quantiles": 0.41}] * N_LOUD
    reps = {}
    for k, a in enumerate(V.ARMS):
        loud = over.get(a, {}).get("loud", {"nested": _spread(0.40, seed=k)})
        ctrl = over.get(a, {}).get("ctrl", {"nested": [-0.01] * N_CTRL})
        reps[a] = {"loud": _rep(a, **loud, d1=d1 if a == "P0" else None),
                   "ctrl": _rep(a, **ctrl, ctrl=True)}
    return reps


def test_welch_matches_scipy():
    from scipy import stats
    a, b = _spread(0.5, 0.06, seed=1), _spread(0.4, 0.09, seed=2)
    w = V.welch(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert w["t"] == pytest.approx(ref.statistic) and w["delta"] == pytest.approx(0.1)


def test_no_effect_reads_null_with_a_bound_below_the_bar():
    v = V.verdicts(_reps())
    h2 = v["primary"]["H2 attention mixing"]
    assert h2["verdict"] == "NULL" and h2["null_bound"] < 0.10
    assert all(v["gates"]["G1_health"].values()) and all(v["gates"]["G2_control"].values())
    assert v["gates"]["G3_apparatus"]["pass"]


def test_a_wide_spread_null_is_labelled_bound_above_bar():
    reps = _reps(P2={"loud": {"nested": _spread(0.43, sd=0.15, seed=3)}},
                 P2c={"loud": {"nested": _spread(0.40, sd=0.15, seed=4)}})
    h2 = V.verdicts(reps)["primary"]["H2 attention mixing"]
    assert h2["verdict"] == "NULL, bound above bar" and h2["null_bound"] >= 0.10


def test_rises_needs_the_held_out_difference_too():
    up = {"loud": {"nested": _spread(0.55, seed=5), "heldout": [0.6] * N_LOUD}}
    assert V.verdicts(_reps(P2=up))["primary"]["H2 attention mixing"]["verdict"] == "RISES"
    up_no_h = {"loud": {"nested": _spread(0.55, seed=5), "heldout": [0.4] * N_LOUD}}
    h2 = V.verdicts(_reps(P2=up_no_h))["primary"]["H2 attention mixing"]
    assert h2["nested"]["grade"] == "INCONCLUSIVE" and h2["verdict"] == "INCONCLUSIVE"


def test_rises_on_one_readout_only_is_readout_dependent():
    """§5's example: nested INCONCLUSIVE with fixed-alpha RISES."""
    p2 = {"loud": {"nested": _spread(0.47, seed=6), "fixed": _spread(0.55, seed=6),
                   "heldout": [0.6] * N_LOUD}}
    h2 = V.verdicts(_reps(P2=p2))["primary"]["H2 attention mixing"]
    assert (h2["nested"]["grade"], h2["fixed"]["grade"]) == ("INCONCLUSIVE", "RISES")
    assert h2["verdict"] == "READOUT-DEPENDENT"


def test_null_against_inconclusive_keeps_the_primary_grade():
    p2 = {"loud": {"nested": _spread(0.41, seed=7), "fixed": _spread(0.47, seed=7)}}
    h2 = V.verdicts(_reps(P2=p2))["primary"]["H2 attention mixing"]
    assert (h2["nested"]["grade"], h2["fixed"]["grade"]) == ("NULL", "INCONCLUSIVE")
    assert h2["verdict"] == "NULL"


def test_falls():
    p1 = {"loud": {"nested": _spread(0.25, seed=8)}}
    assert V.verdicts(_reps(P1=p1))["primary"]["H1 stats mixing"]["verdict"] == "FALLS"


def test_g2_failure_withholds_every_comparison_with_that_arm():
    reps = _reps(P2={"ctrl": {"nested": [-0.01, 0.12, -0.02]},
                     "loud": {"nested": _spread(0.55, seed=5), "heldout": [0.6] * N_LOUD}})
    v = V.verdicts(reps)
    assert not v["gates"]["G2_control"]["P2"]
    for c in (v["primary"]["H2 attention mixing"], v["secondary"]["P2 vs P0"],
              v["secondary"]["H3 mechanism"]):
        assert c["verdict"] == "WITHHELD" and c["withheld_by"] == ["P2"]
    assert v["primary"]["H1 stats mixing"]["verdict"] == "NULL"


def test_g1_failure_withholds_too_and_a_nan_pr_fails_it():
    reps = _reps()
    first = next(iter(reps["P1c"]["ctrl"]["runs"].values()))
    first["latent_pr"] = float("nan")
    v = V.verdicts(reps)
    assert not v["gates"]["G1_health"]["P1c"]
    assert v["primary"]["H1 stats mixing"]["withheld_by"] == ["P1c"]


def test_g3_is_reported_and_withholds_nothing():
    v = V.verdicts(_reps(P0={"loud": {"nested": _spread(0.25, seed=9)}}))
    assert not v["gates"]["G3_apparatus"]["pass"]
    assert v["primary"]["H2 attention mixing"]["verdict"] == "NULL"


@pytest.mark.parametrize("best,reading", [(0.56, "high"), (0.44, "about z"),
                                          (0.50, "unresolved")])
def test_d1_reading(best, reading):
    reps = _reps()
    for r in reps["P0"]["loud"]["runs"].values():
        r["d1"] = {**r["d1"], "concat": best}
    d = V.verdicts(reps)["D1"]
    assert d["best"] == "concat" and d["reading"] == reading
    assert d["z0"] == pytest.approx(0.40)


def test_a_report_from_the_wrong_arm_is_refused():
    reps = _reps()
    reps["P1"], reps["P1c"] = reps["P1c"], reps["P1"]
    with pytest.raises(ValueError, match="view_pool"):
        V.verdicts(reps)


# -- crossview_measure ------------------------------------------------------- #
def test_d1_sets_are_the_pilots_constructions_on_v():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    v = np.random.default_rng(0).random((5, 16, 8))
    s = M.d1_sets(v)
    assert s["concat"].shape == (5, 128) and s["C mean + std"].shape == (5, 16)
    assert s["D |DFT over view index|"].shape == (5, 9 * 8)
    assert s["G sorted across views"].shape == (5, 128)
    assert s["H mean + |dev| quantiles"].shape == (5, 40)
    assert np.allclose(s["G sorted across views"].reshape(5, 16, 8)[:, 0], v.min(1))
    assert np.allclose(s["D |DFT over view index|"].reshape(5, 9, 8)[:, 0], v.sum(1))


def test_nonfinite_flags_any_nan_or_inf_in_the_log():
    ok = [{"epoch": 0, "train_loss": 0.3, "latent_pr": 40.0}]
    assert not M.nonfinite(ok)
    assert M.nonfinite(ok + [{"epoch": 1, "train_loss": float("nan")}])
    assert M.nonfinite(json.loads('[{"latent_pr": Infinity}]'))


def test_measure_reads_a_planted_signal_and_the_dual_d1_agrees_with_nested_grid():
    from z_readout_alpha import nested_grid
    rng = np.random.default_rng(1)
    n = 160
    v = rng.normal(size=(n, 4, 6))
    fr = v[:, 1, 0] - v[:, 2, 3] + 0.3 * rng.normal(size=n)
    E = np.stack([rng.random(n), fr, rng.random(n)], 1)
    z = v.mean(1)
    splits = np.array(["train"] * 140 + ["test"] * 20)
    Y = rng.random((n, 3))
    row = M.measure(z, E, splits, Y, Y.copy(), [{"latent_pr": 12.0}], v)
    assert row["latent_pr"] == 12.0 and not row["nonfinite"]
    assert row["heldout_gain"] == float("inf")                 # a perfect prediction
    assert row["d1"]["concat"] > row["friction_nested"] + 0.3  # the mean loses the views
    ref, _ = nested_grid(v.reshape(n, -1), fr, M.FROZEN_GRID)
    assert row["d1"]["concat"] == pytest.approx(ref, abs=1e-8)
    assert set(row["nested_picks"]) <= set(M.FROZEN_GRID)


def test_encode_reads_an_mlx_run_and_check_cameras_guards_attn_runs(tmp_path):
    from pseudomarble.models import mlx_net
    if not mlx_net._HAVE_MLX:
        pytest.skip("MLX runs on the Mac only")
    from dataclasses import replace

    import test_view_pool as TV
    cfg = replace(TV._cfg("attn"))
    run = tmp_path / "run"
    run.mkdir()
    model = mlx_net.build_model(cfg)
    model.save_weights(str(run / "model.safetensors"))
    (run / "metrics.json").write_text(json.dumps(
        {"config": cfg.__dict__, "history": [{"epoch": 0, "latent_pr": 9.0}]}))
    imgs = np.random.default_rng(2).random((3, TV.N_VIEWS, 16, 16, 3)).astype("float32")
    enc = M.encode(str(run), imgs, views=True, chunk=2)
    assert enc["z"].shape == (3, 16) and enc["v"].shape == (3, TV.N_VIEWS, 16)
    world = tmp_path / "world"
    TV._world(str(world))
    M.check_cameras(enc["cfg"], str(world))                    # the same cameras
    other = tmp_path / "other"
    TV._world(str(other), layout="sphere")
    with pytest.raises(SystemExit, match="cameras differ"):
        M.check_cameras(enc["cfg"], str(other))


def test_encode_runs_on_the_requested_device(tmp_path):
    """CPU encoding is MLX's exact float32 path; the GPU agrees to its matmul
    rounding (test_view_pool's loose tolerance)."""
    from pseudomarble.models import mlx_net
    if not mlx_net._HAVE_MLX:
        pytest.skip("MLX runs on the Mac only")
    import mlx.core as mx

    import test_view_pool as TV
    cfg = TV._cfg("stats")
    run = tmp_path / "run"
    run.mkdir()
    model = mlx_net.build_model(cfg)
    model.save_weights(str(run / "model.safetensors"))
    (run / "metrics.json").write_text(json.dumps({"config": cfg.__dict__, "history": []}))
    imgs = np.random.default_rng(3).random((4, TV.N_VIEWS, 16, 16, 3)).astype("float32")
    cpu = M.encode(str(run), imgs, views=True, chunk=3, device="cpu")
    gpu = M.encode(str(run), imgs, views=True, chunk=3, device="gpu")
    with mx.stream(mx.cpu):
        ref = np.array(model(mx.array(imgs))["z"])
    assert np.allclose(cpu["z"], ref, rtol=1e-6, atol=1e-6)    # chunking moves an ulp
    assert np.allclose(gpu["z"], cpu["z"], rtol=1e-2, atol=1e-2)
    assert np.allclose(gpu["v"], cpu["v"], rtol=1e-2, atol=1e-2)


def test_d1_refuses_a_world_whose_scenes_order_their_cameras_differently(tmp_path,
                                                                         monkeypatch):
    """D1's concat and DFT index views by position; the check runs before any
    image is loaded (this world has none)."""
    import test_view_pool as TV
    TV._world(str(tmp_path), perturb=1)
    monkeypatch.setattr(sys, "argv", ["crossview_measure.py", "--data", str(tmp_path),
                                      "--runs", "unused", "--d1"])
    with pytest.raises(ValueError, match="different cameras"):
        M.main()
