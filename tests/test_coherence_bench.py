"""Tests for the coherence benchmark harness.

Two parts: (1) the harness logic discriminates coupled vs. disjoint decoders
(deterministic, pure-Python); (2) it runs end-to-end on the numpy model and the
independent baseline scores near zero (the control).
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.models.coherence_bench import (  # noqa: E402
    compare,
    independent_coherence,
    mean_coherence,
)

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False

from pseudomarble.config import ModelConfig  # noqa: E402

SMALL = replace(ModelConfig(), conv_channels=(4, 8), latent_dim=12,
                behavior_head_width=12, essence_head_width=8, image_size=16)


def test_mean_coherence_high_for_coupled_decoders():
    # Both heads read the SAME latent dims -> sensitivities co-vary.
    f_render = lambda z: [z[0] + z[1], z[0] - z[1]]
    f_behavior = lambda z: [2 * z[0] - z[1]]
    zs = [[0.3, -0.7, 0.1], [0.5, 0.2, -0.4], [-0.2, 0.6, 0.3]]
    assert mean_coherence(zs, f_render, f_behavior, n_dirs=64, seed=1) > 0.4


def test_mean_coherence_low_for_disjoint_decoders():
    # Render reads dims {0,1}; behavior reads dims {2,3}. Disjoint -> ~0.
    f_render = lambda z: [z[0] + z[1]]
    f_behavior = lambda z: [z[2] - z[3]]
    zs = [[0.3, -0.7, 0.1, 0.5], [0.5, 0.2, -0.4, 0.1], [-0.2, 0.6, 0.3, -0.3]]
    assert abs(mean_coherence(zs, f_render, f_behavior, n_dirs=200, seed=2)) < 0.25


def _skip_no_numpy():
    if not _HAVE_NUMPY:  # pragma: no cover
        print("numpy unavailable; skipping")
        return True
    return False


def test_compare_runs_and_reports_valid_numbers():
    if _skip_no_numpy():
        return
    from pseudomarble.models.numpy_net import NumpyModel

    shared = NumpyModel(SMALL, seed=0)
    render_only = NumpyModel(SMALL, seed=1)
    behavior_only = NumpyModel(SMALL, seed=2)
    images = np.random.default_rng(0).random((2, 2, 16, 16, 3)).astype("float32")

    rep = compare(shared, render_only, behavior_only, images, n_dirs=32, seed=0)
    assert set(rep) == {"shared_coherence", "independent_coherence", "gap", "n_samples"}
    assert rep["n_samples"] == 2
    for k in ("shared_coherence", "independent_coherence"):
        assert -1.0 <= rep[k] <= 1.0
    assert abs(rep["gap"] - (rep["shared_coherence"] - rep["independent_coherence"])) < 1e-9


def test_compare_reports_learned_coherence_with_control():
    # With an untrained-shared control, the honest signal is learned_coherence
    # = shared - architectural. (For two untrained models it should be small.)
    if _skip_no_numpy():
        return
    from pseudomarble.models.numpy_net import NumpyModel

    shared = NumpyModel(SMALL, seed=0)
    render_only = NumpyModel(SMALL, seed=1)
    behavior_only = NumpyModel(SMALL, seed=2)
    untrained = NumpyModel(SMALL, seed=4)
    images = np.random.default_rng(0).random((2, 2, 16, 16, 3)).astype("float32")

    rep = compare(shared, render_only, behavior_only, images,
                  untrained_shared_model=untrained, n_dirs=32, seed=0)
    assert "architectural_coherence" in rep and "learned_coherence" in rep
    assert abs(rep["learned_coherence"]
               - (rep["shared_coherence"] - rep["architectural_coherence"])) < 1e-9


def test_independent_baseline_is_near_zero():
    # The control must sit near zero: two separate models share no latent.
    if _skip_no_numpy():
        return
    from pseudomarble.models.numpy_net import NumpyModel

    render_only = NumpyModel(SMALL, seed=1)
    behavior_only = NumpyModel(SMALL, seed=2)
    rng = np.random.default_rng(3)
    zs_r = [list(rng.standard_normal(SMALL.latent_dim)) for _ in range(3)]
    zs_b = [list(rng.standard_normal(SMALL.latent_dim)) for _ in range(3)]
    val = independent_coherence(render_only, behavior_only, zs_r, zs_b, n_dirs=64, seed=0)
    assert abs(val) < 0.4, f"independent baseline not near zero: {val}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
