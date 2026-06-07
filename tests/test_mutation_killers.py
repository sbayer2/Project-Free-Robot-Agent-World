"""Exact-value tests that pin down logic the looser tests left unverified.

Written in response to the mutation-testing agent (scripts/mutation_check.py):
each assertion here is designed to fail if a specific operator/constant in the
core math is altered, so green means the computation is actually correct, not
merely non-crashing.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble import probes as P  # noqa: E402
from pseudomarble.models.coherence import (  # noqa: E402
    _pearson,
    coherence_score_finite_diff,
    sample_directions,
)
from pseudomarble.splits import (  # noqa: E402
    RegionHoldout,
    assign_scene_materials,
    make_combination_split,
)


# --- coherence math --------------------------------------------------------- #
def test_pearson_rejects_bad_input():
    for xs, ys in (([], []), ([1.0, 2.0], [1.0])):
        try:
            _pearson(xs, ys)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_identical_decoders_score_exactly_one():
    f = lambda z: [z[0], z[1], z[2]]
    z = [0.3, -0.7, 0.2]
    dirs = sample_directions(3, 40, seed=1)
    assert abs(coherence_score_finite_diff(z, f, f, dirs) - 1.0) < 1e-6


# --- probes.summarize exact values ----------------------------------------- #
def _traj(xs, zs, ups, t0=1.0, dt=0.1):
    return [{"t": round(t0 + dt * i, 4), "pos": [xs[i], 0.0, zs[i]], "up": ups[i]}
            for i in range(len(xs))]


def test_summarize_requires_two_frames():
    try:
        P.summarize([{"t": 0.0, "pos": [0, 0, 0], "up": [0, 0, 1]}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for <2 frames")


def test_summarize_exact_slide():
    o = P.summarize(_traj([0.5, 0.6, 0.8], [0.2, 0.2, 0.2], [[0, 0, 1]] * 3))
    assert abs(o.slid_distance - 0.3) < 1e-9     # |0.8 - 0.5|
    assert abs(o.path_length - 0.3) < 1e-9       # 0.1 + 0.2
    assert abs(o.max_height - 0.2) < 1e-9
    assert o.n_bounces == 0
    assert abs(o.settle_time - 0.2) < 1e-9       # t offset 1.0 -> nonzero start
    assert abs(o.final_tilt_deg - 0.0) < 1e-9


def test_summarize_exact_topple_angle():
    o = P.summarize(_traj([0, 0], [0.2, 0.2], [[0, 0, 1], [1, 0, 0]]))
    assert abs(o.final_tilt_deg - 90.0) < 1e-6
    assert o.toppled


def test_summarize_counts_bounces_exactly():
    zs = [1.0, 0.6, 0.2, 0.5, 0.2, 0.35, 0.2, 0.2]
    o = P.summarize(_traj([0.0] * 8, zs, [[0, 0, 1]] * 8))
    assert o.n_bounces == 2
    assert abs(o.max_height - 1.0) < 1e-9


# --- splits validation / summary / assignment ------------------------------ #
def test_holdout_frac_bounds_enforced():
    for bad in (1.0, -0.1, 1.5):
        try:
            make_combination_split(["a"], ["b"], bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for frac={bad}")
    make_combination_split(["a", "b"], ["c", "d"], 0.0)  # valid extreme


def test_combination_summary_counts():
    s = make_combination_split(["a", "b"], ["c", "d"], 0.0, seed=0)
    sm = s.summary()
    assert sm["n_train"] == 4
    assert sm["n_test"] == 0
    assert sm["n_shapes"] == 2
    assert sm["n_materials"] == 2


def test_assign_scene_materials_count_exact():
    s = make_combination_split(["a", "b", "c"], ["d", "e", "f"], 0.3, seed=1)
    recs = assign_scene_materials(["a", "b", "c"], ["d", "e", "f"], s, 100, seed=1)
    assert len(recs) == 100
    assert {r["split"] for r in recs} <= {"train", "test"}


def test_region_holdout_is_inclusive_on_bounds():
    h = RegionHoldout(friction=(0.5, 0.8))
    assert h.contains({"density": 0, "friction": 0.5, "restitution": 0}, "x")
    assert h.contains({"density": 0, "friction": 0.8, "restitution": 0}, "x")
    assert not h.contains({"density": 0, "friction": 0.81, "restitution": 0}, "x")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
