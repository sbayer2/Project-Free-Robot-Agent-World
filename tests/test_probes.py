"""Tests for probe outcome summarization (pure-Python, synthetic trajectories).

We feed hand-built trajectories that clearly represent toppling, sliding, and
bouncing, and check that ``summarize`` reports the right behavior. No MuJoCo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble import probes as P  # noqa: E402

UP = [0.0, 0.0, 1.0]


def _traj(zs=None, xs=None, ups=None, dt=0.1):
    n = len(zs or xs or ups)
    zs = zs or [0.2] * n
    xs = xs or [0.0] * n
    ups = ups or [UP] * n
    return [{"t": round(i * dt, 4), "pos": [xs[i], 0.0, zs[i]], "up": ups[i]}
            for i in range(n)]


def test_default_probes_are_drop_tilt_push():
    kinds = [s.kind for s in P.default_probes()]
    assert kinds == ["drop", "tilt", "push"]


def test_spec_to_dict_carries_kind():
    d = P.spec_to_dict(P.PushSpec())
    assert d["kind"] == "push" and "impulse" in d


def test_bouncing_is_counted():
    zs = [1.0, 0.6, 0.2, 0.5, 0.2, 0.35, 0.2, 0.2]
    out = P.summarize(_traj(zs=zs))
    assert out.n_bounces == 2
    assert abs(out.max_height - 1.0) < 1e-9
    assert not out.toppled


def test_toppling_is_detected():
    ups = [[0, 0, 1], [0.3, 0, 0.95], [0.7, 0, 0.7], [0.95, 0, 0.3], [1, 0, 0.02]]
    out = P.summarize(_traj(ups=ups))
    assert out.toppled
    assert out.final_tilt_deg > 80


def test_sliding_distance_is_measured():
    xs = [0.0, 0.05, 0.12, 0.20, 0.25, 0.27, 0.27]
    out = P.summarize(_traj(xs=xs))
    assert out.slid_distance > 0.2
    assert not out.toppled


def test_resting_object_settles_and_does_not_topple():
    out = P.summarize(_traj(zs=[0.2, 0.2, 0.2, 0.2]))
    assert not out.toppled
    assert out.slid_distance < 1e-6
    assert out.n_bounces == 0


def test_soft_topple_probability():
    # 50 deg is the threshold; > 50 counts as toppled.
    assert P.soft_topple_probability([10.0, 20.0, 30.0]) == 0.0      # none topple
    assert P.soft_topple_probability([60.0, 90.0]) == 1.0            # all topple
    assert P.soft_topple_probability([10.0, 90.0, 90.0, 10.0]) == 0.5  # smooth, in [0,1]
    # exactly at the threshold does NOT count (strict >), matching summarize()
    assert P.soft_topple_probability([50.0, 90.0]) == 0.5
    try:
        P.soft_topple_probability([])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on empty input")


def test_outcome_vector_matches_field_order():
    out = P.summarize(_traj(zs=[0.5, 0.2, 0.2]))
    vec = P.outcome_vector(out)
    assert len(vec) == len(P.OUTCOME_FIELDS)
    assert vec[0] in (0.0, 1.0)  # toppled bool -> float


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
