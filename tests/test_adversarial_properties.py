"""Adversarial property-based tests (hypothesis).

These are the "adversarial input" half of test verification: instead of a few
hand-picked cases, hypothesis searches the input space for any case that breaks an
invariant. They target the pure-Python cores where logic — not numerics — lives,
so they run fast and deterministically in CI. Skips cleanly if hypothesis is not
installed.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
    _HAVE = True
except Exception:  # pragma: no cover
    _HAVE = False

if _HAVE:
    from pseudomarble import probes as P
    from pseudomarble.materials import (
        DENSITY_RANGE,
        FRICTION_RANGE,
        RESTITUTION_RANGE,
        MaterialSampler,
    )
    from pseudomarble.models import losses
    from pseudomarble.models.coherence import (
        coherence_score_finite_diff,
        sample_directions,
    )
    from pseudomarble.splits import RegionHoldout, make_combination_split

    SETTINGS = settings(max_examples=150, deadline=None,
                        suppress_health_check=[HealthCheck.too_slow])

    names = st.text("abcdefghijklmnop", min_size=1, max_size=4)

    # --- splits: coverage + disjointness must always hold ----------------- #
    @SETTINGS
    @given(
        shapes=st.lists(names, min_size=1, max_size=6, unique=True),
        materials=st.lists(names, min_size=1, max_size=6, unique=True),
        frac=st.floats(min_value=0.0, max_value=0.9),
        seed=st.integers(0, 10_000),
    )
    def test_combination_split_invariants(shapes, materials, frac, seed):
        s = make_combination_split(shapes, materials, frac, seed)
        assert set(s.train_pairs).isdisjoint(s.test_pairs)
        # Every shape and material survives in train.
        assert {sh for sh, _ in s.train_pairs} == set(shapes)
        assert {m for _, m in s.train_pairs} == set(materials)
        # Holdout never exceeds the requested fraction of the grid.
        assert len(s.test_pairs) <= round(len(shapes) * len(materials) * frac)

    # --- probes.behavior_vector: stable layout, always finite ------------- #
    outcome = st.fixed_dictionaries({
        f: st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)
        for f in P.OUTCOME_FIELDS
    })

    @SETTINGS
    @given(kinds=st.lists(st.sampled_from(P.PROBE_ORDER), max_size=5),
           out=outcome)
    def test_behavior_vector_is_stable_and_finite(kinds, out):
        recs = [{"probe": k, "outcome": out} for k in kinds]
        vec = P.behavior_vector(recs, normalize=True)
        assert len(vec) == P.BEHAVIOR_DIM
        assert all(math.isfinite(v) for v in vec)

    # --- probes.summarize: outcomes are well-formed for any trajectory ---- #
    finite = st.floats(-50, 50, allow_nan=False, allow_infinity=False)

    @SETTINGS
    @given(frames=st.lists(
        st.fixed_dictionaries({
            "pos": st.tuples(finite, finite, finite),
            "ux": finite, "uy": finite,
            "uz": st.floats(0.1, 50, allow_nan=False),  # keep up-vector non-degenerate
        }),
        min_size=2, max_size=40))
    def test_summarize_outcomes_are_well_formed(frames):
        traj = [{"t": round(0.1 * i, 4),
                 "pos": list(f["pos"]), "up": [f["ux"], f["uy"], f["uz"]]}
                for i, f in enumerate(frames)]
        o = P.summarize(traj)
        assert o.n_bounces >= 0
        assert o.settle_time >= 0.0
        assert o.slid_distance >= 0.0
        assert 0.0 <= o.final_tilt_deg <= 180.0
        assert math.isfinite(o.path_length)

    # --- coherence score is always a correlation in [-1, 1] --------------- #
    @SETTINGS
    @given(dim=st.integers(2, 6), seed=st.integers(0, 9999),
           scale=st.floats(0.1, 5.0))
    def test_coherence_score_is_bounded(dim, seed, scale):
        import random
        rng = random.Random(seed)
        Mr = [[rng.gauss(0, scale) for _ in range(dim)] for _ in range(3)]
        Mb = [[rng.gauss(0, scale) for _ in range(dim)] for _ in range(2)]
        f_render = lambda z: [sum(w * x for w, x in zip(row, z)) for row in Mr]
        f_behavior = lambda z: [sum(w * x for w, x in zip(row, z)) for row in Mb]
        z = [rng.gauss(0, 1) for _ in range(dim)]
        dirs = sample_directions(dim, 32, seed=seed)
        score = coherence_score_finite_diff(z, f_render, f_behavior, dirs)
        assert -1.0001 <= score <= 1.0001

    # --- losses: non-negative, zero on identity --------------------------- #
    row = st.lists(st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                   min_size=1, max_size=8)

    @SETTINGS
    @given(a=st.lists(row, min_size=1, max_size=8))
    def test_mse_is_nonneg_and_zero_on_identity(a):
        # rectangularize: all rows to the min length
        w = min(len(r) for r in a)
        a = [r[:w] for r in a]
        assert losses.mse(a, a) == 0.0
        b = [[x + 1.0 for x in r] for r in a]
        assert losses.mse(a, b) >= 0.0

    # --- material sampler always produces in-range essence ---------------- #
    @SETTINGS
    @given(seed=st.integers(0, 100_000))
    def test_sampled_essence_in_range(seed):
        m = MaterialSampler(seed=seed).sample().material
        assert DENSITY_RANGE[0] <= m.physics.density <= DENSITY_RANGE[1]
        assert FRICTION_RANGE[0] <= m.physics.friction <= FRICTION_RANGE[1]
        assert RESTITUTION_RANGE[0] <= m.physics.restitution <= RESTITUTION_RANGE[1]
        assert 0.0 <= m.visual.roughness <= 1.0

    # --- region holdout: all-None never silently captures everything ------ #
    unit = st.floats(0.0, 1.0)

    @SETTINGS
    @given(d=unit, f=unit, r=unit, shape=names)
    def test_empty_region_holds_out_nothing(d, f, r, shape):
        ess = {"density": d, "friction": f, "restitution": r}
        assert RegionHoldout().contains(ess, shape) is False

else:  # pragma: no cover
    def test_hypothesis_unavailable():
        print("hypothesis not installed; skipping adversarial property tests")


if __name__ == "__main__":
    if not _HAVE:
        print("hypothesis not installed; skipping")
    else:
        fns = [v for k, v in sorted(globals().items())
               if k.startswith("test_") and callable(v)]
        for fn in fns:
            fn()
            print(f"ok  {fn.__name__}")
        print(f"\n{len(fns)} property tests passed")
