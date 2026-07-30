"""The F22 coupling dials: how MUCH, and how LEGIBLY, appearance encodes physics.

Two orthogonal knobs on ``MaterialSampler`` (docs/ABO_COUPLING.md):

* ``coupling_alpha`` (α) — appearance is built from
  ``α*true + (1-α)*decoy``. α=0 is the **zero-coupling control**: appearance
  carries no information about physics, so a model that still scores there has
  found a leak, not an essence (preregistered P1, the hard gate).
* ``coupling_gain`` (g) — blends each appearance channel toward a dedicated,
  full-range, unconfounded map. This is the "letter size" axis that F21 never
  touched: F21 cleaned (noise) and lit (resolution/lighting) the signal without
  ever making it *stronger*.

The load-bearing guarantee tested here is **backward compatibility**: at the
defaults (α=1, g=1) not one RNG draw changes, so every F1-F21 dataset
regenerates byte-identically and the historical findings remain comparable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.materials import MaterialSampler  # noqa: E402


def _sample_n(n=64, **kw):
    s = MaterialSampler(seed=7, **kw)
    return [s.sample(material_id=f"m{i}") for i in range(n)]


def _corr(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return 0.0 if dx == 0 or dy == 0 else num / (dx * dy)


# --------------------------------------------------------------------------- #
# Backward compatibility — the guarantee the historical record depends on.
# --------------------------------------------------------------------------- #
def test_defaults_are_byte_identical_to_the_historical_sampler():
    """Defaults must not perturb the RNG stream, or pm_big stops reproducing."""
    baseline = _sample_n()
    explicit = _sample_n(coupling_alpha=1.0, coupling_gain=1.0)
    for a, b in zip(baseline, explicit):
        assert a.material.visual == b.material.visual
        assert a.material.physics == b.material.physics
        assert a.factors == b.factors


def test_alpha_changes_appearance_but_only_appearance():
    """α<1 must visibly move the appearance while leaving the scene identical."""
    same = _sample_n(coupling_alpha=1.0)
    diff = _sample_n(coupling_alpha=0.5)
    assert any(a.material.visual != b.material.visual for a, b in zip(same, diff))
    assert all(a.factors == b.factors for a, b in zip(same, diff))


def test_decoys_use_a_separate_stream_so_arms_stay_matched():
    """Regression guard. Decoys once drew from the main RNG, which shifted every
    later factor draw — so arms at different α sampled *different scenes* and
    their gains were not comparable. The ladder requires matched arms."""
    a0 = _sample_n(n=100, coupling_alpha=0.0)
    a1 = _sample_n(n=100, coupling_alpha=1.0)
    assert [s.factors for s in a0] == [s.factors for s in a1]


# --------------------------------------------------------------------------- #
# The physics answer key is never touched by either dial.
# --------------------------------------------------------------------------- #
def test_physics_is_identical_across_alpha():
    """α decouples APPEARANCE only. The ground truth must not move, otherwise
    arms would differ in their answer key and gains would be incomparable."""
    for alpha in (0.0, 0.5, 1.0):
        got = _sample_n(coupling_alpha=alpha)
        ref = _sample_n(coupling_alpha=1.0)
        for a, b in zip(got, ref):
            assert a.material.physics == b.material.physics
            assert a.factors == b.factors


def test_physics_is_identical_across_gain():
    for gain in (1.0, 3.0, 10.0):
        got = _sample_n(coupling_gain=gain)
        ref = _sample_n(coupling_gain=1.0)
        for a, b in zip(got, ref):
            assert a.material.physics == b.material.physics


# --------------------------------------------------------------------------- #
# P1: the zero-coupling control must actually be uncoupled.
# --------------------------------------------------------------------------- #
def test_alpha_zero_destroys_the_appearance_physics_correlation():
    """The control arm. If this leaks, P1 fails and no downstream arm is
    interpretable — the whole ladder rests on this being ~0."""
    coupled = _sample_n(n=400, coupling_alpha=1.0, coupling_gain=8.0)
    control = _sample_n(n=400, coupling_alpha=0.0, coupling_gain=8.0)

    def rough_vs_friction(samples):
        return abs(_corr([s.material.visual.roughness for s in samples],
                         [s.material.physics.friction for s in samples]))

    assert rough_vs_friction(coupled) > 0.5, "coupled arm should be strongly correlated"
    assert rough_vs_friction(control) < 0.20, "control arm must be ~uncoupled"
    assert rough_vs_friction(control) < rough_vs_friction(coupled)


def test_alpha_is_monotone_in_coupling_strength():
    """Turning the dial up must not lower the coupling it controls."""
    def strength(alpha):
        s = _sample_n(n=400, coupling_alpha=alpha, coupling_gain=8.0)
        return abs(_corr([x.material.visual.roughness for x in s],
                         [x.material.physics.friction for x in s]))

    weak, mid, strong = strength(0.0), strength(0.5), strength(1.0)
    assert weak < mid < strong


# --------------------------------------------------------------------------- #
# The gain dial: bigger letters, same alphabet.
# --------------------------------------------------------------------------- #
def test_gain_raises_legibility_of_the_friction_cue():
    """`grip` reaches appearance only via a 0.25 coefficient on roughness that is
    also driven by hardness (-0.55) and clarity (-0.70). Gain deconfounds it."""
    def legibility(gain):
        s = _sample_n(n=400, coupling_gain=gain)
        return abs(_corr([x.material.visual.roughness for x in s],
                         [x.material.physics.friction for x in s]))

    assert legibility(8.0) > legibility(1.0) + 0.10


def test_gain_leaves_the_marginal_appearance_in_range():
    """Channels stay in [0,1] at high gain — no clipping artifacts that would
    confound 'stronger coupling' with 'saturated renders'."""
    for s in _sample_n(n=200, coupling_gain=20.0):
        v = s.material.visual
        assert 0.0 <= v.metallic <= 1.0
        assert 0.02 <= v.roughness <= 1.0
        assert 0.0 <= v.transmission <= 1.0
        assert all(0.0 <= c <= 1.0 for c in v.base_color)


# --------------------------------------------------------------------------- #
# Guard rails.
# --------------------------------------------------------------------------- #
def test_out_of_range_dials_raise():
    for bad in (-0.1, 1.1):
        try:
            MaterialSampler(coupling_alpha=bad)
            raise AssertionError(f"coupling_alpha={bad} should raise")
        except ValueError:
            pass
    try:
        MaterialSampler(coupling_gain=0.5)
        raise AssertionError("coupling_gain<1 should raise")
    except ValueError:
        pass


def test_generator_threads_both_dials():
    """The flags must reach the sampler, not sit inert — cf. the PhysicsConfig
    inert-knob lesson in tests/test_physics_config_wiring.py."""
    from pseudomarble.data.generate_mujoco import assign_scenes
    from pseudomarble.splits import EXTRAPOLATION_REGION_HOLDOUT

    base = assign_scenes(["sphere", "box"], EXTRAPOLATION_REGION_HOLDOUT, 40, seed=3)
    ctrl = assign_scenes(["sphere", "box"], EXTRAPOLATION_REGION_HOLDOUT, 40, seed=3,
                         coupling_alpha=0.0)
    loud = assign_scenes(["sphere", "box"], EXTRAPOLATION_REGION_HOLDOUT, 40, seed=3,
                         coupling_gain=8.0)
    assert base != ctrl, "--coupling-alpha is inert"
    assert base != loud, "--coupling-gain is inert"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
