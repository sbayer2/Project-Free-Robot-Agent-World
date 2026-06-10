"""Tests for the continuous MaterialSampler (the fix for the 10-row lookup).

We check three things that make the task meaningful:
  * essence values stay in plausible physical ranges,
  * appearance is *predictive* of physics (so the coupling exists to learn),
  * appearance is *not deterministic* (so it can't be a perfect lookup).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.materials import (  # noqa: E402
    DENSITY_RANGE,
    FRICTION_RANGE,
    RESTITUTION_RANGE,
    MaterialSampler,
    names,
)


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / ((vx * vy) ** 0.5 or 1.0)


def test_sampler_is_deterministic():
    a = MaterialSampler(seed=4).sample().material
    b = MaterialSampler(seed=4).sample().material
    assert a.physics.density == b.physics.density
    assert a.visual.base_color == b.visual.base_color


def test_essence_stays_in_range():
    s = MaterialSampler(seed=1)
    for _ in range(300):
        p = s.sample().material.physics
        assert DENSITY_RANGE[0] <= p.density <= DENSITY_RANGE[1]
        assert FRICTION_RANGE[0] <= p.friction <= FRICTION_RANGE[1]
        assert RESTITUTION_RANGE[0] <= p.restitution <= RESTITUTION_RANGE[1]


def test_factors_recorded_in_unit_range():
    sample = MaterialSampler(seed=2).sample()
    for v in sample.factors.values():
        assert 0.0 <= v <= 1.0
    assert sample.nearest_anchor in names()


def test_appearance_is_predictive_but_not_deterministic():
    s = MaterialSampler(seed=7)
    metallic, log_density = [], []
    for _ in range(500):
        m = s.sample().material
        metallic.append(m.visual.metallic)
        log_density.append(math.log(m.physics.density))
    r = _pearson(metallic, log_density)
    # Predictive: heavier things look more metallic. But noisy: not a perfect map.
    assert 0.3 < r < 0.99, f"coupling correlation out of expected band: {r}"


def test_appearance_formula_directions():
    # Pin the SIGN of each term in the appearance map (no magic numbers): this
    # kills operator-swap mutations in _appearance_from_factors without making the
    # test brittle to the exact coefficients. noise=0 makes it deterministic.
    s = MaterialSampler(seed=0, appearance_noise=0.0)

    def ap(h, g, k, c):
        return s._appearance_from_factors(h, g, k, c)

    assert ap(0.9, 0.5, 0.5, 0.0).metallic > ap(0.1, 0.5, 0.5, 0.0).metallic   # heavier->metallic
    assert ap(0.9, 0.5, 0.5, 0.9).metallic < ap(0.9, 0.5, 0.5, 0.0).metallic   # clarity suppresses
    assert ap(0.5, 0.5, 0.9, 0.0).roughness < ap(0.5, 0.5, 0.1, 0.0).roughness  # harder->smoother
    assert ap(0.5, 0.9, 0.5, 0.0).roughness > ap(0.5, 0.1, 0.5, 0.0).roughness  # grippier->rougher
    assert ap(0.5, 0.5, 0.5, 0.9).transmission > ap(0.5, 0.5, 0.5, 0.05).transmission  # clarity


def test_physics_formula_directions():
    s = MaterialSampler(seed=0, appearance_noise=0.0)
    assert s._physics_from_factors(0.9, 0.5, 0.5).density > s._physics_from_factors(0.1, 0.5, 0.5).density
    assert s._physics_from_factors(0.5, 0.9, 0.5).friction > s._physics_from_factors(0.5, 0.1, 0.5).friction
    assert s._physics_from_factors(0.5, 0.5, 0.9).restitution > s._physics_from_factors(0.5, 0.5, 0.1).restitution


def test_nearest_anchor_recovers_seed_material():
    # Sampling tightly around an anchor must resolve to that anchor — kills the
    # distance-metric operator mutation in nearest_anchor().
    ms = MaterialSampler(seed=1).sample_near("steel", jitter=0.0)
    assert ms.nearest_anchor == "steel"


def test_materials_actually_vary():
    s = MaterialSampler(seed=3)
    densities = {round(s.sample().material.physics.density) for _ in range(50)}
    assert len(densities) > 40  # continuous, not a handful of buckets


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
