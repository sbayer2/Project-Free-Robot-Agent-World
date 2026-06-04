"""Tests for the coherence metric (backend-agnostic reference implementation).

These validate the *definition* of coherence, not a trained model: a coupled
pair of decoders (both reading the same latent coordinate) should score high; an
independent pair (reading disjoint coordinates) should score near zero.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.models.coherence import (  # noqa: E402
    coherence_loss_finite_diff,
    coherence_score_finite_diff,
    sample_directions,
)


def test_coupled_decoders_score_high():
    # Both decoders driven by the SAME latent dims -> shared sensitivity.
    f_render = lambda z: [z[0] + z[1], z[0] - z[1]]
    f_physics = lambda z: [2 * z[0] - z[1]]
    z = [0.3, -0.7, 0.1, 0.5]
    dirs = sample_directions(len(z), 64, seed=1)
    score = coherence_score_finite_diff(z, f_render, f_physics, dirs)
    assert score > 0.5, f"expected coupled score high, got {score}"


def test_independent_decoders_score_low():
    # Render reads dims {0,1}; physics reads dims {2,3}. Disjoint -> ~no coupling.
    # Gaussian (unnormalized) directions are required here: unit directions would
    # make disjoint subspaces anti-correlated (the simplex artifact).
    f_render = lambda z: [z[0] + z[1], z[0] - z[1]]
    f_physics = lambda z: [z[2] - z[3]]
    z = [0.3, -0.7, 0.1, 0.5]
    dirs = sample_directions(len(z), 400, seed=2)
    score = coherence_score_finite_diff(z, f_render, f_physics, dirs)
    assert abs(score) < 0.25, f"expected independent score near 0, got {score}"


def test_loss_is_one_minus_score():
    f_render = lambda z: [z[0]]
    f_physics = lambda z: [z[0]]
    z = [0.2, 0.4]
    dirs = sample_directions(len(z), 32, seed=3)
    score = coherence_score_finite_diff(z, f_render, f_physics, dirs)
    loss = coherence_loss_finite_diff(z, f_render, f_physics, dirs)
    assert abs((1.0 - score) - loss) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
