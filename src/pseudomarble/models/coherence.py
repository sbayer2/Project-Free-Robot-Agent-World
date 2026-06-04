"""The coherence loss / metric — the project's core, still-untested idea.

Motivation (docs/TAXONOMY_NOTES.md): if appearance and physics are two
projections of one underlying understanding (the "eigenvector"), then perturbing
the shared latent should move *both* outputs *consistently*. A model whose render
and physics decoders read from genuinely shared structure will show high
coherence; two independently-trained models will not.

We measure this without assuming the two output spaces are comparable directly
(pixels vs. friction scalars). Instead we compare *how each decoder responds to
the same latent perturbation*, via the alignment of Jacobian-vector products.

Given latent ``z``, a random unit direction ``d``, and the two decoders
``f_render``, ``f_physics``:

    j_r = d/dε f_render(z + ε d)   |ε=0      (render sensitivity to d)
    j_p = d/dε f_physics(z + ε d)  |ε=0      (physics sensitivity to d)

Coherence asks: across many directions ``d``, are the *magnitudes* of j_r and
j_p correlated? If a direction barely changes appearance, it should barely change
physics, and vice-versa — that mutual responsiveness is the signal. The
coherence loss is ``1 - corr(||j_r||, ||j_p||)`` over sampled directions.

IMPORTANT — direction sampling: ``directions`` must be sampled iid Gaussian
(``N(0, I)``), NOT normalized to the unit sphere. Unit-normalized directions
introduce a simplex artifact: the sensitivities of two *disjoint* latent
subspaces become anti-correlated (their squared projections sum to 1), so truly
independent decoders would score negative rather than ~0. With Gaussian
directions, disjoint subspaces are genuinely independent -> score ~0, coupled
-> positive, which is the interpretation we want. ``sample_directions`` below
does this correctly.

This module provides a backend-agnostic reference using finite differences so
the *definition* is executable and testable; an MLX autograd version (faster,
exact) is provided when ``mlx`` is importable.
"""

from __future__ import annotations

import random
from typing import Callable, List, Sequence


def sample_directions(dim: int, n: int, seed: int = 0) -> List[List[float]]:
    """Sample ``n`` iid Gaussian directions in ``dim`` dimensions.

    Gaussian (NOT unit-normalized) — see the module docstring for why this
    matters for the coherence interpretation.
    """
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(n)]


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n == 0 or n != len(ys):
        raise ValueError("xs and ys must be non-empty and equal length")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    if denom == 0:
        return 0.0
    return cov / denom


def coherence_score_finite_diff(
    z: Sequence[float],
    f_render: Callable[[Sequence[float]], Sequence[float]],
    f_physics: Callable[[Sequence[float]], Sequence[float]],
    directions: Sequence[Sequence[float]],
    eps: float = 1e-3,
) -> float:
    """Backend-agnostic coherence in [-1, 1]; higher == more coupled.

    Pure-Python reference implementation (finite differences). Intended for
    tests and for reasoning about the metric, not for training-time speed.
    """

    def _l2(a: Sequence[float], b: Sequence[float]) -> float:
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5

    r0 = f_render(z)
    p0 = f_physics(z)
    rmag, pmag = [], []
    for d in directions:
        zp = [zi + eps * di for zi, di in zip(z, d)]
        rmag.append(_l2(f_render(zp), r0) / eps)
        pmag.append(_l2(f_physics(zp), p0) / eps)
    return _pearson(rmag, pmag)


def coherence_loss_finite_diff(*args, **kwargs) -> float:
    """``1 - coherence_score`` -> a quantity to minimize (in [0, 2])."""
    return 1.0 - coherence_score_finite_diff(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Optional fast path: exact Jacobian-vector products via MLX autograd.
# --------------------------------------------------------------------------- #
def coherence_loss_mlx(z, f_render, f_physics, directions):  # pragma: no cover
    """MLX implementation using jvp; available only when ``mlx`` is installed.

    ``z`` and ``directions`` are ``mlx.core.array`` objects; ``f_render`` and
    ``f_physics`` are MLX-traceable callables. Returns an ``mx.array`` scalar.
    """
    try:
        import mlx.core as mx  # type: ignore
    except Exception as exc:
        raise RuntimeError("coherence_loss_mlx requires mlx (Apple-silicon)") from exc

    r_mags, p_mags = [], []
    for d in directions:
        _, jr = mx.jvp(f_render, (z,), (d,))
        _, jp = mx.jvp(f_physics, (z,), (d,))
        r_mags.append(mx.linalg.norm(jr))
        p_mags.append(mx.linalg.norm(jp))
    r = mx.stack(r_mags)
    p = mx.stack(p_mags)
    r = r - r.mean()
    p = p - p.mean()
    denom = mx.sqrt((r * r).sum() * (p * p).sum()) + 1e-8
    corr = (r * p).sum() / denom
    return 1.0 - corr
