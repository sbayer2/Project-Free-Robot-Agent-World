"""Framework-agnostic loss reference (pure-Python, list-based).

The MLX model (``models/mlx_net.py``) computes its loss with ``mlx`` ops, but the
*math* is mirrored here on plain Python lists so it is unit-tested in any session
(no mlx/numpy needed) and documents exactly what the training objective is.

Objective for the encoder + behavior head (this build step):

    total = MSE(behavior_pred, behavior_target)
          + essence_weight * MSE(essence_pred, essence_target)

The behavior term is the real target — predicting drop/tilt/push outcomes. The
essence term is a light auxiliary that asks the latent to also be decodable to
(density, friction, restitution); it both stabilizes training and lets us later
measure how well the model *recovers* the hidden essence from images alone.
"""

from __future__ import annotations

from typing import Dict, Sequence

Matrix = Sequence[Sequence[float]]


def mse(pred: Matrix, target: Matrix) -> float:
    """Mean squared error over a (batch x dim) pair of nested lists."""
    total, n = 0.0, 0
    for pr, tg in zip(pred, target):
        if len(pr) != len(tg):
            raise ValueError(f"row length mismatch: {len(pr)} vs {len(tg)}")
        for a, b in zip(pr, tg):
            total += (a - b) ** 2
            n += 1
    if n == 0:
        raise ValueError("empty batch")
    return total / n


def _flatten_sample(s) -> list:
    out: list = []

    def rec(v):
        if isinstance(v, (list, tuple)):
            for e in v:
                rec(e)
        else:
            out.append(float(v))

    rec(s)
    return out


def flatten_batch(x) -> list:
    """Flatten each sample of a (B, ...) nested list to (B, D) — e.g. images."""
    return [_flatten_sample(s) for s in x]


def combined_loss(
    behavior_pred: Matrix,
    behavior_target: Matrix,
    essence_pred: Matrix,
    essence_target: Matrix,
    essence_weight: float = 0.3,
    render_pred=None,
    render_target=None,
    render_weight: float = 1.0,
    behavior_weight: float = 1.0,
) -> Dict[str, float]:
    """Total + component losses; mirrors the MLX training objective.

    With the render head: total = behavior_weight * behavior MSE
    + essence_weight * essence MSE + render_weight * reconstruction MSE (the render
    target is the mean view). ``behavior_weight``/``render_weight`` default to 1.0
    so the default is the full shared objective; zero one to build the coherence
    experiment's independent render-only / behavior-only models.
    """
    b = mse(behavior_pred, behavior_target)
    e = mse(essence_pred, essence_target)
    out = {"behavior": b, "essence": e}
    total = behavior_weight * b + essence_weight * e
    if render_pred is not None:
        r = mse(flatten_batch(render_pred), flatten_batch(render_target))
        out["render"] = r
        total += render_weight * r
    out["total"] = total
    return out


def per_field_mse(pred: Matrix, target: Matrix) -> list:
    """MSE per output dimension — useful to see which behavior fields are hard."""
    if not pred:
        raise ValueError("empty batch")
    dim = len(pred[0])
    sums = [0.0] * dim
    for pr, tg in zip(pred, target):
        for j in range(dim):
            sums[j] += (pr[j] - tg[j]) ** 2
    n = len(pred)
    return [s / n for s in sums]
