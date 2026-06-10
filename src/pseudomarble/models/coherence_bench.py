"""Coherence benchmark harness — the experiment the project was built to run.

The question (README / TAXONOMY_NOTES): are appearance and behavior **two
projections of one understanding**, or **two outputs glued together**? We answer
it with a *measurement*, not a vibe.

Coherence (``models/coherence.py``): nudge the latent ``z`` in random directions
and ask whether the directions that change the *render* a lot also change the
*behavior* a lot. If render-sensitivity and behavior-sensitivity correlate, the
two heads read shared structure (high coherence). If not, they live in private
subspaces (coherence ~ 0).

The comparison:
  * **Shared model** — one encoder, one latent ``z``, both heads read it. We
    measure coherence directly on ``z``.
  * **Independent baseline** — a render-only model and a behavior-only model with
    *separate* latents. We form the joint latent ``z = [z_render ; z_behavior]``;
    the render reads only the first half, the behavior only the second. By
    construction their sensitivities are independent, so this scores ~0 — the
    control for "genuinely separate representations".

The honest reading of the result:
  * shared coherence **>> 0** (above the ~0 baseline) ⇒ the shared latent really
    couples appearance and physics — evidence for the "eigenvector".
  * shared coherence **≈ 0** ⇒ the network learned to use *disjoint* parts of
    ``z`` for the two tasks: "two outputs in one wrapper". A real, publishable
    null. The shared model is NOT guaranteed to win — that's what makes this a
    test rather than a demo.

CRITICAL CONTROL (measured, not hypothetical): a randomly-initialized shared
model already scores HIGH coherence (~0.45 in a quick numpy check) purely from
architecture — both heads read the same ``z``, so any latent nudge moves both.
The independent baseline is ~0. **Therefore the shared-vs-independent gap is NOT,
by itself, evidence of a learned eigenvector** — an untrained network shows it
too. The honest signal is *how much training raises coherence above the
untrained-shared baseline*: ``learned_coherence = trained_shared - untrained_shared``.
``compare`` computes this when given an untrained shared model. Note the
architectural baseline itself *varies across random seeds* (~0.36-0.46 observed),
so the real run should average it over several untrained seeds and require the
*learned* coherence to clear that noise band. Always evaluate on **held-out
essence regions**; pair coherence with behavior generalization (does the shared
latent predict behavior better on unseen essences?).

Smooth vs. chaotic target (de-risking a false null): the physics-side head can be
the **behavior** head (drop/tilt/push outcomes — the headline, but ``toppled`` is
a threshold, so near tipping points its sensitivity is erratic and can drag
coherence DOWN artificially) or the **essence** head ((density, friction,
restitution) — a smooth function of z, robust to that chaos). Run both via
``compare_both_targets``: a *high* essence-coherence beside a *low* behavior-
coherence is the signature of the topple-chaos artifact, not a real absence of
coupling.

Coherence is a finite-difference (forward-only) measurement, so it runs with any
backend — including the numpy reference in a plain sandbox. ``numpy_model_decoders``
adapts a ``NumpyModel``; the same shape works for the MLX model on the Mac.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

from pseudomarble.models.coherence import (
    coherence_score_finite_diff,
    sample_directions,
)

Decoder = Callable[[Sequence[float]], List[float]]


def mean_coherence(zs: Sequence[Sequence[float]], f_render: Decoder,
                   f_behavior: Decoder, n_dirs: int = 48, eps: float = 1e-3,
                   seed: int = 0) -> float:
    """Average coherence over a set of latent vectors (e.g. held-out samples)."""
    if not zs:
        raise ValueError("need at least one latent vector")
    scores = []
    for i, z in enumerate(zs):
        z = list(z)
        dirs = sample_directions(len(z), n_dirs, seed=seed + i)
        scores.append(coherence_score_finite_diff(z, f_render, f_behavior, dirs, eps))
    return sum(scores) / len(scores)


def numpy_model_decoders(model, target: str = "behavior") -> Tuple[Decoder, Decoder]:
    """Wrap a model's render head and a chosen physics-side head as z->flat-list.

    ``target`` selects the physics-side decoder:
      * ``"behavior"`` — the drop/tilt/push outcome head (the headline target, but
        partly non-smooth: ``toppled`` is a threshold, so near tipping points its
        Jacobian is erratic and can bias coherence DOWNWARD — a false null).
      * ``"essence"`` — the (density, friction, restitution) head, a SMOOTH
        function of z. Coherence here is robust to the topple-chaos bias, so it's
        the de-risked read: "does appearance couple with the physical *properties*?"
    Report both; a low behavior-coherence next to a high essence-coherence points
    to the chaos artifact rather than a true absence of coupling.
    """
    import numpy as np

    head = {"behavior": model.behavior_from_z, "essence": model.essence_from_z}[target]

    def f_render(z: Sequence[float]) -> List[float]:
        img = model.decode(np.asarray([z], dtype="float32"))[0]
        return [float(v) for v in np.asarray(img).reshape(-1)]

    def f_target(z: Sequence[float]) -> List[float]:
        out = head(np.asarray([z], dtype="float32"))[0]
        return [float(v) for v in np.asarray(out).reshape(-1)]

    return f_render, f_target


def independent_coherence(render_model, behavior_model,
                          zs_render: Sequence[Sequence[float]],
                          zs_behavior: Sequence[Sequence[float]],
                          n_dirs: int = 48, seed: int = 0,
                          target: str = "behavior") -> float:
    """Control: coherence of two separate models via a joint disjoint latent.

    z = [z_render ; z_target]; render reads the first block, the physics-side head
    (``target``) the second. Independent by construction -> expected ~0.
    """
    fr, _ = numpy_model_decoders(render_model)
    _, fb = numpy_model_decoders(behavior_model, target)
    dr = len(zs_render[0])

    def f_render(z: Sequence[float]) -> List[float]:
        return fr(z[:dr])

    def f_behavior(z: Sequence[float]) -> List[float]:
        return fb(z[dr:])

    joint = [list(zr) + list(zb) for zr, zb in zip(zs_render, zs_behavior)]
    return mean_coherence(joint, f_render, f_behavior, n_dirs=n_dirs, seed=seed)


def compare(shared_model, render_model, behavior_model, images,
            untrained_shared_model=None, n_dirs: int = 48,
            seed: int = 0, target: str = "behavior") -> Dict[str, float]:
    """Run the coherence comparison on a batch of images.

    ``target`` ("behavior" | "essence") picks the physics-side head. "essence" is
    smooth and de-risks the topple-chaos bias (see ``numpy_model_decoders``); run
    both and report both. Returns ``{target, shared_coherence,
    independent_coherence, gap, n_samples}``. If ``untrained_shared_model`` (same
    architecture, freshly initialized) is given, also returns
    ``architectural_coherence`` and ``learned_coherence`` (= shared - architectural)
    — the honest "did training couple them?" signal.
    """
    import numpy as np

    imgs = np.asarray(images, dtype="float32")
    zs = [list(z) for z in np.asarray(shared_model.encode(imgs))]
    fr, ft = numpy_model_decoders(shared_model, target)
    shared = mean_coherence(zs, fr, ft, n_dirs=n_dirs, seed=seed)

    zs_r = [list(z) for z in np.asarray(render_model.encode(imgs))]
    zs_b = [list(z) for z in np.asarray(behavior_model.encode(imgs))]
    indep = independent_coherence(render_model, behavior_model, zs_r, zs_b,
                                  n_dirs=n_dirs, seed=seed, target=target)
    rep = {
        "target": target,
        "shared_coherence": shared,
        "independent_coherence": indep,
        "gap": shared - indep,
        "n_samples": len(zs),
    }
    if untrained_shared_model is not None:
        zs0 = [list(z) for z in np.asarray(untrained_shared_model.encode(imgs))]
        fr0, ft0 = numpy_model_decoders(untrained_shared_model, target)
        arch = mean_coherence(zs0, fr0, ft0, n_dirs=n_dirs, seed=seed)
        rep["architectural_coherence"] = arch
        rep["learned_coherence"] = shared - arch
    return rep


def compare_both_targets(shared_model, render_model, behavior_model, images,
                         untrained_shared_model=None, n_dirs: int = 48,
                         seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Run the comparison for BOTH targets — ``{"behavior": {...}, "essence": {...}}``.

    The recommended read: a high essence-coherence with a low behavior-coherence
    flags the topple-chaos artifact, not a true absence of coupling.
    """
    return {
        t: compare(shared_model, render_model, behavior_model, images,
                   untrained_shared_model=untrained_shared_model,
                   n_dirs=n_dirs, seed=seed, target=t)
        for t in ("behavior", "essence")
    }


def _demo() -> None:  # pragma: no cover - illustrative, not a result
    """Run the harness on UNTRAINED numpy models + random images.

    This shows the apparatus works and what the architectural baseline looks like
    BEFORE training. It is NOT the experiment's result — that needs trained models
    (MLX on the Mac) evaluated on held-out essence regions with real renders.
    """
    from dataclasses import replace

    import numpy as np

    from pseudomarble.config import ModelConfig
    from pseudomarble.models.numpy_net import NumpyModel

    cfg = replace(ModelConfig(), conv_channels=(8, 16), latent_dim=32,
                  behavior_head_width=32, essence_head_width=16, image_size=16)
    shared = NumpyModel(cfg, seed=0)
    render_only = NumpyModel(cfg, seed=1)
    behavior_only = NumpyModel(cfg, seed=2)
    untrained = NumpyModel(cfg, seed=3)  # architectural-baseline control
    images = np.random.default_rng(0).random((6, 4, 16, 16, 3)).astype("float32")
    reps = compare_both_targets(shared, render_only, behavior_only, images,
                                untrained_shared_model=untrained, n_dirs=48, seed=0)
    print("[coherence bench] UNTRAINED illustrative numbers (not the result):")
    for tgt, rep in reps.items():
        print(f"  target={tgt}")
        for k, v in rep.items():
            if k == "target":
                continue
            print(f"    {k:22s} {v:.4f}" if isinstance(v, float) else f"    {k:22s} {v}")
    print("  (run on trained MLX models over held-out essence regions for the real result)")


if __name__ == "__main__":
    _demo()
