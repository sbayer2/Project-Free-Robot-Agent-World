# Taxonomy notes — the conceptual lineage

This file preserves the argument that motivates the code, so the *why* travels
with the *what*. It is research framing, not settled fact.

## 1. The taxonomy (the shadow)

Fei-Fei Li / World Labs, *A Functional Taxonomy of World Models*, organizes the
field around the POMDP loop (action → state → observation → action) into three
functional classes:

- **Renderers** — output **pixels** for human eyes; optimize visual fidelity
  (text-to-video, Genie). No explicit 3D/state.
- **Simulators** — output **state**; optimize geometric/physical/dynamic
  fidelity for agents and professionals (RL envs, robot controllers).
- **Planners** — output **actions** given observations + goals; "the inverse of
  the renderer" (VLA / world-action models).

The essay's thesis: these are **three projections of a single underlying
understanding**, converging toward a unified model that switches output
modalities.

## 2. The critique — the causal arrow may be reversed

The taxonomy presents renderer → simulator → planner as ascending structural
depth, with **simulation as the linchpin**. But the convergence evidence cuts the
other way:

- Video **renderers** are being repurposed as **planning** backbones.
- Marble emits **Gaussian splats + collision meshes from one model**.

If a pixel-trained renderer develops internally repurposable structure *without
explicit state supervision*, the three "functions" may be **one computation seen
through different output interfaces** — a taxonomy of I/O, not of computation.

Two sharper consequences:

- **JEPA may discard the simulator.** LeCun's case against pixel-prediction is
  that it wastes capacity on unpredictable detail. But those "irrelevant" details
  may *be* the implicit physics — shadow-fall encodes surface geometry, motion
  blur encodes mass. Abstracting them away could be **losing** the simulator, not
  transcending it.
- **The planner may be primary.** Evolution built perception in service of
  action, never the reverse. Renderer and simulator may be **artifacts a planner
  constructs under uncertainty** — not prerequisites. That inverts the hierarchy.

(And: making simulation the privileged tier is also a claim that World Labs is
building the most important thing. Worth holding the hierarchy loosely.)

## 3. Marble as the object casting the shadow

Marble today:

- **Read-path to the eigenvector, at scale** — splats + collision meshes from one
  latent ⇒ the same representation is queried for appearance *and* structure.
- **Writable in the weak sense** — multimodal prompts *sculpt* the latent.
- **Not writable in the sense that matters** — the loop closes through a **human
  navigating** the space, not through an agent whose failed grasps write back
  into the representation.

**Closing the loop** would mean: an agent's failures inside a Marble environment
propagate into the shared latent — spilling the cup in sim changes how the cup is
subsequently rendered *and* simulated. The simulator sculpted by planning
trial-and-error. *The loop is open by architecture; the socket exists; no planner
is plugged in.*

## 4. What pseudo-marble actually tests

We cannot build Marble. We *can* test its central, falsifiable claim on
commodity hardware:

> **Are appearance and physics two projections of one latent — and is that
> coupling learnable and generalizable?**

Concretely:

1. **Author** a principled appearance↔physics coupling (`materials.py`) — because
   in a synthetic substrate the coupling has to be injected, so inject it
   honestly.
2. **Learn** it with a shared-latent dual-decoder (MLX).
3. **Measure coherence** — does perturbing the latent move both outputs
   consistently? — against an independent-models baseline.
4. **Judge by generalization** to held-out `(shape, material)` combinations
   (`splits.py`), not reconstruction. That guards against rediscovering a
   correlation we typed in.

## 5. Honest limitations (carried, not hidden)

- We learn **Blender's eigenvector**, not reality's. Friction/restitution are
  human-authored reference values, not measurements. Marble has the same
  limitation until robots plug in.
- A "shared model beats independent models" result on *seen* pairs is near
  **circular**; only the held-out-combination generalization result is
  meaningful.
- The render path is a **simplified MLX splat decoder**, not 3DGS — lower
  fidelity, chosen for Metal/MLX portability.
- Prior art exists nearby (PhysGaussian, NeRF2Physics, physical-property
  estimation from NeRF/splats). The specific *coherence loss + coherence
  benchmark* framing may be novel, but the novelty claim deserves a literature
  pass before it's leaned on.

## North star

Build the **socket** — the shared-latent coherence between neural rendering and
physics prediction — even before any planner is plugged into it. The eventual,
out-of-scope-for-now move is to let **action-consequences write back** into the
latent: free agents with autonomous sensors operating on a world-model backbone.
That is the bear we are poking, deliberately and at small scale.
