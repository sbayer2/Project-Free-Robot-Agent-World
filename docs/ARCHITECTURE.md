# Architecture & design decisions

This document records *why* the code is shaped the way it is, including the
uncomfortable parts. Read [`TAXONOMY_NOTES.md`](TAXONOMY_NOTES.md) first for the
motivation.

## The object we're building

```
scene description (shape id + material name)         single shared input
        │
   ┌────┴─────┐
   │ encoder  │  (MLX MLP / small transformer over scene + view conditioning)
   └────┬─────┘
        ▼
        z   ∈ ℝ^latent_dim          the one latent ("eigenvector" candidate)
       ╱ ╲
      ╱   ╲
┌────────────┐     ┌──────────────┐
│  render    │     │   physics    │
│  decoder   │     │   decoder    │
│ (MLX splat)│     │  (MLP head)  │
└─────┬──────┘     └──────┬───────┘
      ▼                   ▼
 N RGB views        (density, friction, restitution)

loss = L_render + λ_phys · L_physics + λ_coh · L_coherence(z, decoders)
```

The hypothesis under test: forcing both decoders to read from a *single* `z`,
plus an explicit **coherence** term, produces a latent that carries appearance
and physics *jointly* — and that this generalizes to unseen `(shape, material)`
combinations better than two independent single-task models.

## The honesty problem

Blender's **visual** material (Principled BSDF) and its **physics** material
(rigid-body friction / restitution / mass) are independent. Nothing makes a
ceramic-looking shader behave like ceramic. So:

- Any appearance↔physics coupling the model can learn is coupling **we inject**.
- Therefore a shared model "beating" independent models on **seen** pairs proves
  almost nothing — it rediscovers a hand-authored correlation.

Mitigations, all already reflected in code:

1. **Principled injection** — `materials.py` binds appearance and physics through
   real material identities (glass: transmissive *and* dense/brittle; rubber:
   matte-dark *and* grippy/bouncy). The coupling is physically motivated, not
   arbitrary.
2. **Generalization, not reconstruction** — with continuous materials,
   `splits.RegionHoldout` reserves a *region* of essence-space for test. The
   default is now **extrapolation** (`EXTRAPOLATION_REGION_HOLDOUT`: the heavy+bouncy
   corner, where no training object lies jointly beyond the held-out points, so
   the coupling must extrapolate outside the training hull) — `interpolation` (an
   interior box) is available but is a weak test (an audit finding). The discrete
   `make_combination_split` remains for the named/primitive path. The result that
   counts is behavior inferred for an essence never seen in training.
3. **Honest framing** — the deliverable is *"shared latents capture and
   **generalize** appearance↔physics coupling better than independent models, on
   a controlled synthetic world,"* not *"reality's eigenvector exists."*

## Generators: MuJoCo primary, Blender optional, one contract

Every generator writes the **same** `sample.json` (`data/samples.py`) — the
single source of truth for the schema. Choosing a generator is choosing code
*behind* that contract; the loader, splits, and coherence benchmark never change.

**Primary — MuJoCo** (`data/generate_mujoco.py`):
- ✅ arm64-native on macOS (no Docker, no bpy quirks); least setup work.
- ✅ The coupling lives in **one geom**: `rgba` (appearance) + `density`
  (→ auto mass) + `friction` (physics) — literally our `Material` dataclass in
  MJCF. Physics ground truth (mass, contacts, trajectory) is the engine's
  *native* output, so there's no separate bake step.
- ⚠️ Renderer is **not** photorealistic — no real glass transmission/ice, so the
  appearance→physics cue is weaker for *transparent* materials specifically.
  Acceptable because we measure coherence, not photorealism.
- ⚠️ No first-class restitution scalar (bounce emerges from solref/solimp); the
  restitution *label* always comes from `materials.py`, and we only approximate
  bounce in the trajectory.

**Optional — Blender** (`data/generate_blender.py`):
- ✅ Photoreal renders (Cycles/EEVEE) with true transmission — use when the
  transparent-material visual cue matters.
- ⚠️ **Not** a physics-fidelity advantage — Blender's rigid-body solver *is* the
  Bullet library. The win is render fidelity, not realism of physics.
- ⚠️ Slower, and depends on the finicky `bpy` API.

**Common caveat:** both author friction/restitution rather than measuring them ⇒
we learn the *generator's* eigenvector. Escaping that means real measured data —
see [`GSO_EXPERIMENT.md`](GSO_EXPERIMENT.md).

## Collision: decomposition, not convex hull

The canonical object here is a **cup**. A convex hull of a cup is a solid lump —
no cavity, no rim. Convex-hull collision would discard exactly the concavity that
makes a container a container. So `collision.py` defaults to **approximate convex
decomposition** (CoACD / V-HACD) and only falls back to a single hull when no
backend is installed — **with a loud warning**, because that fallback silently
changes what the physics means.

## Mass needs a watertight mesh

`mass = density · volume` is only defined for a watertight mesh. Objaverse is full
of non-manifold/open meshes where volume is undefined. `mesh_validate.py` gates on
watertightness before a mesh contributes physics ground truth. (Hollow/composite
objects still violate the uniform-density assumption — a known, accepted limit.)

## Render head (the appearance projection)

The reference 3D Gaussian-splatting rasterizer and most of the ecosystem
(gsplat/nerfstudio) are **CUDA-only custom kernels** — a poor fit for Metal/MLX
and for a CPU-testable design.

Decision (implemented): the render head is a **lightweight conv decoder** —
`z -> Linear -> seed feature map (render_seed²) -> (nearest-upsample 2x + Conv2d
+ ReLU) × k -> Conv2d -> sigmoid`, producing an `image_size × image_size × 3`
image. The reconstruction target is the **mean over the input views** (the
pose-averaged "canonical appearance"), consistent with a pose-invariant latent.
`image_size` must be `render_seed · 2^k` (32/64/128/256…).

Why a conv decoder, not a Gaussian-splat decoder: we measure **coherence**
(do appearance and behavior move together when the latent is nudged?), **not
photorealism**. A conv decoder is simpler, mirrors cleanly across all three
backends (MLX/numpy/torch), runs on CPU for in-sandbox tests, and is sufficient
to carry the appearance↔physics coupling (colour/material survive view-averaging).
A splat decoder remains a possible higher-fidelity swap later; `brush`
(Rust + wgpu/Burn) is the route if full 3DGS on Metal is ever wanted.
(`ModelConfig.num_gaussians` is retained only as a reserved knob for that path.)

## The coherence metric

Defined and implemented in [`models/coherence.py`](../src/pseudomarble/models/coherence.py).
Intuition: if appearance and physics are projections of one latent, a latent
**direction** that strongly changes appearance should also change physics, and a
direction that barely touches appearance should barely touch physics. We measure
the correlation of the two decoders' sensitivity magnitudes (Jacobian-vector
product norms) over many sampled directions:

```
coherence = corr_d( ‖∂ render / ∂d‖ , ‖∂ physics / ∂d‖ )
loss      = 1 − coherence
```

- **Reference impl** (finite differences, pure Python) makes the *definition*
  executable and testable.
- **MLX impl** uses exact `jvp` for training-time speed.
- **Direction sampling must be iid Gaussian, not unit-normalized** — unit
  directions induce a simplex artifact that makes *disjoint* latent subspaces
  anti-correlated, so an independent baseline would score negative instead of ~0.
  `sample_directions()` does this correctly. (This bug was caught by a test; the
  fix is recorded here so it isn't reintroduced.)

Unit-test sanity checks: coupled decoders (shared latent dims) score high;
independent decoders (disjoint dims) score ~0.

**Smooth vs. chaotic target (de-risking a false null).** The physics-side decoder
can be the **behavior** head (drop/tilt/push outcomes — the headline) or the
**essence** head (density/friction/restitution). `toppled` is a *threshold*, so
near a tipping point the behavior head's sensitivity is erratic — its Jacobian
magnitudes are noisy, which can drag coherence DOWN artificially (a false null).
The essence head is a *smooth* function of `z`, so coherence on it is robust to
that chaos. `coherence_bench.compare(..., target="essence")` /
`compare_both_targets(...)` measure both; the recommended read is **a high
essence-coherence beside a low behavior-coherence ⇒ the topple-chaos artifact, not
a real absence of coupling.** (`*_from_z` accessors exist on all three backends.)

## Build order

1. **Data pipeline** — *done.* Continuous materials + drop/tilt/push behavior
   outcomes (the v2 task — see [`BEHAVIOR_TASK.md`](BEHAVIOR_TASK.md)), MuJoCo
   primary + Blender optional, behind the shared `sample.json` contract.
2. **Dataset loader** — *done.* Reads manifests + sample.json, assembles the
   flattened drop/tilt/push behavior target (`probes.behavior_vector`), serves
   batches; image loading + MLX conversion are lazy
   (`data/dataset.PseudoMarbleDataset`).
3. **Encoder + behavior head** — *done.* Multi-view CNN -> shared latent ->
   behavior head (drop/tilt/push outcomes) + auxiliary essence head. One
   architecture, three mirrored backends from one `ModelConfig`:
   `models/mlx_net.py` (canonical trainer, MLX/Metal on the Mac, via
   `models/train.py`); `models/numpy_net.py` (forward-only, shape-tested in any
   session, since MLX has no Linux runtime); `models/torch_net.py` (PyTorch CPU
   stand-in that verifies the training loop converges in-sandbox — a 4-5x loss
   drop on a synthetic overfit batch is checked by a test). Eval is on the
   held-out essence region.
4. **Render head** — *done.* A lightweight conv decoder (z -> mean-view image),
   mirrored across mlx/numpy/torch; reconstruction MSE added to the loss. At
   128px/~1M params the full model trains in-sandbox (torch CPU, ~0.85 s/step).
5. **Coherence benchmark harness** — *done* (`models/coherence_bench.py`).
   Shared-latent model vs. two independent models, render-vs-behavior coherence
   over held-out essence regions. **Key finding from building it:** an *untrained*
   shared model already scores high coherence (~0.45) purely from architecture
   (both heads read one `z`); the independent baseline is ~0. So the shared-vs-
   independent gap is NOT evidence of a learned eigenvector — the honest signal is
   `learned_coherence = trained_shared − untrained_shared` (averaged over several
   untrained seeds, since the baseline varies ~0.36–0.46). The apparatus caught
   this before any Mac time was spent.
6. **Run the experiment on the Mac** — train the shared + two independent models
   on real renders, report `learned_coherence` + behavior generalization on
   held-out essence regions. Honestly, including a null.
6. **(Parked)** GSO real-scan experiment — reality's eigenvector
   ([`GSO_EXPERIMENT.md`](GSO_EXPERIMENT.md)).

Starting at (1) was deliberate: the project's validity is won or lost in data
generation, and it's independent of the ML stack.

## Configuration

All knobs live in [`config.py`](../src/pseudomarble/config.py): `RenderConfig`,
`PhysicsConfig`, `DatagenConfig`, `ModelConfig`. Defaults are sized for an M5 /
64 GB budget (modest resolution, view count, Gaussian budget).
