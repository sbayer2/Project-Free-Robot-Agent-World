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
2. **Generalization, not reconstruction** — `splits.py` holds out specific
   `(shape, material)` combinations. The result that counts is physics inferred
   for an appearance never seen on that shape.
3. **Honest framing** — the deliverable is *"shared latents capture and
   **generalize** appearance↔physics coupling better than independent models, on
   a controlled synthetic world,"* not *"reality's eigenvector exists."*

## Why Blender (and what it is *not* better at)

- ✅ One tool yields **paired** data: photoreal renders (Cycles/EEVEE) *and*
  structured physics, with aligned camera poses and material assignments via a
  scriptable API (`bpy`).
- ⚠️ **Not** a physics-fidelity advantage over PyBullet/MuJoCo — Blender's
  rigid-body solver *is* the Bullet library. The win is integration, not realism.
- ⚠️ Friction/restitution are **authored**, not measured ⇒ "Blender's
  eigenvector."

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

## Render path on Apple silicon

The reference 3D Gaussian-splatting rasterizer and most of the ecosystem
(gsplat/nerfstudio) are **CUDA-only custom kernels**. On an M5 that's the real
bottleneck — far more than dataset size.

Decision: the pseudo-marble render path is a **simplified, MLX-native splat
decoder** — a fixed budget of Gaussians (`ModelConfig.num_gaussians`) splatted
with a differentiable, MLX-expressible approximation. We accept lower visual
fidelity in exchange for training entirely on Metal/MLX with no CUDA dependency.
This is the project's central "engineer around the constraint" move: a *pseudo*
renderer is fine, because we measure **coherence**, not photorealism.

> If full 3DGS is ever wanted on the Mac, [`brush`](https://github.com/ArthurBrussee/brush)
> (Rust + wgpu/Burn) trains splats cross-platform on Metal and is the fallback
> substrate. Out of scope for the coherence experiment.

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

## Build order

1. **Data pipeline** (renders + physics GT) — *done, runs in Blender.*
2. **MLX dataset loader** — read manifests, batch views + physics targets.
3. **MLX encoder + physics decoder** — easiest path to a first coherence number.
4. **MLX simplified splat render decoder.**
5. **Coherence benchmark harness** — shared vs. independent, on held-out combos.

Starting at (1) is deliberate: the project's validity is won or lost in data
generation, and it's independent of the ML stack.

## Configuration

All knobs live in [`config.py`](../src/pseudomarble/config.py): `RenderConfig`,
`PhysicsConfig`, `DatagenConfig`, `ModelConfig`. Defaults are sized for an M5 /
64 GB budget (modest resolution, view count, Gaussian budget).
