# pseudo-marble

> A constrained, MLX/Metal-native instrument for testing the **core idea** behind
> World Labs' [Marble](https://www.worldlabs.ai/) — *one latent, two coupled
> output paths (appearance + physics)* — on commodity Apple-silicon hardware,
> fed by data everyday coders can actually generate (Blender).

**This is personal research.** It is **not** Marble, is **not affiliated with
World Labs**, and makes **no claim** to reproduce their system. It is a *thinking
instrument*: a deliberately small "pseudo-marble" we can run, perturb, and reason
about end-to-end — to develop intuition about world models on a non-CUDA,
non-H100 substrate.

---

## The idea in one paragraph

Fei-Fei Li's [*A Functional Taxonomy of World
Models*](https://www.worldlabs.ai/blog/taxonomy-of-world-models) sorts world
models into **renderers** (output pixels), **simulators** (output state), and
**planners** (output actions), and closes by gesturing at a *unified* model that
switches between them. We read that essay as a **shadow** of Marble: the
interesting claim is that appearance and physics are **two projections of one
underlying latent** — an "eigenvector" of the object. Marble already hints at
this by emitting Gaussian splats *and* collision meshes from a single model.

**pseudo-marble tests whether that shared-latent coupling is real and
learnable** on hardware we own — by building the smallest model that emits *both*
a neural render representation and a physics description **from the same latent**,
and measuring how coupled that latent actually is.

> Full conceptual lineage (the five-part argument, the planner-primacy inversion,
> the "open socket" reading of Marble) lives in
> [`docs/TAXONOMY_NOTES.md`](docs/TAXONOMY_NOTES.md).

## What we actually measure

Not visual quality. Not physics accuracy in isolation. **Latent coherence across
modalities**: if you perturb the shared latent, do appearance *and* physics move
*consistently*? A genuinely shared representation says yes; two independently
trained models say no. The headline experiment is:

> **shared-latent dual-decoder** vs. **two independent single-task models**,
> compared on a [coherence metric](src/pseudomarble/models/coherence.py),
> over **held-out material × shape combinations**.

## The honesty problem (read this before believing any result)

Blender **decouples** how a thing looks from how it behaves. So the
appearance↔physics coupling the model could learn is coupling **we inject** via a
[principled material library](src/pseudomarble/materials.py). That makes the
naive "shared model wins" result nearly circular.

The defensible question is therefore **generalization**, baked into the data
split from day one: hold out specific `(shape, material)` *combinations* and test
whether the model infers physics for an appearance it never saw paired with that
shape. If it interpolates the coupling → a real result. If it only memorizes seen
pairs → the claim fails, and that failure is worth knowing. See
[`splits.py`](src/pseudomarble/splits.py) and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#the-honesty-problem).

We are also, unavoidably, learning **Blender's eigenvector** — a decent
approximation of reality's, not reality's itself. That's the same limitation
Marble has until real robots plug in.

## Architecture (where we're headed)

```
scene description (shape id + material)            <- the single shared input
        │
     encoder  (MLX)
        │
        ▼
        z   ← one latent
       ╱ ╲
      ╱   ╲
render_decoder        physics_decoder              <- two projections, shared weights upstream
(simplified MLX        (density, friction,
 gaussian splats)       restitution)

loss = render + physics + coherence_weight · coherence(z)
```

The **coherence loss** is the novel, still-untested contribution. The render path
is a *simplified MLX-native* splat decoder — we trade 3DGS fidelity for the
ability to train on Metal/MLX without CUDA kernels (see
[ARCHITECTURE.md](docs/ARCHITECTURE.md#render-path-on-apple-silicon)).

## Status

| Component | State |
|---|---|
| Principled material library (the coupling) | ✅ implemented + tested |
| Held-out combination splits (the honest test) | ✅ implemented + tested |
| Coherence metric (reference + MLX fast path) | ✅ definition implemented + tested |
| Blender data pipeline (renders + physics GT) | ✅ written; runs on your Mac in Blender |
| Mesh validation / convex decomposition | ✅ written (optional deps) |
| MLX encoder + dual decoder | 🔜 design stage ([ARCHITECTURE.md](docs/ARCHITECTURE.md)) |
| Coherence benchmark harness | 🔜 next |

## Quick start

```bash
# pure-python core + tests (no Blender/MLX needed)
python -m pytest            # or: python tests/test_materials.py

# generate a tiny paired dataset (requires Blender on your Mac)
blender --background --python src/pseudomarble/data/generate_blender.py -- \
    --output data/pseudo_marble --num-scenes 16 --views 16 --resolution 256
```

Full setup, including Apple-silicon/MLX, is in
[`docs/HOWTO.md`](docs/HOWTO.md).

## Repository layout

```
src/pseudomarble/
  materials.py            # the authored appearance↔physics coupling
  splits.py               # held-out material×shape combinations (generalization test)
  config.py               # render / physics / model configs
  data/
    generate_blender.py   # bpy pipeline: paired renders + physics ground truth
    mesh_validate.py      # watertightness gate (mass needs valid volume)
    collision.py          # convex DEcomposition (not convex hull — keeps concavity)
  models/
    coherence.py          # the coherence loss/metric (core idea)
docs/
  TAXONOMY_NOTES.md       # the conceptual lineage / argument
  ARCHITECTURE.md         # design decisions + honest limitations
  HOWTO.md                # setup & usage
tests/                    # pure-python suites (run anywhere)
```

## License

[MIT](LICENSE) © 2026 sbayer2. Independent research project; not affiliated with
or endorsed by World Labs.
