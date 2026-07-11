# Experiment: escaping "Blender's eigenvector" with real scans

> Status: **data layer IMPLEMENTED** (`data/generate_gso.py`); **run design
> PREREGISTERED** (§ "Run design", below) — execution needs a real GSO download
> + `pip install -e ".[gso]"` (adds `trimesh`; not yet in the Mac venv). Then:
> `python -m pseudomarble.data.generate_gso --gso-root <dir> --output <dir>`.
> The pure-Python core (object/mass/category parsing, mesh MJCF, category split,
> the convex_hull branch of the concave path) is unit-tested; mesh loading +
> rendering + simulation are guarded.

## What is built

- **`generate_gso.py`** — discovers scanned-object folders, reads **measured mass**
  (from `model.sdf` `<inertial>`, with JSON/pbtxt fallbacks), renders multi-view
  appearance from the real texture, runs drop/tilt/push on the mesh, and writes the
  **same `sample.json` contract**.
- **`build_mjcf(mesh=MeshAsset(...))`** — the mesh path (extends the primitive
  builder): explicit measured mass, optional texture material, and multi-part
  convex collision.
- **Concave path wired** — `mesh_validate` gates watertightness; `collision.py`
  (CoACD/V-HACD) decomposes the mesh so a real cavity (mug, bowl) survives instead
  of collapsing to a convex hull. Provenance recorded per scene.
- **`splits.make_category_holdout`** — holds out whole object **categories**
  (materials are baked per object, so synthetic combination/region splits don't
  transfer); `make_object_holdout` is the fallback when categories are unknown.
- **Thinner, honest physics** — `samples.physics_labels_measured`: mass measured,
  density derived, **friction/restitution assumed** and flagged in
  `physics.provenance`.

## The limitation it addresses

Both our generators (MuJoCo primary, Blender optional) learn an appearance↔physics
coupling that **we authored** in `materials.py`. We defend that with a principled
material library and a generalization split — but it remains *Blender's* (or
MuJoCo's) eigenvector: friction and restitution are human-chosen reference
numbers, not measurements of reality.

## The idea

**[Google Scanned Objects (GSO)](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/)**
is a dataset of ~1000 real household objects, 3D-scanned, that ships with
**real measured mass** (and real geometry / textures). Using it means:

- **Zero generation work** — the meshes, textures, and mass labels already exist.
- **Reality's coupling, not ours** — the appearance↔mass relationship is the one
  that actually holds in the world, because both came from the same physical
  object. This is the closest we can get, on commodity hardware, to the coupling
  Marble learns from real data.

So a coherence result on GSO would be a *much* stronger claim than the same
result on synthetic data: the model would be recovering a coupling no one typed
in.

## Why it's not the default (the honest trade-offs)

1. **Loss of free recombination.** Our generalization test depends on holding out
   `(shape, material)` *combinations* — which requires freely pairing any shape
   with any material. GSO objects have fixed, baked-in materials; you can't make a
   "ceramic version of this wooden bowl." The held-out split has to be
   reconceived (e.g. hold out object *categories*, or cluster by visual/physical
   nearest-neighbors), which is its own design problem.
2. **Sparser physics labels.** GSO gives **mass** (real, valuable) but not clean
   per-object friction / restitution. The physics target shrinks toward
   "mass + geometry," at least at first.
3. **Real-mesh messiness.** Scanned meshes need the same watertightness gating
   (`mesh_validate.py`) and concave collision (`collision.py`) we already built —
   those carry over directly.

## How it would plug in

The `sample.json` contract (`data/samples.py`) is generator-agnostic, so a GSO
loader would be `data/generate_gso.py` emitting the *same* records:

- `input`: object id (+ a category tag standing in for "shape").
- `appearance`: multi-view renders of the scanned mesh (reuse MuJoCo's renderer
  with a mesh asset, or Open3D).
- `physics`: `raw.density`/`mass` from the GSO metadata; friction/restitution
  left null or estimated, and flagged as `measured: true` vs the synthetic
  `measured: false`.
- A new split strategy in `splits.py` for non-recombinable materials.

## When to reach for it

After the synthetic MuJoCo experiment gives a first coherence number and the
benchmark harness exists. GSO then becomes the "does this survive contact with
reality?" follow-up — the strongest version of the result, and the honest answer
to "but you just learned Blender's eigenvector."

**That condition is now met** (F13-candidate, run 2026-07-10: unconditional
learned coherence +0.146 behavior (t≈8.3) / +0.123 essence (t≈5.5) at lr 5e-4,
all 20 seeds healthy, report at `runs/basin_coherence_lrlo/coherence_report.json`;
FINDINGS entry pending). The run design below was written and preregistered
*before* any GSO data was generated.

---

## Run design (preregistered 2026-07-11)

### Question

Does the shared-latent model recover an appearance↔behavior coupling that
**nobody authored** — reality's mass↔look relationship, baked into scanned
objects — or was the synthetic +0.146 the generator's eigenvector only?

### What changes vs the synthetic run (and what deliberately doesn't)

| axis | synthetic (`pm_big`) | GSO run | why |
|---|---|---|---|
| objects | 5 primitives × sampled materials | ~200–400 real scans | the point |
| coupling | authored (`materials.py`) | reality's (mass ↔ look) | the point |
| holdout | essence-region (extrapolation corner) | **category holdout** (`make_category_holdout`) | materials are baked per object; recombination splits don't transfer |
| physics labels | density/friction/restitution, all true | **mass measured; friction/restitution ASSUMED** (flagged in `physics.provenance`) | GSO ships mass only |
| essence-head weight | 0.3 | **0.0** | 2 of 3 essence channels would supervise assumed constants — noise dressed as signal. Density-only aux supervision is a follow-up lever, not the default |
| lr | 1e-3 (historical) | **5e-4** | F12: 1e-3 collapses ~1/3 of seeds; 5e-4 → 20/20 healthy |
| everything else | — | **identical** | 128px, 8 views, drop/tilt/push, 21-dim behavior target, `OUTCOME_NORMALIZERS`, 20 seeds, 10 untrained inits, render-only/physics-only controls retrained on GSO data |

### Pipeline (stages; all machinery exists)

1. **Setup**: `pip install -e ".[gso]"` (adds `trimesh`); download GSO (~1k
   objects; Gazebo Fuel `GoogleResearch` collection or a HF mirror; budget
   10–20 GB disk).
2. **Subset selection** (scripted, criteria fixed here): objects with (a) a
   parseable measured mass, (b) a mesh that passes `mesh_validate` or
   decomposes cleanly via CoACD, (c) category label present. Take ALL such
   objects up to ~400, no hand-picking beyond these gates; record the
   rejection tally (no silent drops).
3. **Label-stability spot-check (F8 analog, ~minutes, pure sim):** run the
   probe battery 3× with 3% action jitter on ~50 meshes. Real shapes may make
   `toppled` degenerate (flat-bottomed boxes never topple) or chaotic. Decide
   from the data, *before training*: keep binary topple / switch to
   `--topple-jitter-reps 16` soft labels / exclude the field from headline
   metrics. Whichever branch, it applies to ALL conditions equally.
4. **Generate** (`generate_gso.py`, phase-aware parallel widths): renders from
   the real textures + drop/tilt/push outcomes → same `sample.json` contract.
   Hold out 3–5 whole categories (~15–20% of scenes) as test.
5. **Train**: 20 seeds × 50 epochs, lr 5e-4, essence-weight 0, per-epoch
   latent-PR logged. Plus render-only and physics-only models (independent
   control), and 10 untrained inits (architectural baseline) — all on GSO data.
6. **Measure**: `run_coherence_experiment.py` on the held-out categories —
   unconditional statistics over all 20 seeds, reported (per F10 law) as
   coherence + gain-over-mean + PR together. Behavior target is primary;
   essence-target coherence is reported but expected uninformative (see P4).

### Preregistered predictions (written before any GSO data existed)

- **P1 (training health):** ≥19/20 seeds train healthy at lr 5e-4 (PR floor
  > 2, no collapse plateau). F12's mechanism is an optimizer property, not a
  dataset property.
- **P2 (prediction):** held-out behavior gain-over-mean lands in **1.10–1.40×**
  — positive (mass/geometry drive drop outcomes strongly) but below the
  synthetic 1.41× (reality's coupling is noisier: hollow-but-big objects,
  baked textures, assumed friction flattening the tilt/push signal).
- **P3 (the headline):** unconditional learned coherence (behavior target) is
  **positive but smaller than synthetic**: point expectation ≈ +0.05 to +0.12,
  vs +0.146 synthetic. **Falsifier:** ≈ 0 (within the untrained band) ⇒ the
  synthetic result does not survive contact with reality ⇒ FINDINGS records
  "we measured the generator's eigenvector, not reality's" — a negative
  result we commit to publishing with the same prominence.
- **P4 (essence target):** essence-side coherence is weak/uninterpretable —
  with friction/restitution assumed constants, the essence head has one real
  channel. Not a failure; a designed limitation, stated in advance.
- **P5 (the confound we expect to fight):** category-recognition shortcut.
  The encoder may predict behavior by *recognizing the object category*
  (mugs behave like mugs) rather than reading material properties. Category
  holdout is the defense; residual risk is visually-similar categories
  spanning train/test. Post-hoc check: correlate per-scene behavior error
  with visual nearest-neighbor distance to the training set — a strong
  correlation flags the shortcut.

### Cost estimate (M5 Pro)

Generation ~400 scenes ≈ 15–40 min (mesh sim slower than primitives; render
phase GPU-bound as usual). Training 20 seeds + 2 controls ≈ 40–60 min
(dataset ~2× pm_big). Coherence ≈ 10 min. **One evening, end to end.**

### Open decisions (need a human call before execution)

1. GSO download source (Fuel scrape vs HF mirror) — whichever is practical.
2. Subset ceiling (400 default; 200 if generation is slower than estimated).
3. Step-3 outcome adjudication if topple is BOTH degenerate for most shapes
   AND chaotic for the rest (likely resolution: soft labels + report
   with-and-without, mirroring F8's guidance).
