# Experiment: escaping "Blender's eigenvector" with real scans

> Status: **data layer IMPLEMENTED** (`data/generate_gso.py`); the experiment
> itself is still pending — it needs a real GSO download + a machine with
> `trimesh`/`mujoco`. Build: `pip install -e ".[gso]"`, then
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
