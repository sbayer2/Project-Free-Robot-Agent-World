# HOWTO — setup & usage

A practical guide for running pseudo-marble on an Apple-silicon Mac (the intended
substrate) or just poking the pure-Python core anywhere.

## 0. What needs what

| You want to... | You need |
|---|---|
| Run the core + tests | Python 3.10+ only |
| Generate paired data | Blender 4.x (ships its own Python) |
| Mesh validation / collision decomposition | `pip install -e ".[datagen]"` (trimesh; optional CoACD/V-HACD) |
| Train the MLX model | Apple silicon + `pip install -e ".[mlx]"` |

The core package is intentionally dependency-light: it imports with **none** of
the above installed, so you can read, test, and reason about it anywhere.

## 1. Clone & core sanity check

```bash
git clone <your-fork-url> pseudo-marble
cd pseudo-marble

python -m pytest                 # if pytest installed
# or run suites directly, no deps required:
python tests/test_materials.py
python tests/test_splits.py
python tests/test_coherence.py
```

Expected: all suites print `N passed`.

## 2. Install (editable)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + pytest/ruff
# optional extras:
pip install -e ".[datagen]"      # trimesh etc. (outside Blender)
pip install -e ".[mlx]"          # Apple silicon only
```

## 3. Generate a paired dataset (Blender)

Install [Blender 4.x](https://www.blender.org/download/). The pipeline runs
inside Blender's bundled Python.

```bash
blender --background --python src/pseudomarble/data/generate_blender.py -- \
    --output data/pseudo_marble \
    --num-scenes 16 \
    --views 16 \
    --resolution 256 \
    --shapes cube,sphere,cylinder,cone,torus,cup \
    --engine CYCLES --samples 64
```

> Tip: the built-in primitive shapes (including a hollow `cup`) give a
> dependency-free smoke test of the whole pipeline before you wire in an asset
> library like Objaverse.

Each scene `data/pseudo_marble/<scene_id>/` contains:

```
renders/view_000.png ... view_NNN.png   # appearance path (+ poses in sample.json)
sample.json                             # the paired record (below)
```

and a top-level `manifest.json` listing every scene, its split, and the held-out
`(shape, material)` combinations.

### The paired sample schema (`sample.json`)

```json
{
  "scene_id": "train_000001",
  "split": "train",
  "input":  { "shape": "cup", "material": "ceramic" },
  "appearance": {
    "resolution": 256,
    "frames": [{ "index": 0, "file": "view_000.png",
                 "position": [..], "look_at": [..] }]
  },
  "physics": {
    "raw":        { "density": 2300, "friction": 0.40, "restitution": 0.30 },
    "normalized": { "density": 0.2875, "friction": 0.266, "restitution": 0.30 },
    "tags": ["brittle", "hard"]
  },
  "dynamics": { "fps": 60, "trajectory": [{ "frame": 1, "position": [..] }] }
}
```

The single `input` drives **both** `appearance` and `physics` — that pairing is
the whole point.

## 4. Faithful collision shapes (optional but recommended)

For concave objects, install a decomposition backend so collision isn't a convex
hull:

```bash
pip install coacd            # preferred
# or provide a testVHACD binary on PATH for trimesh's V-HACD path
```

`pseudomarble.data.collision.compute_collision(mesh, out_dir, method="coacd")`
returns convex pieces that preserve concavity. Without a backend it falls back to
a convex hull **and warns** — see `docs/ARCHITECTURE.md`.

## 5. Train (coming soon)

The MLX encoder + dual decoder and the coherence benchmark harness are the next
build steps (see `docs/ARCHITECTURE.md#build-order`). The interface is fixed in
`pseudomarble.models`; the coherence loss is already implemented and tested.

## 6. Troubleshooting

- **`generate_blender requires Blender's Python (bpy)`** — you ran it in plain
  Python. Use the `blender --background --python ... --` form.
- **`falling back to CONVEX HULL collision` warning** — install `coacd`;
  concavity is being discarded otherwise.
- **`trimesh unavailable`** in a mesh report — `pip install -e ".[datagen]"`.
- **MLX import errors** — MLX is Apple-silicon only; the core and data pipeline
  don't need it.
