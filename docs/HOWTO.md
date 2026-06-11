# HOWTO — setup & usage

A practical guide for running pseudo-marble on an Apple-silicon Mac (the intended
substrate) or just poking the pure-Python core anywhere.

## 0. What needs what

| You want to... | You need |
|---|---|
| Run the core + tests | Python 3.10+ only |
| Generate paired data (**primary**) | `pip install -e ".[mujoco]"` (arm64-native on Mac) |
| Generate paired data (high-fidelity, optional) | Blender 4.x (ships its own Python) |
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

## 3. Generate a paired dataset

Both generators write the **identical** `sample.json` schema (below), so anything
downstream is agnostic to which one you used.

### 3a. MuJoCo — the primary path (recommended)

```bash
pip install -e ".[mujoco]"
python -m pseudomarble.data.generate_mujoco \
    --output data/pseudo_marble \
    --num-scenes 16 --views 16 --resolution 256 \
    --shapes box,sphere,cylinder,capsule,ellipsoid \
    --render-workers 0 --sim-workers 0     # 0 = phase-specific auto
# or: scripts/run_datagen_mujoco.sh data/pseudo_marble 16 16 256
```

**Parallelism is phase-aware (unified-memory aware).** Scenes are independent, so
generation fans out across *processes* (not threads — a MuJoCo render/sim context
is per-process). But on an Apple-silicon SoC the two stages have **opposite**
optimal widths, so they run as separate phases:

- **render** (`--render-workers`) is GPU/Metal-bound. There is **one** GPU sharing
  the unified-memory bus with the CPU (on the M5 Pro: 20-core GPU, 64 GB @
  307 GB/s), so a worker-per-core just makes processes queue on it and fight for
  bandwidth. Auto keeps this **small**.
- **simulate** (`--sim-workers`) is pure CPU (`mj_step`). The GPU is idle in this
  phase, so it scales ~linearly across the performance cores; auto uses most of
  them, leaving a little headroom.

`--workers N` is a combined fallback that sets both when the per-phase flags are 0.
Manifest order is preserved regardless of finish order. The real optimum per phase
(vs. resolution × views) is worth a quick sweep on your Mac. The GSO path
(`generate_gso --workers`) stays single-phase for now, but its auto default is the
same conservative CPU width (not one-per-core) so it doesn't starve the GPU either.

Why primary: MuJoCo is arm64-native (no Docker, no bpy), and the
appearance↔physics coupling lives in **one geom** (`rgba` + `density` +
`friction`) with physics ground truth as the engine's native output. Honest
caveat: its renderer isn't photorealistic — transparent materials (glass, ice)
look weaker than opaque ones. See `docs/ARCHITECTURE.md`.

> Primitive shapes give a dependency-free smoke test. Concave/custom shapes
> (cup, cone, torus) need a mesh asset — that's the GSO/Objaverse path
> (`docs/GSO_EXPERIMENT.md`).

### 3b. Blender — optional high-fidelity path

Install [Blender 4.x](https://www.blender.org/download/); the pipeline runs in
its bundled Python. Use this when you want photoreal renders (real glass/ice
transmission) at the cost of slower generation and bpy setup.

```bash
blender --background --python src/pseudomarble/data/generate_blender.py -- \
    --output data/pseudo_marble \
    --num-scenes 16 --views 16 --resolution 256 \
    --shapes cube,sphere,cylinder,cone,torus,cup \
    --engine CYCLES --samples 64
```

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
