# CLAUDE.md — project memory for pseudo-marble

Durable context so any future session can pick up without re-reading the whole
history. Keep this current when state changes.

## What this project is

**pseudo-marble** — a small, runnable research instrument testing the core idea
behind World Labs' Marble: that a model can hold the *physical essence* of an
object (what it will **do** when acted upon — the inverse of rendering) from
which both **appearance** and **behavior** are projected, rather than being two
separate outputs glued together.

Plain framing (see `README.md`): rendering goes idea→picture; the hard inverse is
picture/words→physical understanding. Humans have it from a lifetime of embodied
exploration; machines don't. We build the smallest system that *could* contain
that shared understanding, plus a **measurement** (`models/coherence.py`) that
tells us whether it actually does — and we're willing to find that it doesn't.

Personal research. Not affiliated with World Labs. Not an attempt to copy Marble.

## Key docs (read in this order)

- `README.md` — plain-language motivation + status.
- `docs/TAXONOMY_NOTES.md` — Li's taxonomy; what Marble is and isn't (it's likely
  "two outputs in a package," not a unified eigenvector — that gap IS our target).
- `docs/BEHAVIOR_TASK.md` — the current (v2) task design.
- `docs/ARCHITECTURE.md` — design decisions, honest limitations, build order.
- `docs/GSO_EXPERIMENT.md` — parked: real scanned objects to escape "authored coupling."

## What's built and merged to `main` (all tested)

- `materials.py` — discrete named anchors **and** continuous `MaterialSampler`
  (hidden essence → both physics and appearance, with noise so appearance is
  predictive-but-not-invertible).
- `probes.py` — drop/tilt/push actions + summary outcomes; `behavior_vector()`
  flattens to a stable **21-dim** target (PROBE_ORDER × 7 fields), with normalizers.
- `splits.py` — `RegionHoldout` (held-out region of essence-space) + the older
  discrete `make_combination_split`.
- `data/samples.py` — the generator-agnostic `sample.json` contract, **schema v2**
  (adds `behavior` + `material_truth`).
- `data/generate_mujoco.py` — PRIMARY generator (continuous materials → renders →
  drop/tilt/push). MJCF builder is pure-Python/tested; sim+render guarded by `mujoco`.
- `data/generate_blender.py` — optional high-fidelity appearance generator, same contract.
- `data/dataset.py` — `PseudoMarbleDataset`: manifest/target logic pure-Python;
  image loading + `mlx` conversion lazy.
- `models/coherence.py` — the coherence metric ("do look and behavior move
  together when you nudge the latent?"); reference (finite-diff) + MLX `jvp`.
- 56 tests across 9 suites; core imports with **no** mujoco/bpy/trimesh/numpy/mlx.

PRs #1, #2, #3 all merged to `main`.

## IMPORTANT constraint change (2026-06): no MacBook / no Apple silicon

The project was scoped around an **M5 / MLX / Metal** substrate, but the user
**does not have access to a Mac.** This blocks two assumptions:

- **MLX runs only on Apple silicon** — the planned "MLX encoder + behavior head"
  cannot run on the user's hardware as-is. The model layer should likely be made
  **framework-agnostic** or retargeted (PyTorch/JAX on CPU/CUDA, or numpy for a
  tiny first model) before any training step.
- The "runs on a MacBook Pro" language in README/docs is now aspirational, not
  literal. Revisit it when the compute target is known.

Good news — most of the stack is portable:
- The data layer is **pure-Python**; `MuJoCo` has **Linux/CPU wheels**, so data
  generation can run off-Mac (even in a Linux container), though renders are
  software-GL and slower.
- The coherence metric has a pure-Python reference path (no MLX needed to test it).

**Open decision for next session:** confirm the actual compute the user has
(Linux? CUDA GPU? CPU only? cloud?), then choose the model framework accordingly.
Do NOT default to MLX again without confirming Apple-silicon access.

## Next steps (when resumed)

1. Decide compute target + model framework (see constraint above).
2. Generate a small real dataset on whatever box can run MuJoCo, and sanity-check
   that probe outcomes are stable (watch the "chaos near tipping points" risk in
   `docs/BEHAVIOR_TASK.md`) and that renders load.
3. Build encoder + action-conditioned behavior head (+ optional render head).
4. Run the headline experiment: **shared-essence model vs. two glued-together
   models**, compared on coherence over **held-out essence regions** — and report
   the result honestly, including a null.

## Working conventions

- Pure-Python core stays importable with no heavy deps; heavy deps (`mujoco`,
  image libs, ML framework) are optional extras imported lazily.
- Every behavioral change gets a test; tests encode design decisions.
- Feature branch → draft PR → user reviews → merge to `main`. Don't default to
  pushing straight to `main`.
- Keep claims calibrated and limitations explicit; a negative result is a result.
