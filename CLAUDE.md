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
- `docs/GSO_EXPERIMENT.md` — real scanned objects to escape "authored coupling";
  **data layer now implemented** (`data/generate_gso.py`): measured-mass parsing,
  mesh MJCF (`build_mjcf(mesh=...)`), category holdout, concave path
  (mesh_validate + collision.py) wired; experiment run still pending real meshes.

## What's built and merged to `main` (all tested)

- `materials.py` — discrete named anchors **and** continuous `MaterialSampler`
  (hidden essence → both physics and appearance, with noise so appearance is
  predictive-but-not-invertible).
- `probes.py` — drop/tilt/push actions + summary outcomes; `behavior_vector()`
  flattens to a stable **21-dim** target (PROBE_ORDER × 7 fields), with normalizers.
- `splits.py` — `RegionHoldout` (held-out region of essence-space): default is
  `EXTRAPOLATION_REGION_HOLDOUT` (heavy+bouncy corner, real extrapolation) vs the
  weak `DEFAULT_REGION_HOLDOUT` (interior interpolation); `--holdout-kind` picks.
  Plus `make_category_holdout` (GSO) and the older discrete `make_combination_split`.
- `data/samples.py` — the generator-agnostic `sample.json` contract, **schema v2**
  (adds `behavior` + `material_truth`).
- `data/generate_mujoco.py` — PRIMARY generator (continuous materials → renders →
  drop/tilt/push). MJCF builder is pure-Python/tested; sim+render guarded by `mujoco`.
- `data/generate_blender.py` — optional high-fidelity appearance generator, same contract.
- `data/dataset.py` — `PseudoMarbleDataset`: manifest/target logic pure-Python;
  image loading + `mlx` conversion lazy.
- `models/coherence.py` — the coherence metric ("do look and behavior move
  together when you nudge the latent?"); reference (finite-diff) + MLX `jvp`.
- `models/mlx_net.py` — trainable encoder + behavior head + aux essence head +
  **render head** (conv decoder z→mean-view image), MLX, runs on the Mac.
  `models/numpy_net.py` — same architecture in numpy, forward-only, any session.
  `models/torch_net.py` — same in PyTorch CPU, trainable in-sandbox.
  `models/losses.py` — pure-Python loss reference (behavior+essence+render).
  `models/train.py` — MLX training loop, eval on held-out essence region.
  `scripts/bench_torch.py` — CPU scale sweep (full model trains at 128px/~1M).
- `models/coherence_bench.py` — the experiment harness: shared-latent vs. two
  independent models, render-vs-behavior coherence. **KEY FINDING:** an untrained
  shared model already scores ~0.45 coherence (architecture, not learning);
  independent baseline ~0. So the honest signal is `learned_coherence =
  trained_shared − untrained_shared` (average the baseline over seeds; it varies
  ~0.36–0.46). Coherence is forward-only, so it runs on numpy in-sandbox.
  Supports `target="essence"` (smooth physics head) to de-risk the topple-chaos
  bias; `compare_both_targets()` reports behavior+essence (high essence + low
  behavior ⇒ chaos artifact, not a true null).
- Render head decision: a **conv decoder** (upsample+conv → image_size, which must
  be render_seed·2^k), reconstructing the **mean view**; NOT a splat decoder —
  we measure coherence, not photorealism. Splat/`brush` is a later option.
- 76 tests across 14 suites; core imports with **no** mujoco/bpy/trimesh/numpy/mlx/torch.

PRs #1–#5 merged to `main`. PR #6 = render head.

### Sandbox note on MLX / backends
The pip `mlx` wheel on plain Linux x86 is **non-functional** (missing
`libmlx.so`) — there is no usable Linux/CPU MLX runtime in these sessions. The
architecture therefore has three mirrored backends from one `ModelConfig`:
- `mlx_net` — canonical trainer, MLX/Metal, Mac only.
- `numpy_net` — forward-only, validates shapes in any session (numpy installs).
- `torch_net` — PyTorch **CPU** stand-in; runnable in-sandbox to verify the
  training loop converges (loss drops) before spending Mac time. NOT canonical.
There is no Claude Code feature to run MLX on Anthropic hardware; the only real
MLX execution is the user's Mac (or an NVIDIA box via MLX's CUDA backend).
`scripts/bench_torch.py` measures in-sandbox CPU scaling (4-core/15GB): comfortable
to ~1M params / 128px / 16 views (sub-second–~1.4s/step); past ~192px / multi-M
params CPU step-time (7–22s) makes real training impractical — memory never bound.
Legacy GPU backends (cudamat/gnumpy/Theano) were ruled out: no GPU here, deprecated.

## Hardware note (2026-06): MacBook Pro exists, just not always on hand

The target substrate **is** the user's **MacBook Pro (Apple silicon, M5 / 64 GB)** —
the MLX / Metal plan is intact and correct. The only limitation is per-session:
some sessions run in a **Linux container without the Mac**, so **Mac-only steps
(MLX training, Metal rendering) can't be executed in those sessions** — only
written and reviewed.

Implication for how we work, not what we build:
- Keep doing what we've done — author MLX/Mac code with **pure-Python cores that
  test in any session**, and **defer Mac-only execution** (running the MLX model,
  Metal/MuJoCo rendering at speed) to when the user has the laptop.
- MLX remains the intended model framework. Do **not** retarget away from it.
- Data generation can also run off-Mac if needed (`MuJoCo` has Linux/CPU wheels,
  slower software GL), but the canonical runs happen on the Mac.

## Next steps (when resumed)

1. On the Mac: generate a small real dataset with MuJoCo and sanity-check that
   probe outcomes are stable (watch the "chaos near tipping points" risk in
   `docs/BEHAVIOR_TASK.md`) and that renders load. (Can be smoke-tested off-Mac.)
2. Encoder + behavior head — **done** (`models/mlx_net.py` + `numpy_net.py` +
   `train.py`). On the Mac: `python -m pseudomarble.models.train --data <dir>` and
   confirm behavior MSE drops on the held-out essence region.
3. Render head — **done** (conv decoder in mlx/numpy/torch; recon loss; trains at
   128px/~1M in-sandbox). On the Mac: generate at `--resolution 128` and train.
4. Coherence harness — **done** (`models/coherence_bench.py`, with the
   architectural-baseline control). RUN THE EXPERIMENT on the Mac: train shared +
   two independent models on real renders; report `learned_coherence` (over
   several untrained seeds) + behavior generalization on held-out essence regions.
   Report honestly, including a null. This is the project's payoff.

## Working conventions

- Pure-Python core stays importable with no heavy deps; heavy deps (`mujoco`,
  image libs, ML framework) are optional extras imported lazily.
- Every behavioral change gets a test; tests encode design decisions.
- Feature branch → draft PR → user reviews → merge to `main`. Don't default to
  pushing straight to `main`.
- Keep claims calibrated and limitations explicit; a negative result is a result.
