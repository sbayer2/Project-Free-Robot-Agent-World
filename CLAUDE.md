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
- `docs/HARDWARE.md` — the target M5 Pro substrate and *why it shapes the code*:
  unified memory's two sides (zero-copy win vs one shared bus/GPU → the data-gen
  phase split), and the ANE-vs-GPU-Neural-Accelerator distinction (MLX runs on the
  GPU, not the ANE). Read before touching anything performance-shaped.
- `docs/PREDICTIVE_CODING.md` — framing note: render-loss as a Rao–Ballard (1999)
  prediction error; coherence control (`learned = trained − untrained`) as the same
  subtract-the-prior residual logic. A connection to neuroscience, not a result.
- `docs/GSO_EXPERIMENT.md` — real scanned objects to escape "authored coupling";
  **data layer implemented** (`data/generate_gso.py`) AND **run design
  preregistered** (§ "Run design": subset gates, category holdout, lr 5e-4,
  essence-weight 0, predictions P1–P5 incl. the ≈0-coherence falsifier);
  execution pending a GSO download + `.[gso]` install (trimesh not in venv).

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
- `data/parallel.py` — generator-agnostic process-parallel scheduler
  (`resolve_workers(default=...)` + `ordered_parallel_map`, order-preserving) +
  phase-aware auto widths. **Unified-memory aware** (M5 Pro: 18-core CPU / 20-core
  GPU / 64 GB @ 307 GB/s, one shared pool): the primary generator runs render and
  sim as SEPARATE phases because they have opposite optima — `default_render_workers`
  (GPU/Metal: small, one shared GPU) vs `default_cpu_workers` (CPU `mj_step`: most
  cores). Flags: `--render-workers` / `--sim-workers` (+ `--workers` combined
  fallback); GSO single-phase but auto = conservative CPU width, not `cpu_count`.
  Processes not threads (MuJoCo context per-process). Pure stdlib, unit-tested
  (`tests/test_parallel.py`); the assemble/serialize seam tested in `test_mujoco_mjcf.py`.
  The model's heavy compute stays MLX/Metal on the GPU's Neural Accelerators.
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
- `tests/batch_probe_stability.py` — probe-label stability study (pure-sim, ~5s, no
  render); the F8 result. `scripts/run_coherence_experiment.py` — the coherence
  experiment runner (multi-seed trained checkpoints + architectural + independent
  disjoint-latent baselines, both targets); the F9 result. Per-head loss weights
  (`ModelConfig.behavior_weight` + `train.py --behavior-weight/--render-weight/
  --essence-weight`) build the render-only / physics-only independent models.
  Soft-topple option (`generate_mujoco --topple-jitter-reps K`) records
  `push.toppled` as smooth P(topple)∈[0,1] (F8 mitigation; default off, no model change).
- `llm_transfer.py` + `scripts/eval_llm_transfer.py` — the LLM world-model
  transfer test (external language world model vs our exact MuJoCo ground truth,
  scored with the behavior-head normalizers; pure-stdlib core, endpoint runner).
- 155 tests across 22 suites; core imports with **no** mujoco/bpy/trimesh/numpy/mlx/torch.

PRs #1–#29 merged to `main` (through 2026-07-12: #25 F11 transfer test, #26 F11
vision condition, #27 F12 basin-selection solved, #28 GSO preregistered run
design, #29 F13 unconditional headline + F14 ABO reality-test void + ABO
loader + lr default 5e-4 — test counts synced to 177 across 24 suites).

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

## Hardware note (2026-06): MacBook Pro M5 Pro — now in hand (per-session caveat below)

The target substrate **is** the user's **MacBook Pro (Apple silicon, M5 Pro:
18-core CPU / 20-core GPU / 64 GB @ 307 GB/s)** — the MLX / Metal plan is intact
and correct. Full topology + why it shapes the code (unified memory's two sides;
MLX runs on the GPU's Neural Accelerators, **not** the ANE) is in
[`docs/HARDWARE.md`](docs/HARDWARE.md). The only limitation is per-session: some
sessions run in a **Linux container without the Mac**, so **Mac-only steps (MLX
training, Metal rendering) can't be executed in those sessions** — only written and
reviewed.

Implication for how we work, not what we build:
- Keep doing what we've done — author MLX/Mac code with **pure-Python cores that
  test in any session**, and **defer Mac-only execution** (running the MLX model,
  Metal/MuJoCo rendering at speed) to when the user has the laptop.
- MLX remains the intended model framework. Do **not** retarget away from it.
- Data generation can also run off-Mac if needed (`MuJoCo` has Linux/CPU wheels,
  slower software GL), but the canonical runs happen on the Mac.

## Status (2026-06-17): the experiment has been RUN on the Mac

The full loop now runs on the Mac (MLX/Metal + MuJoCo): generate → train → measure.
Steps 1–4 below are all **done**; the payoff experiment produced a real result.

1. Real-dataset generation + probe stability — **done.** MuJoCo generation is fast
   on the GPU (256 scenes/128px in ~3s). Stability study (`tests/batch_probe_stability.py`)
   → **F8**: the binary `toppled` is locally chaotic but bounded (~2–4% of scenes)
   and degenerate for box/capsule/sphere; mitigated by the soft-topple option.
2. Encoder + behavior head — **done & trained.** behavior MSE drops on the held-out
   essence region (modestly beats predict-mean); essence does NOT extrapolate to the
   heavy+bouncy corner (worse than mean — the "test with teeth" biting).
3. Render head — **done & trained** (render MSE → ~1e-3 at 128px).
4. Coherence experiment — **done → F9 (the payoff, reported honestly).** On `pm_big`
   (512 scenes, 5 shared seeds + render-only + physics-only): ordering
   independent(~0.04) < architectural(~0.10) < trained(~0.27 avg) holds ON AVERAGE,
   but trained coherence is **seed-unstable** (essence 0.10–0.49) — learned gain
   (+0.16) is WITHIN the cross-seed band. A single-seed pilot overstated it (7.7σ);
   multi-seed corrected it. Prediction quality is stable across seeds; coupling is not.

### Status (2026-07-02): F10 run on the Mac — F9 resolved into two basins
The 20-seed sweep (F9's agreed next step) is DONE, plus its falsification test:
- **F10** (`docs/FINDINGS.md`): mean learned coherence is significant (behavior
  +0.221 t≈5.2; essence +0.165 t≈4.6; 20 trained seeds vs 10 untrained), BUT the
  seed spread is **two basins**: 13/20 escaped (really predict, gain 1.36–1.64×,
  PR 8–84), 7/20 **encoder-collapsed** (exactly predict-mean, participation ratio
  0.0 — z constant across scenes). Collapsed seeds score HIGHER raw coherence
  (0.469 vs 0.279; corr(coherence, held-out MSE)=+0.54) → **never report coherence
  without gain-over-mean + PR**. Retraining all 7 stuck seeds to 150 epochs (runs
  are deterministic per seed): **0/7 escaped** — basin is selected at init, not
  by patience. Honest headline: escaped-only learned coherence ≈ **+0.15** both
  targets (t≈3–4). Artifacts: `runs/big/shared_s0..19`, `runs/big_coherence_20seed/`,
  `runs/big/shared_s*_e150`, `runs/big_coherence_e150/` (gitignored, regenerable).
- **LLM transfer harness** (PR #23, MERGED): prompts a local
  language world model (Qwen-AgentWorld-35B-A3B Q8 MLX, in the user's HF cache;
  text-only — NO AgentWorld artifact ships vision tensors, verified upstream) with
  scene state + probe action, scores JSON predictions with the behavior-head
  normalizers. Core `src/pseudomarble/llm_transfer.py` (pure stdlib), runner
  `scripts/eval_llm_transfer.py` (OpenAI-compatible endpoint, e.g. oMLX on
  127.0.0.1:8000, Bearer auth via `--api-key`/`$OMLX_API_KEY`; response caching;
  `--condition essence|appearance`; needs `--max-tokens 32768` — the model
  reasons in ~17k-token chains at ~64 tok/s).

### Status (2026-07-04): F11 RUN — the transfer test result is in
Both text conditions complete (120/120 responses, zero parse failures), scored
on the 20 held-out extrapolation-corner scenes. **F11** (`docs/FINDINGS.md`):
aggregate 15–25× worse than predict-mean, but the decomposition is the finding —
essence condition beats predict-mean on **8/21 fields** (ballistics: max_height
2–4×, drop settle/path, push final_tilt) with **push.toppled Brier 0.10 vs 0.20**
(calibrated topple on the F8-chaotic label); catastrophic on ramp contact
(tilt.slid_distance 18.7 vs 0.013 — it derives sliding down the 20° ramp,
MuJoCo's high-friction objects stay put). Essence→appearance ablation: hiding
physics numbers shrinks blowups but collapses ballistic wins (8→4 fields) —
the model measurably uses the quantitative inputs. Trained model still wins
aggregate by 15–40×. Artifacts: `runs/llm_transfer_{essence,appearance}/`
(gitignored; responses cached per scene,probe). Don't run eval while MLX
training uses the GPU (one unified-memory pool).

### Status (2026-07-05): F12 RUN — basin selection solved
Collapse is an early-LR overshoot pathology, not init destiny: per-epoch
latent-PR logging (now in train.py, with --behavior-warmup-epochs lever)
showed collapse is the UNIVERSAL early state (every seed PR≈0 in the first
epochs); "escaped" = exits it, driven by BEHAVIOR gradients (warmup delayed/
killed escapes — collapsed healthy s3 outright). Registered predictions were
wrong; the decisive lever: **lr 5e-4 → 20/20 seeds healthy** (7/7 stuck
rescued, none ever collapse, PR floor 6.7; 13/13 healthy confirm, s14 visits
collapse ep0–17 and exits; gains 1.32–1.64 = healthy-basin range) vs 13/20
at default 1e-3. F12 in docs/FINDINGS.md; artifacts runs/basin/ (gitignored);
suite 164. train.py --lr default STAYS 1e-3 until the coherence re-run
validates 5e-4 end-to-end.

### Status (2026-07-11): F13 + F14 RUN — headline number in; reality test void
- **F13**: unconditional coherence over runs/basin/lrlo_s* (all 20 seeds
  healthy): behavior **+0.146 (t≈8.3)**, essence **+0.123 (t≈5.5)** — both
  clear the band, no conditioning; seed spread ±0.065 (was ±0.184). THE
  project headline. `--lr` default FLIPPED to 5e-4 (guarded by test).
- **F14 (ABO reality test)**: GSO canonical source failed its gate (0/1033
  ship mass → docs/GSO_EXPERIMENT.md amendment); switched to ABO (user-
  approved): 437 real products (listed weights, 2.65 OoM), gates tallied
  (deformables, hull-bound density — caught 1 unit error), stability
  spot-check kept binary topple, dataset data/pm_abo (355/82, category
  holdout). Verdict VOID by preregistered gates: P1 pass (20/20 healthy,
  PR 28), **P2 FAIL (held-out gain 0.96, no field >1.02)** → P3
  uninterpretable per the F10 law. Mechanism: assumed contact params ⇒
  probe battery nearly mass-blind; cross-category holdout doesn't
  generalize from ~4 objs/category. Repair menu in F14. New loader
  `data/generate_abo.py` (+ scipy dep); suite 177.

### Status (2026-07-12): F15 + F16 RUN — both loose ends closed as informative nulls
- **F15 (soft topple)**: pm_big_soft regenerated with --topple-jitter-reps 24
  (18/512 scenes fractional = F8's thin shell). The topple FIELD improves
  (gain 3.34→5.52) but coherence does NOT (+0.102±0.081 vs F13 +0.146±0.065,
  t≈1.9 negative-direction) — label noise was never binding. Binary stays
  default.
- **F16 (F14 repair #1 falsified)**: --holdout-kind object lever added to
  generate_gso/abo (tested); same ABO data, unseen objects of SEEN
  categories: gain **0.975** (≈ category split's 0.96) → the probes, not the
  split, are binding. Sole open reality route: mass-sensitive probe family
  (multi-impulse push, log-space labels).
Artifacts runs/big_soft*, runs/abo_obj*, data/pm_big_soft, data/pm_abo_objsplit
(gitignored). Suite 178.

Next, in priority order:
1. **Mass-sensitive probe family** (the F16-mandated F14 repair): multi-impulse
   push battery + log-space displacement labels; re-run ABO object-holdout.
2. **FSQ latent-bottleneck** — measure the essence's size in bits/trits;
   preregistered asymmetric-knee predictions discussed 2026-07-10.
3. GSO Fuel geometry (13GB local) usable for a future no-mass study; parked.

### Status (2026-07-04 evening): F11 vision condition RUN — graft confound resolved
VL36 pipeline complete: downloaded havok2 graft (65 GB), converted mixed_4_6
via mlx_vlm → `~/mlx-models/Qwen-AgentWorld-35B-A3B-VL36-mixed46` (21 GB,
5.01 bpw, 333 vision_tower tensors), vision sanity gate PASSED, then the
20-scene vision condition (3 views/scene as base64 image parts, served by
oMLX — `~/mlx-models` added to its model_dirs): **vision 0.798 ≈ appearance
0.789** (same gain 0.065×, same Brier 0.150, 4/21 wins each; per-field 5
better/6 worse/10 tied). The theoretically correct outcome — renders are
generated FROM appearance params, so a lossless vision tower must reproduce
the appearance condition. Harness `--condition vision` + `--views` on branch
`claude/f11-vision-condition` (suite 161). Full addendum in docs/FINDINGS.md
F11; artifacts `runs/llm_transfer_vision/` (gitignored).

### Housekeeping (pending, safe)
All 17 merged `claude/*` remote branches are safe to delete (verified: every one
is merged into `main`, none unmerged). The user may delete them from the GitHub UI
or any session; use `git fetch --prune` locally afterwards. No work is lost either way.

## Working conventions

- Pure-Python core stays importable with no heavy deps; heavy deps (`mujoco`,
  image libs, ML framework) are optional extras imported lazily.
- Every behavioral change gets a test; tests encode design decisions.
- Feature branch → draft PR → user reviews → merge to `main`. Don't default to
  pushing straight to `main`.
- Keep claims calibrated and limitations explicit; a negative result is a result.
