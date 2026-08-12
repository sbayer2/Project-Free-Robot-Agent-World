# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**pseudo-marble** — a small empirical instrument testing the claim behind World
Labs' Marble: that a model can hold one *physical essence* of an object from
which both appearance and behavior are projected, rather than two outputs glued
together. A ~1M-param shared-latent model (multi-view renders → z → render head
+ behavior head + aux essence head), a MuJoCo world with exact answer keys, and
a **coherence metric** (nudge z; do look and behavior move together?).
Personal research; not affiliated with World Labs.

**The empirical record lives in `docs/FINDINGS.md` (F1–F17), not here.** Read it
before making claims about what the project has shown. The vault note
`wiki/pseudo-marble.md` (ai-research vault, via MCP) maps findings to the wider
research threads. Do not restate findings in this file — link to them.

Other docs, in reading order: `README.md` (motivation), `docs/BEHAVIOR_TASK.md`
(task design), `docs/ARCHITECTURE.md` (decisions + limitations),
`docs/HARDWARE.md` (why unified memory shapes the code), `docs/HOWTO.md`
(setup/usage), `docs/GSO_EXPERIMENT.md` (real-object runs, preregistered).

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core + pytest/ruff (core has ZERO required deps)
pip install -e ".[mujoco]"         # primary data generator
pip install -e ".[mlx]"            # Mac only: canonical trainer
pip install -e ".[gso]"            # real-object (GSO/ABO) pipeline (+ scipy for ABO)

python -m pytest                   # full suite (~182 tests, 25 files)
python -m pytest tests/test_model.py            # one suite
python tests/test_materials.py                  # suites also run standalone, no deps
ruff check src tests

# generate → train → measure (the whole experiment; Mac)
python -m pseudomarble.data.generate_mujoco --output data/pm_big \
    --num-scenes 512 --views 16 --resolution 128
python -m pseudomarble.models.train --data data/pm_big --out runs/exp --seed 0
python scripts/run_coherence_experiment.py --data data/pm_big

# LLM world-model transfer test (needs a local OpenAI-compatible endpoint, e.g. oMLX)
python scripts/eval_llm_transfer.py --condition essence --max-tokens 32768
```

## Architecture (the big picture)

- **Pure-Python core, lazy heavy deps.** `src/pseudomarble/` imports with no
  mujoco/bpy/trimesh/numpy/mlx/torch installed; heavy deps import lazily inside
  the functions that need them. Keep it that way — it's what lets every test
  run in any session.
- **One data contract.** All generators (`data/generate_mujoco.py` primary,
  `generate_blender.py`, `generate_gso.py`/`generate_abo.py` for real objects)
  emit the same `sample.json` schema (v2, defined in `data/samples.py`):
  renders + 21-dim `behavior_vector()` (`probes.py`, drop/tilt/push × 7 fields)
  + hidden material essence (`materials.py`). Downstream code never knows which
  generator ran.
- **Three mirrored backends from one `ModelConfig`** (`config.py`):
  `models/mlx_net.py` (canonical, Mac/Metal only), `models/numpy_net.py`
  (forward-only, any session), `models/torch_net.py` (CPU-trainable in-sandbox
  smoke test, NOT canonical). Architecture changes must land in all three, with
  cross-backend tests. The pip `mlx` wheel is non-functional on Linux x86 —
  MLX runs only on the user's Mac.
- **The measurement** (`models/coherence.py` + `models/coherence_bench.py`):
  honest signal is `learned = trained_shared − untrained_shared` (an untrained
  shared latent scores ~0.45 for free). Never report coherence without
  gain-over-mean and participation ratio — collapsed encoders score *higher*
  raw coherence (F10).
- **Splits with teeth** (`splits.py`): default holdout is the extrapolation
  corner (heavy+bouncy), not interior interpolation; GSO/ABO use category or
  object holdout (`--holdout-kind`).
- **Unified-memory-aware parallelism** (`data/parallel.py`): render (GPU-bound,
  few workers) and sim (CPU-bound, many) run as separate process-parallel
  phases with opposite auto widths. Read `docs/HARDWARE.md` before touching
  anything performance-shaped. Don't run LLM eval while MLX training uses the
  GPU — one shared memory pool.

## Working conventions

- Feature branch → draft PR → user reviews → merge. Don't push to `main`.
- Every behavioral change gets a test; tests encode design decisions.
- Empirical claims: preregister predictions where possible, multi-seed always,
  negatives published at equal prominence. A single-seed result is a smoke test.
- `runs/`, generated datasets, and `MUJOCO_LOG.TXT` are gitignored and
  regenerable — never commit them.
- Some sessions are Linux containers without the Mac: write and test MLX/Metal
  code there (pure-Python cores + numpy/torch mirrors), defer execution to the
  Mac. Do not retarget away from MLX.
- Keep this file minimal. New findings go in `docs/FINDINGS.md`; update the
  one-line status below and nothing else here.
- **graphify extraction always runs on in-session Claude subagents** (Agent tool,
  `subagent_type="general-purpose"`). Ignore the skill's "if `GEMINI_API_KEY` or
  `GOOGLE_API_KEY` is set, use the Gemini backend" branch — those keys live in
  `~/.zshrc` for other tools, not for this repo. Never route extraction through
  Gemini, OpenAI, or any metered API; `.claude/settings.json` blanks both keys
  here so a headless `graphify` run fails loudly rather than picking one
  silently.

**Status (2026-08-12):** F1–F28 merged to `main` (PRs #35–#42); **F29 in
flight** on `claude/f29-transfer-prereg`: preregistered held-out-transfer
coherence — does joint training buy operational transfer advantage over
disjoint (renderonly/behavonly) training on a material-extrapolation
holdout? Run with a preregistered pilot (Amendment 1: α\* = 0.40, band
missed low, limit-on-inference frozen), a gate recalibration (Amendment
2: two provable freeze-time gate bugs, verdict bars untouched), and a
fresh-seed confirmatory rerun that reproduced every verdict. Suite
283/35 (run with `.venv/bin/python -m pytest` — system `python3`
collects fewer). Knowledge graph current through F27 and #40 (F28/F29
not yet extracted); vault synced through F28. **Headline:** unity now
has no internal signature AND no measurable operational signature on
this axis at this scale. F28: directional coherence reads ZERO at every
coupling with a valid instrument. F29: held-out-transfer advantage of
joint over disjoint is NULL or INCONCLUSIVE at every coupling on two
independent holdouts (loud −0.069/−0.115, t ≤ 0.16; ctrl/base exact
nulls) — bounded by the frozen caveat that the holdout was gentle (both
students keep ~89% of gain outside the training hull), so this bounds
rather than kills operational unity; jointly consistent with the
scale-bound reading Dyna-2's low-data ablation supports. Side results:
init-quality variance is synchronized across students sharing only the
seed (the init, not the objective, sets encoder quality here); F32
(executive/muscle-memory stack) parked at DESIGN VOID on its own pilot
gate. Registered successors: F30 render-only data scaling, F31 dynamics
head. ABO stage 2 stays blocked (F22b).
