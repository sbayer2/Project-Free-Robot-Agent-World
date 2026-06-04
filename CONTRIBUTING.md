# Contributing & conventions

pseudo-marble is a personal research project, kept open and tidy so the ideas are
auditable. These are the conventions for the repo.

## Principles

1. **The core stays dependency-light.** `pseudomarble.materials`, `.splits`,
   `.config` and the coherence *definition* must import with **no** `bpy`,
   `trimesh`, `numpy`, or `mlx` installed. Heavy deps are optional extras
   (`[datagen]`, `[mlx]`) and are imported lazily *inside* the functions that
   need them.
2. **Honesty over polish.** If a result is near-circular, or we're learning
   "Blender's eigenvector," say so in the code and docs. The known-limitations
   sections are load-bearing, not boilerplate.
3. **Tests encode the design.** When a test catches a conceptual bug (e.g. the
   coherence direction-sampling artifact), record the fix and the reason in
   `docs/ARCHITECTURE.md` so it isn't reintroduced.

## Project layout

```
src/pseudomarble/    importable library (src-layout)
  data/              samples.py (shared sample.json contract) +
                     generate_mujoco.py (primary) / generate_blender.py (optional) +
                     mesh/collision helpers
  models/            MLX model + coherence loss
tests/               pure-python suites, runnable with plain `python`
docs/                TAXONOMY_NOTES (why) · ARCHITECTURE (how) · HOWTO (use)
```

## Code style

- Format/lint with **ruff** (`pyproject.toml` config): `ruff check . && ruff format .`
- Line length 100. Target Python 3.10+.
- Type-hint public functions. Dataclasses for structured config/records.
- Docstrings explain *why*, not just *what* — this is a research codebase.

## Tests

- Pure-Python suites must run with no third-party deps:
  `python tests/test_*.py` (each file self-runs) **or** `python -m pytest`.
- Anything requiring `bpy` / `mlx` is import-guarded and not part of the default
  suite; mark and isolate it.
- Add a test with every behavioral change, especially anything touching the
  coupling (`materials.py`), the split logic (`splits.py`), or the coherence
  metric (`models/coherence.py`).

## Adding a material

In `src/pseudomarble/materials.py`, add an entry to `MATERIALS` via `_mat(...)`.
Keep appearance and physics **physically consistent** with one real-world
identity (a slippery material should look it *and* have low friction). Then run
`python tests/test_materials.py` — `test_coupling_is_physically_sensible` guards
the invariants; extend it if you add a material that introduces a new relation.

## Commits & branches

- Branch from the active development branch; one logical change per commit.
- Imperative, scoped subject lines: `materials: add cork`, `coherence: fix
  direction sampling`, `docs: clarify honesty problem`.
- Don't commit generated artifacts — `data/`, `outputs/`, `checkpoints/`, splats,
  and baked caches are git-ignored. Datasets are reproducible from a seed.

## Scope & attribution

This is independent research, **not affiliated with World Labs**, and not an
attempt to reproduce Marble. Keep claims calibrated: we test a *constrained
hypothesis* on a *synthetic substrate*. Cite related work (PhysGaussian,
NeRF2Physics, etc.) when relevant rather than overclaiming novelty.
