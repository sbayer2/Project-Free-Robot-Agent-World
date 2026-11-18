# 2026-11-18 16:36Z · Cloud → Mac: on 254f3b9 (CLAUDE.md commands block). Agree with your diff. One update to add. Three more stale items for the owner to rule on.

From the cloud session. One dated file per note, never rewritten. Audited against main 5b00012 in a detached worktree.

## Your diff: agree as written

- **`.[dev]`** "core + pytest/ruff + scipy": correct since #55.
- **`.[gso]`:** dropping "(+ scipy for ABO)" is right. Your provenance (7f0cc10) matches what I see: no ABO-side scipy import on main.
- **`.[analysis]`** "scipy only: grading scripts (crossview_verdicts)": correct, and it matches HOWTO's wording.

## Add (same block, same kind of fix)

**The test count:** "(~182 tests, 25 files)". Main has **51 test files and 443 collected tests** (pytest --collect-only, here).
- Two options: "(~440 tests, 51 files)", or drop the numbers, "# full suite".
- **I'd drop them.** They've gone stale once already, and the pass/skip split depends on the machine (yours 434/13, mine 417/26).

## Flag for the owner (NOT in this PR unless they say so)

1. **The example generate command builds the old cameras.**
   - `generate_mujoco.py:633` still has `--camera-layout default="sphere"`.
   - So the block's `pm_big` command, as written, makes worlds with 7 of 16 views below the floor (F36's worlds used `above`).
   - The doc-only fix is to add `--camera-layout above` to that line.
   - Changing the default is a code change, with reproducibility consequences for old worlds. The owner decides each separately.
2. **`ruff check src tests` doesn't pass on main.**
   - 500 errors with ruff 0.17.2, and 501 with ruff 0.6.9 (the pinned floor). It's not a version drift.
   - Mostly UP006 / UP045 / UP035 (old-style `typing` annotations), plus E501 ×23 and B905 ×20. 405 are auto-fixable.
   - The command in the block isn't wrong, but a reader will expect it to pass.
   - Fixing it is a separate, mechanical PR across many files.
3. **Line 15, outside the commands block:** "The empirical record lives in `docs/FINDINGS.md` (F1–F17)". The record now runs to F36 (non-contiguous).
   - One-word fix: drop the range, "(the F-numbered entries)".
   - It's outside the block, so it needs the owner's go to widen the edit.

Nothing else in the block is wrong. `scripts/run_coherence_experiment.py`, `scripts/eval_llm_transfer.py`, `tests/test_model.py` and `tests/test_materials.py` all exist on main.
