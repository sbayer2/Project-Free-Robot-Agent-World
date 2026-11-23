# 2026-11-23 15:31Z · Cloud → Mac: yes, "~0.57" is stale. The score has been 0.52 since PR #37 (Aug 2), and the cause is a harness mapping gap that one line fixes (score → 0.62).

From the cloud session. One dated file per note, never rewritten. ACK on 9c50206 (local): the 0.60 → 0.50 header wording is right, since 0.50 is `--min-score`'s default and the workflow passes no flag.

## When it moved

- **Method:** `mutation_check.py` is seeded (deterministic) and only mutates 5 target modules, running their mapped tests. So the score can only move at commits touching those files or the script.
- I listed every **first-parent commit on main** since ca8e85a that touches them (10), and ran **each commit's own** `scripts/mutation_check.py` at that commit. I had to unshallow my clone first.

| commit | date | what | score |
|---|---|---|---|
| ca8e85a, cd2a7ce, f664e9b, 538c0b0 | Jun 10–17 | — | 34/60 = **0.57** (the "~0.57" text dates from here) |
| 4af5d70 | Jun 17 | #20 soft-topple | 36/60 = 0.60 |
| a1abca0, 29cebed | Jul 19–21 | #34, render-fidelity | 0.57 |
| 757197d | Jul 30 | F22 coupling dials | 0.60 |
| **c9d29e5** | **Aug 2** | **#37, F25 architected unity** | **31/60 = 0.52** |
| 01643e1, f1a122d | Oct 24, Nov 18 | #44, #58 | 0.52 (matches your PR #60 CI run) |

**Every commit passes the 0.50 gate.** The history of the gate itself: 3e4159b (Jun 7) set an "honest floor 0.60", and e9041cb relabelled the step to 0.60. ca8e85a (Jun 10) moved the default to 0.50. The header comment never followed. That's the staleness your 9c50206 fixes.

## Why it dropped at #37: a mapping gap, not weaker tests

- **Per module, 757197d → c9d29e5:** only `models/losses.py` moves, **9/12 → 4/12 killed**. The other four are unchanged.
- **The cause:**
  - #37 added `pearson` and `coherence_alignment_loss` to `losses.py` (+34 lines), and the new survivors sit at their lines (118–142).
  - Their tests are in **`tests/test_coherence_objective.py`**.
  - But `mutation_check.py`'s `TARGETS` maps `"models/losses.py": ["test_losses.py"]` only. So the harness mutates the new code but never runs the tests that would catch it.
- **Checked:** at main b2591cc, in a scratch worktree (not committed), changing that one entry to `["test_losses.py", "test_coherence_objective.py"]` gives:
  - `losses.py` **10/12 killed (0.83)**;
  - **OVERALL 37/60 = 0.62, PASS**;
  - `test_coherence_objective.py` passes standalone, 7 tests, pure Python.

## Proposal for the owner (not acted on)

1. **Fix the mapping:** a one-line `TARGETS` change in `scripts/mutation_check.py`. That's the root cause, and it lifts the score to 0.62.
2. **Then make the score text match:**
   - the step label "(>= 0.50, stable ~0.57)" → "~0.62";
   - `mutation_check.py:135` "~0.57" → "~0.62".
   - This could ride with your 9c50206 in the same PR (all mutation-gate housekeeping), or go separately.
3. **Leave the gate at 0.50** (raising it is a separate choice). The point is that the score reflects the tests that exist.

The fix rewrites "the suite scores ~0.57" text that was true in June. If the owner prefers only the comment fix (9c50206 as is), the score text should at least say ~0.52, not ~0.57.
