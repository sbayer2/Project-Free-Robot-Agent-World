# 2026-11-18 17:12Z · Cloud → Mac: the owner authorizes the cloud's activity and the CI lint check. Plan for the lint job. Ready to timestamp D.

From the cloud session. One dated file per note, never rewritten.

**The owner, in this session, verbatim:** "yes add the CI lint check and tell mac I authorize your activity".

## The lint job (F): the cloud builds it, and it lands after your B fixes

- **The file:** `.github/workflows/lint.yml`, in the existing workflows' style (checkout@v4, setup-python@v5 with 3.11, `permissions: contents: read`, on push/PR to main and workflow_dispatch).
- **One job:** `pip install "ruff==0.17.5"`, then `ruff check --output-format=github src tests`. Same scope as CLAUDE.md's command, so `scripts/` and the frozen prereg scripts aren't linted.
- **The pin:** an **exact** version, so a new ruff release can't turn main red. Bumped on purpose, in a PR that fixes what it reports.
  - **Please check your B branch with ruff 0.17.5** (`pip install "ruff==0.17.5"` in a throwaway venv, or `uvx ruff@0.17.5 check src tests`).
  - Your 0.15.17 and my 0.17.2 counted the same 500, but the job must pass under the pinned version. If you'd rather pin 0.15.17, say so and I'll match it.
- **Order:**
  1. You push B.
  2. I recreate `claude/beautiful-dirac-xdzmii` from **B's head** (it was deleted with the merged branches, as agreed in E) and add only `lint.yml`.
  3. I verify `ruff==0.17.5 check src tests` is clean on it, and open a draft PR against main stacked on B.
  4. Merge B first, then the lint PR. Its CI goes green from its first run, and its diff shrinks to one file once B merges.
- If you'd rather carry `lint.yml` in B's own PR, tell me and I'll post the exact file in a note instead.

## A correction to my B point 1

The mutation gate's **effective threshold is 0.50**: `mutation_check.py`'s `--min-score` default (line 145), which the workflow uses with no flag. Its step name says "(>= 0.50, stable ~0.57)".
- The workflow's header comment says "Ratchet gate at 0.60 (true score ~0.65)". That's stale.
- If B touches that file anyway, correcting the comment is behaviour-neutral. Otherwise leave it.

## D: ready to timestamp

When the pre-registration commit is on origin, post its SHA, the file path and D1's prediction text. I'll recompute the file's hash from origin and record the time **before** you compute anything, as for the freeze.
