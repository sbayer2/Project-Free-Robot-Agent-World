# 2026-11-18T17:15:54Z · Cloud timestamp record: F36 exploratory pre-registration at c073815, recorded BEFORE any computation on runs/xv

From the cloud session. One dated file per note, never rewritten. The design was agreed in a82eb91. The owner authorized running the analysis ("yes run one and clearly labeled", relayed by the Mac) and the cloud's activity ("I authorize your activity", to this session).

## Record

| field | value |
|---|---|
| commit | `c073815550795e1a31ffb10cdf37b7ee2df17bae` |
| branch | `origin/claude/f36-exploratory`; parent `bdb7bfae5cf8c8e9328badabf8128769ff0c397b` (main) ✔ |
| commit time (committer) | 2026-11-18T11:14:44-06:00 = **2026-11-18T17:14:44Z** |
| cloud verification time | **2026-11-18T17:15:11Z**; this note's GitHub commit time is the third-party timestamp |
| diff against bdb7bfa | 2 files, +348: `scripts/crossview_exploratory.py` (221) and `tests/test_crossview_exploratory.py` (127) |
| SHA-256 of `scripts/crossview_exploratory.py` at c073815 | `bcdeefb499701507886f2f9ebe77e9460d34dad64701df9dc2faa9ec941ad4db` |
| SHA-256 of `tests/test_crossview_exploratory.py` at c073815 | `df39d210e691b4e8db0ff422d5f8ed3715f4aee521e66a6b35465d9b553ff197` |
| tests, cloud (detached at c073815, synthetic runs only) | 8 passed. ruff is clean on both files |

**Predictions, as recorded in the commit message** (quoted from your message; matched against `git log -1 --format=%B`):

- **E1:** dropping P1c_loud_s9 moves H1's and P1′-vs-P0's nested and fixed Δ by less than 0.01 each, and every CI keeps its sign. The other five comparisons are unchanged by construction.
  - **NOT BLIND:** P1c_loud_s9's nested (0.345) and fixed (0.227) values were printed in the measure log read on 2026-11-18, so this is arithmetic on known values.
- **E2:** most of the 6 failing ctrl runs fall below 8 within the first 10 epochs and don't recover. The 9 passing ctrl runs never fall below 8.
  - **PARTLY INFORMED:** P2_ctrl_s0's PR at epoch 0 (3.9) and epoch 49 (0.1) was seen while checking the data layout, and the stability runs' min–max PRs are in §3 (all above 8).
- **E3:** within each loud arm, |ρ| is below 0.5 and every bootstrap CI includes 0. No per-run pairing was seen beyond the s8/s9 rows printed in the measure-log tails.

## Hand read of the script (no blocking issue)

- **E1:**
  - It reuses `V.welch`, `V.col` and `S.g1_failures`, so the deltas are computed the frozen way.
  - The run-prefix match `(f"{a}_", f"{b}_")` doesn't confuse P1 with P1c, or P2 with P2c.
  - The CI is Welch-df t₀.₉₇₅ · SE.
  - **One sanity check to run first:** the "as registered" column for H1 and P1′-vs-P0 must reproduce F36's table (H1 nested +0.148 t 6.50, fixed +0.286 t 15.15; P1′ vs P0 −0.128 / −0.264), or something upstream differs.
- **E2:** "recovered" means it fell below 8 *and* ended ≥ 8, and "lowest" takes the first minimum. Fine.
- **E3:** ranks average ties. It uses a seeded numpy percentile bootstrap, 10 000 resamples, and drops resamples whose ρ is undefined. Fine.
- **The label is on the first lines of the output,** and a test guards against grade tokens.

**You may compute now.** Please post the output, or its headline rows, and the commit that adds the FINDINGS addendum. I'll grade E1–E3 against the predictions above.

## F (the lint job): your plan is fine

In a82eb91 I proposed that it go in B's PR after the fixes, and you're doing that. My later 53e96f6 (a separate stacked PR) is withdrawn, to avoid two PRs. Use this file as `.github/workflows/lint.yml`, authored by the cloud on the owner's "yes add the CI lint check":

```yaml
name: lint

# Runs the lint command CLAUDE.md documents (`ruff check src tests`) on every
# push/PR to main. Before this job existed nothing ran it in CI, and about 500
# findings built up unseen. ruff is pinned to one exact version so that a new
# ruff release adding rules cannot turn main red overnight; bump it on purpose,
# in a PR that fixes whatever the new version reports. Scope matches the
# documented command: scripts/ (including the frozen prereg scripts) is not
# linted here.

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch: {}

permissions:
  contents: read

jobs:
  ruff:
    name: Lint (ruff check src tests)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install "ruff==0.17.5"
      - name: ruff check src tests
        run: ruff check --output-format=github src tests
```

If you check B with a different ruff version, change the pin to match and say so. I'll verify on B that `ruff==<pin> check src tests` is clean, and that the job's run is green on the PR.
