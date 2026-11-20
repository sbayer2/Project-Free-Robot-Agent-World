# 2026-11-20 18:47Z · Cloud → Mac: closing verification of the A–F round. Every item confirmed from origin.

From the cloud session. One dated file per note, never rewritten. For the record while the Mac is stopped by the usage limit.

| item | evidence on origin |
|---|---|
| #57 (F36 exploratory addendum) | merged as 2be2d4a |
| #58 (ruff fixes + lint job) | merged as f1a122d (parents 2be2d4a, 325766d) |
| #59 (line 15) | merged as 2b92420 (parents f1a122d, 125ba25); CLAUDE.md line 15 now reads "(the F-numbered entries)" |
| Lint job on main | runs #2 (f1a122d) and #3 (2b92420), both **success** |
| `.github/workflows/` on main | adversarial-verify.yml, lint.yml, security.yml |
| F36 §6 hash on main | `d4f58988…`, unchanged |
| deleted remote branches | ruff-clean, claude-md-line15, f36-exploratory, f36-findings, crossview-pooling-prereg, claude-md-extras and beautiful-dirac-xdzmii: none remain |
| kept | cloud-notes and f31a-mlx-replication, plus 10 older branches the owner didn't name |

**Still open with the owner:**
- the `--camera-layout` default (`sphere`) in `generate_mujoco.py` and CLAUDE.md's example;
- the stale "0.60" header comment in `adversarial-verify.yml` (the effective gate is 0.50);
- `claude/f31a-mlx-replication` (GPU-encoded readouts, no PR).

Nothing pending from the cloud.
