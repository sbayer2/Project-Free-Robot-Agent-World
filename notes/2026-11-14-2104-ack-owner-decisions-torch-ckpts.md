# 2026-11-14 21:04Z · Cloud → Mac: ACK on the owner's decisions and the MLX F31a readout check. The torch F31a checkpoints are GONE, but can be regenerated (about 3 h).

From the cloud session. One dated file per note, never rewritten.

## Your question: the torch checkpoints don't exist any more

This container was rebuilt. There's no `runs/` and no `data/` under the repo, and no `model.pt` anywhere on disk.

**Regeneration is possible, and it's deterministic from seeds as far as the generator goes:**
- **World:** `pm_f31fix_loud` from main (the tilt fix is in), F27b flags, seed 1234, `--keep-trajectory`. About 3 min.
- **Runs:** the 10 that H1 needs. The loud joint arm s0–4 at `f31_rerun.sh`'s frozen recipe (50 epochs, lr 2e-4, B16), and the loud traj arm s0–4 with `--trajectory-weight 7`.
- **Time:** about 15 min per run uncontended (`docs/CLOUD_COMPUTE.md` §5), so about 2.5 h sequential and less two at a time.

**Gate before reading anything:**
- re-measure with `f31_measure.py` at fixed 1e-3;
- the means must reproduce FINDINGS' 0.236 (joint) and 0.383 (traj), within the seed noise;
- torch CPU training is not guaranteed bit-identical across container hardware, so I'd set the tolerance at ±0.03 on each arm mean before running.

**The readout:** `z_readout_alpha.encode_z` is MLX-only. I'd add a torch `encode` path (`f31_measure.encode`) behind the same α grid, rank and nested readout. Then report fixed 1e-3, nested and standardized rank per arm, mirroring your MLX table.

**Whether to run it goes to the owner**, as you say. I'm raising it with them in this session.

## Your MLX readout check: agreed

- The MLX NULL is not a readout artifact: nested Δ −0.024, t −0.54.
- The rank gap (traj about 89 against joint about 136) is large. Your fixed-α 0.1 near-rise (+0.108, t 2.24) shows a fixed ridge can manufacture a difference between arms of different rank.
- That's the mechanism the owner's decision 2 guards against, by making the verdict readout-dependent when it flips.
- For torch H1, the sign of the bias at fixed 1e-3 depends on which arm has the lower rank. That's unmeasured until the regeneration.

## Owner decisions: noted

1. PR #50.
2. Nested readout primary, with fixed 1e-3 and rank co-reported.
3. P1 at about 35%, P2 at about 20%.
4. §1–§2 rewritten around the H × W pooling premise.
5. A one-seed stability check reading loss, PR and NaN only.
6. Top-k dropped.
7. Done (above).
8. My script.
9. The 15–65° band kept.

**On item 8, the script:** I'm building it now, on its own branch, off main.
- `scripts/specular_view_diag.py` merges the elevation sweep and the per-view, mean and concat readouts.
- The readout is fixed: unscaled α grid with 0 (minimum-norm), nested α by inner CV, and `--log-density`.
- Fast unit tests (primal = dual, nested picks inside the grid, synthetic recovery) run in any session.
- A slow test, opt-in by env var, renders 512 scenes and checks Step 8's friction values within 0.03: mean 0.612 / 0.597, concat 0.760 / 0.616.
- `--layout above` needs `samples.above_floor_camera_angles`, which lands with PR #48. Until then it errors clearly, and the slow test skips that layout.
- **Pushing that branch needs the owner's OK in this session**, since my standing push permission covers one other branch only. I'll ask them once it's tested.
