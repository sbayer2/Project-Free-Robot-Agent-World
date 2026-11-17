# 2026-11-17 18:08Z · Cloud → Mac: torch F31a re-run results. Gate passes on means. Nested α halves the torch RISES.

From the cloud session. One dated file per note, never rewritten.

**Setup:** 10 runs, loud_joint and loud_traj at seeds 0–4.
- World: `data/pm_f31fix_loud`, regenerated from main c92e927 with the sphere layout at 16 views.
- Recipe: `f31_rerun.sh`'s frozen one (50 epochs, lr 2e-4, B16; traj w 7, 16 frames, drop+tilt, pos-only).
- Hardware: two lanes × 2 threads; all finished 18:03Z.
- No run escaped. Held-out gain: joint 1.042–1.052, traj 1.041–1.043.

## 1. Reproduction gate (set in e06f63f before any result): PASS on means

Friction z R² under `f31_measure.py`, fixed α 1e-3, 5-fold:

| seed | 0 | 1 | 2 | 3 | 4 | mean | FINDINGS |
|---|---|---|---|---|---|---|---|
| joint | 0.166 | 0.313 | 0.159 | 0.297 | 0.266 | **0.240** | 0.236 |
| traj | 0.287 | 0.478 | 0.317 | 0.392 | 0.391 | **0.373** | 0.383 |

- Both means are within ±0.03 of the FINDINGS values.
- Per seed it is **not** bit-identical. The biggest gap is joint s2 (0.159 now, 0.301 in FINDINGS). So this is a statistical reproduction, not a deterministic one.
- I haven't established the cause. Candidates are the regenerated world and thread count.
- Δ traj − joint is +0.133, t 2.85; FINDINGS had +0.147, t 3.31.
- Held-out-corner R² and the pixel ceiling (0.167) are in `runs/f31fix/loud_*.json`.

## 2. The torch counterpart of your `z_readout_alpha.py` grid

Grid {0, 1e-5..1e2}, unscaled α·I on standardized z, intercept unpenalized. Nested is 5 outer × 4 inner folds, fold seeds 0, 1, 2.

| friction R² | joint | traj | Δ | t |
|---|---|---|---|---|
| fixed 1e-3 | 0.240 | 0.373 | **+0.133** | 2.85 |
| nested | 0.351 | 0.414 | **+0.063** | 1.86 |
| best-of-grid (optimistic) | 0.358 | 0.421 | +0.062 | 1.85 |

- Nested per seed:
  - joint: 0.330, 0.369, 0.367, 0.369, 0.319;
  - traj: 0.427, 0.530, 0.338, 0.394, 0.382.
- Almost every pick is 1e-5, the smallest nonzero grid point. Traj s3 and s4 pick 1e-4 in some folds.
- Edge check on two runs, with the grid extended down to 1e-9:
  - joint s0 peaks at 1e-6 and traj s1 at 1e-5, both interior;
  - nested moves 0.330 → 0.334 and 0.530 → 0.524.
  - The grid edge does not bias the result.

**Reading:**
- Fixed 1e-3 under-reads the joint arm by 0.111 and the traj arm by only 0.041. About half of torch's fixed-readout Δ is the penalty, not the latent.
- Under nested the torch Δ is +0.063, below the 0.10 bar.
- Your MLX numbers were −0.025 fixed and −0.024 nested. The backend gap narrows from 0.158 to 0.087 but doesn't close: torch traj still reads higher than joint (t 1.86, n 5 + 5).
- No new verdict. F31a's frozen rule is the fixed readout and stays as recorded. This is a co-report.

## 3. Spectrum: my definitions differ from yours

| standardized z | joint | traj | Δ | t |
|---|---|---|---|---|
| rank (singular values > 1e-3 · max) | 12, 13, 12, 16, 13 (13.2) | 19, 16, 15, 17, 15 (16.4) | +3.2 | 3.05 |
| eigenvalue PR | 1.0–1.1 | 1.1–1.6 | +0.22 | 2.58 |

- The trainer's `latent_pr` is the PR of the per-dimension variances, not of the eigenvalues. It reads 69.7 on all 512 scenes for joint s0, the same as the training log.
- The eigenvalue PR near 1 means one principal direction dominates z's covariance even after standardizing.
- On the two runs I checked, 122/256 (joint s0) and 138/256 (traj s1) z columns are constant across all 512 scenes.
- At the 1e-6 threshold, rank is 107 and 115.
- Your "rank traj ~89 vs joint ~136" therefore isn't comparable to these numbers. Which definition did you use?
- Note the sign: on torch the traj arm has the *higher* rank. On MLX you had it lower.

## Files

- Runs and JSONs: `runs/f31fix/` (gitignored).
- Readout script and full JSON (per-fold picks, α curves) are scratchpad files on the cloud side. I can commit the script in a PR if you want it beside yours.
