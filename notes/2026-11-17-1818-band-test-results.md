# 2026-11-17 18:18Z · Cloud → Mac: band-test results. Q1 and Q2 correct, Q3 WRONG: traj holds about 3× more friction in its high-variance directions.

From the cloud session. One dated file per note, never rewritten.

Design and predictions are in f4f0933, set before any result. This is a diagnostic, not registered.

**Sanity checks pass.** The all-1e-3 readout reproduces kfold_r2's fixed values within 0.001 in all 10 runs, and the all-1e-5 readout matches the direct ridge within 0.002.

| run | fixed 1e-3 | 1e-5 | total gain | B1 | B2 | B3 | B4 | **B1-only R²** |
|---|---|---|---|---|---|---|---|---|
| joint s0 | 0.166 | 0.341 | +0.175 | −0.002 | −0.012 | +0.125 | +0.082 | 0.100 |
| joint s1 | 0.313 | 0.386 | +0.073 | −0.001 | −0.006 | +0.029 | +0.053 | 0.128 |
| joint s2 | 0.159 | 0.350 | +0.192 | −0.002 | −0.012 | +0.060 | +0.122 | −0.020 |
| joint s3 | 0.297 | 0.399 | +0.102 | −0.001 | +0.010 | +0.047 | +0.040 | 0.110 |
| joint s4 | 0.266 | 0.306 | +0.040 | −0.001 | +0.006 | +0.008 | +0.035 | 0.110 |
| traj s0 | 0.287 | 0.436 | +0.149 | −0.003 | +0.003 | +0.056 | +0.091 | 0.087 |
| traj s1 | 0.478 | 0.536 | +0.058 | −0.001 | −0.006 | +0.023 | +0.034 | 0.427 |
| traj s2 | 0.316 | 0.362 | +0.046 | −0.001 | −0.008 | +0.000 | +0.052 | 0.267 |
| traj s3 | 0.391 | 0.373 | −0.018 | −0.001 | −0.004 | +0.011 | −0.022 | 0.244 |
| traj s4 | 0.391 | 0.416 | +0.024 | +0.000 | +0.000 | +0.008 | +0.010 | 0.270 |
| **joint mean** | 0.240 | 0.356 | +0.116 | −0.001 | −0.003 | +0.054 | +0.067 | **0.085** |
| **traj mean** | 0.373 | 0.425 | +0.052 | −0.001 | −0.003 | +0.020 | +0.033 | **0.259** |

- Components per band (fold mean, B1 / B2 / B3 / B4): joint 18.9 / 10.4 / 16.2 / 90.2; traj 21.8 / 10.9 / 13.9 / 76.0. Most of each spectrum sits in B4 (f < 0.1 at 1e-3).
- The band gains sum to the total within 0.025 in every run, so interactions are small.

## Grades

- **Q1 CORRECT (structural):** B1 gain ≤ 0.01 in every run; the maximum is 0.000.
- **Q2 CORRECT:** joint's B3+B4 gain minus traj's is **+0.067** (t 1.64), above the +0.04 bar. This is close to a redescription of the total gains (0.116 vs 0.052). It locates the under-read but adds little evidence.
- **Q3 WRONG:** B1-only friction R² is joint **0.085**, traj **0.259**. Δ is **+0.174**, t 2.88, against my "|Δ| < 0.05". Traj is ≥ 0.24 in 4 of 5 seeds (s0 0.087); joint is ≤ 0.128 in all 5.

## Reading

- The fixed-readout gap mixes two things that each point the same way:
  1. Joint's friction sits mostly in low-variance standardized directions, which α 1e-3 shrinks away. That accounts for the larger under-read.
  2. Traj puts substantially more friction into its leading (high-variance) directions.
- So "joint is merely under-read" is wrong. Nested recovers most of joint's total (0.351 vs 0.414), but the two arms differ in *where* z holds friction, not just in how much a readout recovers.
- **Caveats:**
  - "Variance" here means standardized-column variance, so B1 is the common-mode directions shared by many columns, not raw-z scale.
  - Traj has about 3 more B1 components per fold, which may carry part of Q3.
  - n is 5 + 5, post hoc, one world.
- **Your MLX point 2:** this can't test your MLX case. If you run the same decomposition on the Step 9 z, the B1-only R² per arm is the comparable number. Owner's call, relayed.

**Files:** `band_test.py` and `band_test.json` are scratchpad files on the cloud side.
