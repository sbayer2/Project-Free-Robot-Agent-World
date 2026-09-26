# F31a preregistration — does trajectory supervision put friction into the latent?

*(Frozen 2026-09-26 in the commit that adds this file and
`scripts/f31_verdicts.py`, before any trajectory-supervised model has been
trained. Written in the program's first cloud session; every run is torch
on CPU, which passed the torch↔MLX parity gate (`docs/CLOUD_COMPUTE.md`
§5). All four arms train on torch, so no cross-backend comparison is made.)*

## 1. Why this, why now

F31 was registered (F28 Amendment 2) as "a dynamics head": the honest
disanalogy with Dyna-2 is that this program's render head is static.
The F31 pilot (FINDINGS, "F31 pilot") established the preconditions:

- The recorded probe paths carry friction that the 21 summaries discard
  (essence R² read back out: 0.84 vs 0.34).
- The essence-determined content sits in the **drop and tilt position
  paths** (forward gain 4.87 and 2.70); push and orientation are saturated
  or noise.

So F31 splits. **F31a (this document)** asks the precondition: does
supervising z with those paths change **what z contains**, above all
friction? **F31b** (not designed here) would ask whether that changes
unity, using the validated instruments (F28 directional, F29 transfer)
ported to torch. If F31a is NULL, F31b has no content to find.

**F31a is not a unity test** and makes no claim about "one latent, two
projections."

## 2. Baselines measured before this freeze (disclosed)

Measured with `scripts/f31_measure.py` on existing checkpoints and pixels
only; no trajectory-supervised model existed.

| loud world (r = 0.992) | density | friction | restitution |
|---|---|---|---|
| pixel ceiling (F19 pixel features, 5-fold ridge R²) | 0.233 | **0.167** | 0.946 |
| joint torch latent, seeds 0–2 (parity runs) | 0.646 / 0.664 / 0.654 | **0.241 / 0.379 / 0.168** | 0.987 / 0.983 / 0.986 |

ctrl world (r = 0.008) pixel ceiling: −0.030 / −0.039 / −0.019. The pixels
carry nothing about the material, as designed.

Target scale, measured on the loud train split: predict-the-mean MSE is
0.01662 for the 21-dim behavior target and 0.00239 for the 96-dim
drop+tilt position trajectory, a ratio of 6.97.

## 3. Arms (torch, F27b recipe: lr 2e-4, 50 epochs, batch 16, 128 px, 16 views)

| arm | world | trajectory head | seeds |
|---|---|---|---|
| loud-joint | `pm_f31_loud` | off | 0–4 (0–2 are the parity runs, same recipe) |
| loud-traj | `pm_f31_loud` | on | 0–4 |
| ctrl-joint | `pm_f31_ctrl` | off | 0–2 |
| ctrl-traj | `pm_f31_ctrl` | on | 0–2 |

Trajectory head, frozen: `--trajectory-weight 7.0 --trajectory-frames 16
--trajectory-probes drop,tilt --trajectory-pos-only` (96-dim target,
head width 256). **Weight rule:** 7.0 ≈ 6.97 equalizes the two heads'
predict-the-mean losses, so neither target starts out dominating the
gradient. No other weight is run.

## 4. Readouts (all from `scripts/f31_measure.py`)

- **z R²**: 5-fold ridge from the trained latent z (all 512 scenes) to each
  normalized essence axis (F19's probe).
- **gain**: final-epoch held-out behavior gain on the 20-scene corner split
  (the F27/F29 metric).
- **PR**: final latent participation ratio.
- **escape epoch**: first epoch with held-out gain ≥ 2.0.

## 5. Hypotheses and frozen rules

Every Welch t is two-sample over the listed seeds. Power disclosure: with
the baseline friction sd of about 0.107, the smallest difference reaching
t = 2.5 at n = 5 per arm is about 0.17. Effects between 0.05 and 0.17
will read INCONCLUSIVE, and that is the honest reading at this n.

**Gates (read first; any failure withholds the dependent verdicts):**
- G1 health: every run's final PR ≥ 8.
- G2 control: every ctrl run (joint and traj) has friction z R² < 0.10.
  The ctrl pixels carry nothing, so a positive here means leakage (the
  head teaching z something it cannot see). A G2 failure withholds H1.

**H1 — friction extraction (primary).** Δ = mean friction z R²(loud-traj)
− mean(loud-joint).
- RISES: Δ ≥ +0.10 and t ≥ 2.5
- NULL: |Δ| < 0.05
- FALLS: Δ ≤ −0.10 and t ≤ −2.5
- otherwise INCONCLUSIVE

**H2 — behavior prediction (secondary).** Δ = mean gain(loud-traj) −
mean(loud-joint).
- RISES: Δ ≥ +0.5 and t ≥ 2.5
- FALLS: Δ ≤ −0.5 and t ≤ −2.5
- NULL: |Δ| < 0.25
- otherwise INCONCLUSIVE

**Descriptive (no verdict):** density and restitution z R² (loud),
escape epochs per arm, held-out trajectory gain (traj arms).

## 6. Predictions (registered at real odds)

- **P1: H1 is NULL or INCONCLUSIVE, 65 / 35 against RISES.** Friction
  reaches the pixels only through roughness, which F19/F21/F28 found
  nearly illegible at 128 px. The linear pixel ceiling (0.167) sits below
  what joint z already holds (0.26). My expectation is that supervision
  cannot manufacture legibility; a RISES would say the encoder had friction
  signal available that the behavior summaries never asked it to keep.
- **P2: H2 is NULL or INCONCLUSIVE, 60 %** (RISES 20, FALLS 20).
- **P3: G2 passes, 90 %.**
- **P4 (descriptive):** density z R² does not rise. Drop and tilt paths
  are mass-blind (pilot §3), so a 96-dim target competing for z has no
  reason to keep density.

## 7. What each outcome means

- **H1 RISES:** a richer, world-referencing target changes what the latent
  keeps, even for a nearly illegible channel. F31b becomes worth building.
- **H1 NULL:** supervision does not add material content that the pixels
  barely carry. F31b on this world would be measuring nothing. The next
  lever is legibility (the noted 256 px oblique regeneration), not
  targets.
- **H2 FALLS with H1 RISES:** content bought at the price of prediction
  (a capacity trade), the F25 k-ladder pattern.

## 8. Contamination disclosure

Seen before freezing: all pilot numbers (FINDINGS "F31 pilot"), the parity
curves (loud-joint seeds 0–2, including their friction z R² above), and
both worlds' pixel ceilings. Not seen: any trajectory-supervised model,
loud-joint seeds 3–4, and any ctrl model.

Reproduce: see §3 for the runs; then `scripts/f31_measure.py` per arm to
`runs/f31/{loud_joint,loud_traj,ctrl_joint,ctrl_traj}.json`, then
`python scripts/f31_verdicts.py`.
