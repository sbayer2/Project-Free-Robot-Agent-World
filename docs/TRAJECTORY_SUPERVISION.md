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

## 9. Result (runs 2026-09-26; recorded 2026-10-23)

Run as frozen; `runs/f31/f31_verdicts.json`. Gates: G1 PASS (PR 40.6–89.2),
G2 PASS (ctrl friction z R² −0.165 to −0.065). **H1 NULL** (Δ +0.040,
t 1.05). **H2 INCONCLUSIVE** (Δ −0.99, t −1.30), driven by loud-traj seed 4,
which never left predict-the-mean (gain 1.06) despite PR 49. P1–P4 all
correct. Post-hoc observations (escape delayed by ~12 epochs in every traj
seed; friction seed sd 0.082 → 0.027; the PR gate does not catch
non-escape) are recorded in FINDINGS, "F31a". Consequence per §7: F31b on
this world is not worth building; the next lever is legibility.

## 10. Amendment 1 (2026-10-23, before any fixed-world run): rerun on the repaired ramp

**Why.** PR #45 (F34) found the tilt ramp ran at 0.349°, not 20°. Every run
in §9 used pre-fix worlds. On the fixed ramp (`data/pm_f31fix_*`, same seed
and flags; drop, push, friction and renders byte-identical, only tilt
changes) both arms' supervision carries far more friction. Friction R² from
the 21 summaries is 0.34 → 0.83, and the tilt position block gain is 2.70 →
16.16. §9's NULL therefore cannot separate two explanations:
(a) **pixel limit**: the renders barely show friction (linear pixel
ceiling 0.167; the appearance *parameters* determine friction at 5-fold R²
0.998, so the loss is all in rendering legibility at 128 px), versus
(b) **weak target**: on the broken ramp neither head's target asked for
much friction.

**Unchanged (every frozen rule in §§3–6 holds):** arms, seeds, recipe,
the trajectory head (weight 7.0, drop+tilt positions, 16 frames), gates G1
and G2, H1 and H2 thresholds, and `scripts/f31_verdicts.py` as frozen. The
weight rule is not re-derived on the fixed world; 7.0 stays.

**Data.** `data/pm_f31fix_{loud,ctrl}`; runs under `runs/f31fix/`.

**Staging.** Stage A: loud-joint seeds 0–4, then report. Stage B:
loud-traj seeds 0–4, then ctrl-joint and ctrl-traj seeds 0–2.

**New registered co-report C1 (the pixel-limit-vs-weak-target test).**
Δ_C1 = mean friction z R²(loud-joint, fixed world, 5 seeds) − 0.284 (the
§9 loud-joint mean on the broken world), with Welch t against those 5 §9
values, classified on H1's thresholds: RISES (Δ ≥ +0.10 and t ≥ 2.5), NULL
(|Δ| < 0.05), FALLS, or INCONCLUSIVE.
- C1 RISES → (b) was binding: a target that asks for friction pulls more
  of it out of the same pixels, so §9's NULL was partly the broken ramp.
- C1 NULL → (a) is binding: even a strongly friction-bearing behavior
  target leaves z's friction where the pixels put it.

**Non-escapers reported separately.** A run is a non-escaper if its final
held-out gain is < 2.0 (§4's escape threshold). H1 and H2 are read as
frozen over all seeds (primary). They are also reported over escaped seeds
only (secondary, labeled), with each non-escaper listed by seed. A
non-escaper's friction z R² is still reported.

**Predictions (stated now, at real odds):**
- **P5 (C1): NULL or INCONCLUSIVE, 60 / 40 against RISES.** The pixel
  legibility gap (0.998 in the parameters vs 0.167 linearly in the
  pixels) looks like the binding constraint. But joint z already exceeded
  the linear pixel ceiling (0.284), so some nonlinearly readable friction
  exists, and a stronger target could pull more of it.
- **P6 (H1 on fixed worlds): NULL or INCONCLUSIVE, 65 / 35**, unchanged
  from P1. The trajectory target's extra friction (0.92 vs 0.83 from
  summaries) is small once the behavior target already carries it.
- **P7:** the trajectory arm's escape is still delayed relative to joint
  (descriptive, 60 / 40).

**Contamination disclosure.** Seen before this amendment: every §9
number; the fixed-vs-broken oracle table above; the appearance-parameter
ceiling (0.998). Not seen: any model trained on a fixed world.

Reproduce: `scripts/f31_rerun.sh`, then `python scripts/f31_coreport.py
--root runs/f31fix --broken runs/f31`.

## 11. Amendment 2 (2026-10-24 01:2x UTC; before any ctrl-traj run had started)

**Disclosure: H1 was read before G2.** The loud-traj arm was measured as
soon as its five seeds finished, before any ctrl run completed:
friction z R² 0.308 / 0.458 / 0.389 / 0.385 / 0.374 (mean 0.383) against
loud-joint 0.236. That gives Δ +0.150, Welch t 3.36, df 6.8, 95 % CI
[+0.044, +0.256]. It would read RISES if the gates pass, and **P6 would
then be wrong** and must be reported as such at equal prominence. Only
ctrl-joint seeds 0–2 had finished when this was written; no ctrl-traj
run existed.

**Unchanged:** every gate, bar and verdict rule in §§5 and 10, and
`scripts/f31_verdicts.py`. The co-reports below are **descriptive only**.

**(a) Leakage-adjusted difference.** Δ_L = (loud-traj − loud-joint) −
(ctrl-traj − ctrl-joint) on mean friction z R², with SE = √(Σ var/n) over
the four arms. If G2 passes and Δ_L stays ≥ +0.10, the RISES reading
stands. If Δ_L shrinks toward 0, the loud difference was leakage: the head
teaching z friction-correlated structure that the ctrl world shares.

**(b) Held-out friction readout.** A ridge is fit on training-scene z and
scored on the 20 held-out corner scenes, per run, in all four arms
(`f31_measure.py`, `z_r2_heldout`). It is post hoc for the loud arms and is
labelled that way. It guards against the 5-fold all-scene R² rewarding
memorized training geometry.

**(c) Target scale on the fixed world (measured before this amendment;
data only, no model involved).** Predict-the-mean MSE on the loud train
split:

| | behavior (21) | trajectory (96, drop+tilt positions) | effective trajectory : behavior loss at w = 7.0 |
|---|---|---|---|
| broken world | 0.01662 | 0.00239 | 1.005 |
| fixed world | 0.13741 | 0.37900 | **19.3** |

The weight rule (§3) equalized the two heads' starting losses *on the
broken world*. On the fixed world the trajectory head starts with about
19× the behavior head's loss. So the fixed-world traj arm asks for the
trajectory roughly 19 times harder than the joint arm asks for the
summaries. **No claim of the form "how the target asks, not how much" can
be made from this design.** A RISES is compatible with "more friction
pressure" just as well as with "denser, time-resolved friction pressure".
Separating the two would need a fixed-world arm at w ≈ 0.36 (equalized), not
run here.

**H2 is degenerate on the fixed world.** All arms, ctrl included, sit at
held-out gain 1.03–1.05: the gain metric is saturated by the
friction-driven tilt slide (79 % of target variance). H2 is reported as
**metric saturation, not "no cost to prediction"**.
