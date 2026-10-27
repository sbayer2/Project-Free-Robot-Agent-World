# Cross-view pooling: does replacing the view average let z hold friction?

**Status: DRAFT preregistration, 2026-10-27. Not frozen.** It freezes in the
commit that adds the implementation (all three backends, with cross-backend
tests, per CLAUDE.md) and `scripts/crossview_verdicts.py`, before any
pooling-variant model is trained. Until then every number, arm and bar here
can change. F-number: unassigned (the owner assigns it at the freeze).

## 1. Why this, why now

The encoder averages its 16 per-view latents into z (`models/mlx_net.py`,
`Encoder.__call__`, last line; mirrored at `numpy_net.py:126` and
`torch_net.py:95`). Anything carried by how the views *differ* is discarded
before z exists.

Friction reaches the photos only through roughness, rendered as Phong
shininess (`shininess = 1 − roughness`): the tightness of the specular
highlight. A highlight appears only where camera, light and surface line
up, so it is a cross-view signal. The cross-view pilot (§2) found fixed
image statistics carry far more friction when the views are compared than
when they are averaged.

Two further facts make pooling the suspect:

- z is already supervised on friction directly: the essence head (weight
  0.3, `mlx_net.loss_fn`) regresses all three essence axes from z. Under
  that direct supervision, mean-pooled z still holds friction at 0.267
  (5-fold, MLX). The target is not what is missing.
- The project's "pixel ceiling" (0.167, F19 pixel features) is computed on
  the averaged view. The F28 and F31a legibility arguments rest on it.

The alternative explanation is upstream of pooling: the per-view CNN
(stride-2 convolutions, then a global average over H × W) may already have
smeared the highlight before the views are combined. Diagnostic D1 (§4)
separates the two.

This is a question about **appearance** (the F13–F31 line). It is separate
from F35 (`docs/CROSS_INTERACTION.md` on `claude/f35-cross-interaction-design`),
which keeps appearance fixed and reads physics from motion.

## 2. Baselines measured before this draft (disclosed)

**Cross-view pilot**, post hoc, no training (`scripts/crossview_pilot.py`):
nested 5-fold ridge R² (α chosen by inner 4-fold CV), mean of 3 fold seeds,
plus ridge on the train split scored on the 20 held-out (corner) scenes.

| `pm_f31fix_loud` (r = 0.992) | dim | friction (seed range) | held-out friction | density | restitution |
|---|---|---|---|---|---|
| A mean-view pixels (the "pixel ceiling") | 198 | 0.177 (0.169–0.186) | 0.321 | 0.233 | 0.946 |
| C per-view stats, mean + std across views | 26 | 0.309 (0.300–0.327) | 0.406 | 0.553 | 0.975 |
| D \|DFT over view index\| | 117 | **0.636** (0.622–0.644) | 0.678 | 0.657 | 0.979 |
| D′ same, views shuffled first | 117 | **0.613** (0.595–0.628) | 0.678 | 0.647 | 0.980 |
| G per-view stats sorted across views | 208 | 0.469 (0.437–0.489) | 0.615 | 0.563 | 0.980 |
| H mean + \|deviation\| quantiles | 65 | 0.491 (0.478–0.501) | 0.662 | 0.656 | 0.980 |

- Every feature set reads friction at or below zero on `pm_f31fix_ctrl`
  (−0.08 to −0.01) and on `pm_big` (−0.06 to −0.02), where the photos carry
  no friction by construction.
- **D ≈ D′:** shuffling the view order costs 0.02. The Fourier structure is
  not doing the work. The cameras sit on a Fibonacci sphere, so the view
  index has no natural order; |DFT| acts as a set of cross-view energy
  features. No Fourier arm is preregistered (§8).
- These are fixed hand-made features with a linear readout. They bound
  what the photos contain from below. They are not what a trained encoder
  will learn.

**Trained baseline**, MLX, mean pooling, F27b recipe, joint arm, repaired
worlds (`docs/F31A_MLX_REPLICATION.md` on `claude/f31a-mlx-replication`):

| | seeds | friction z R², 5-fold | held-out friction |
|---|---|---|---|
| loud joint | 0–4 | 0.319 / 0.361 / 0.286 / 0.181 / 0.187 (mean **0.267**, sd 0.080) | 0.598 |
| ctrl joint | 0–2 | −0.204 / −0.171 / −0.216 | — |

The held-out readout and the 5-fold readout disagree in level (0.60 vs 0.27
for the same z). Both are reported; the 5-fold one is primary because it
uses all 512 scenes and matches F31a's H1.

## 3. Arms

MLX (canonical), F27b recipe: lr 2e-4, 50 epochs, batch 16, 128 px, 16
views. **Joint arm only: no trajectory head** (the F31a head collapsed on
both backends). Worlds `data/pm_f31fix_loud` (seeds 0–4) and
`data/pm_f31fix_ctrl` (seeds 0–2). `pm_big` is not run: the pilot finds
nothing there to recover.

Per-view latent v_i = relu(proj(GAP(CNN(image_i)))) ∈ ℝ²⁵⁶, unchanged. Only
the step from {v_1 … v_16} to z changes, selected by a new
`ModelConfig.view_pool` (default `"mean"`, so every existing config is
unchanged):

| arm | `view_pool` | z = | new parameters |
|---|---|---|---|
| P0 | `mean` | mean_i v_i (today's encoder, rerun under the new code) | 0 |
| P1 | `stats` | relu(W [mean_i v_i ; std_i v_i]), W: 512 → 256 | 131 k |
| P2 | `attn` | mean_i of one pre-LayerNorm transformer block over the 16 tokens v_i + E·d_i (4 heads, FFN 512, residual), where d_i is camera i's unit direction from `samples.fibonacci_sphere_poses` and E: 3 → 256 is learned | ≈ 0.53 M |

P2 is the principled mixer: a highlight depends on where each camera sits
relative to the light, so each view token carries its camera direction. The
views are deterministic and in a fixed order for every scene, so the
direction is the same information as the view index, stated physically.

24 runs in all (3 arms × 8), about 3 min each on the Mac at the replication's
pace (P2 somewhat more).

## 4. Readouts

From `scripts/f31_measure_mlx.py` (on `claude/f31a-mlx-replication`; it
lands on this branch with the freeze), per checkpoint:

- **friction z R²**: 5-fold ridge from z (all 512 scenes) to normalized
  friction. Primary.
- **held-out friction**: ridge on train-split z, R² on the 20 corner scenes.
  Co-reported.
- **PR**: final latent participation ratio.
- **Descriptive:** density and restitution z R²; held-out behavior gain.

**Diagnostic D1 (where friction is lost):** for every P0 checkpoint,
5-fold ridge from the 16 concatenated per-view latents (4096 dims) to
friction. If D1 is well above P0's z, the information exists per view and
the average discards it. If D1 ≈ z, the per-view CNN loses it first and no
pooling change can recover it. D1 needs a per-view encode helper (added
with the freeze).

## 5. Gates, hypotheses and frozen rules

Every Welch t is two-sample over the listed seeds. Power disclosure: with
P0's friction sd of about 0.080, the smallest difference reaching t = 2.5
at n = 5 per arm is about 0.13. Effects between 0.05 and 0.13 will read
INCONCLUSIVE, and that is the honest reading at this n.

**Gates (read first):**
- G1 health: every run's final PR ≥ 8.
- G2 control: every ctrl run of an arm has friction z R² < 0.10. The ctrl
  pixels carry no material, so a positive is leakage. A G2 failure
  withholds that arm's verdict.
- G3 apparatus (reported, does not withhold): P0's loud mean is within
  ±0.10 of the replication's 0.267. Verdicts compare against the P0 rerun
  either way.

**H1 — stats pooling (P1 vs P0)** and **H2 — attention pooling (P2 vs P0)**,
each graded separately on Δ = mean friction z R²(arm) − mean(P0):
- RISES: Δ ≥ +0.10 and t ≥ 2.5
- NULL: |Δ| < 0.05
- FALLS: Δ ≤ −0.10 and t ≤ −2.5
- otherwise INCONCLUSIVE

Two comparisons share one baseline. At df ≈ 8, t ≥ 2.5 is about p = 0.02
one-sided per test, so about 0.04 family-wise; disclosed, not corrected.

**H3 — mechanism (P2 vs P1)**, same bars, secondary: does direction-aware
mixing beat order-free statistics?

## 6. Predictions (registered at real odds)

- **P1: H2 (attention) RISES 50 %**, NULL or INCONCLUSIVE 45 %, FALLS 5 %.
  The pilot shows cross-view structure worth about +0.3 to +0.45 over the
  averaged view on fixed features. Against it: the per-view CNN's global
  average may already have flattened the highlight, which D1 will show.
- **P2: H1 (stats) RISES 30 %.** The pilot's C (mean + std) gained only
  +0.13 over A on fixed features.
- **P3: G2 passes for every arm, 90 %.**
- **P4 (descriptive):** restitution z R² stays ≥ 0.97 in every arm
  (brightness reads the same from every view; nothing to gain).
- **P5: D1 (per-view concatenation, P0) reads friction ≥ 0.45, 55 %.**

## 7. What each outcome means

- **H2 or H1 RISES:** the view average was a bottleneck. The mean-view
  "pixel ceiling" understated what the photos carry, and the F28/F31a
  legibility arguments built on it need restating. Adopting a pooling
  variant as the default is a separate decision.
- **NULL, D1 high:** the information survives per view but the mixer does
  not learn to use it under this loss. The next question is the training
  signal, not the architecture.
- **NULL, D1 ≈ z:** the per-view CNN loses friction before pooling. The
  next lever is the per-view encoder (resolution, no global average over
  H × W), not the pooling.
- **G2 failure in an arm:** that mixer reads something other than the
  material from the ctrl photos; its loud result is not interpretable.

## 8. What this does not test

- **No Fourier (FNet) arm.** Owner decision, 2026-10-27: no FNet testing on
  this project. The pilot already shows the view order barely matters, so
  an FFT over the view index has no principled content here.
- No trajectory head (F31a collapse), no `pm_big`, no unity measurement.
- The pilot's features are post hoc, chosen after seeing the
  mean-view result; they motivate the arms but are not graded.

## 9. Reproduce

    python scripts/crossview_pilot.py --data data/pm_f31fix_loud data/pm_f31fix_ctrl data/pm_big
