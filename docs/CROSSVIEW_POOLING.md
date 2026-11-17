# Cross-view pooling: does replacing the view average let z hold friction?

**Status: FROZEN preregistration, 2026-11-17.** Draft 1 is dated
2026-10-27 and draft 3 2026-11-14. This commit is the freeze, made on the
owner's go. F-number: assigned by the owner after the freeze, in a commit
that changes nothing else.

The freeze conditions are met:
- **Implementation.** `view_pool` is in all three backends with
  cross-backend tests. The numpy reference and MLX are tested on the Mac.
  Torch parity passes for all five arms on the cloud (PR #52 head
  `f66943a`, cloud-notes `9855e04`). Code changed after that commit
  touches only the readout and verdict scripts, not the backends.
- **Scripts.** `scripts/crossview_measure.py` and
  `scripts/crossview_verdicts.py` are committed with rule tests.
- **No loud-world training.** No pooling-variant model has been trained on
  a loud world. The only pooling-variant runs are the four ctrl-world
  stability runs (§3), and no essence readout was computed from them.

**§6 is hashed.** Its SHA-256, over the bytes from the `## 6.` heading
line up to, not including, the `## 7.` heading line, is

    d4f58988fb900aac788ed889e1aa3e0e96a5a4f981939cea67adf643c10c0e6c

Recompute it with:

    awk '/^## 6\. /{p=1} /^## 7\. /{p=0} p' docs/CROSSVIEW_POOLING.md | shasum -a 256

The cloud session records the freeze commit and this hash, with a
timestamp, in `origin/claude/cloud-notes`. From here on, any change to
§3–§6 (arms, readouts, gates, bars, rules or predictions) is a dated,
disclosed amendment, and none is made after a loud pooling run starts.

**Disclosed at the freeze.** §6 stands as the owner accepted it
(2026-11-17), so it is hashed unchanged. Its P1 rationale cites the
margin over P0-equivalent nested z as +0.15 to +0.17, or 1.5–1.7 times the
0.10 bar. That margin came from the GPU-encoded baseline (0.398). On the
frozen CPU baseline (0.402, §2) it is +0.15 to +0.16, or 1.5–1.6 times the
bar. The odds are unchanged.

*Revision 2026-10-27 (draft 2), from the cloud session's review: matched
no-mixing controls P1′/P2′ become the primary comparisons; D1 gains
nonlinear readouts and numeric thresholds; G2 is named the memorisation
guard and RISES also needs a positive held-out difference; loud seeds 5 →
10.*

*Revision 2026-11-14 (draft 3), owner's go: the experiment moves to the
above-floor worlds `pm_f31fix_{loud,ctrl}_cam`. On the worlds drafts 1–2
used, 7 of the 16 cameras sat below the floor (the camera bug; the fix is
`--camera-layout above`, PR #48). §1–§2 are rewritten with the pilot and
the trained baseline re-measured on the new worlds (baseline at 50 and 150
epochs). §3–§9 take the cloud session's review of the switch (cloud-notes
`8b0fdbb`, `19a367f`, `f342299`). A readout-strength diagnostic (§2, end;
`scripts/z_readout_alpha.py`) adds two co-reported readouts (§4). Bars
and predictions that rested on old-camera numbers were first flagged ⚑.
All are now re-derived. The last, the RISES bar's reachability (§5), was
closed by the owner on 2026-11-17.*

*Owner decisions, 2026-11-14:
- The epoch budget is 50 (§3).
- The focus stays on the pooling arms, P1/P2 against their matched controls, with D1 as the co-diagnostic.
- The primary readout is nested α on a frozen grid; the fixed α = 10⁻³ readout is co-reported, and a verdict that differs between them is READOUT-DEPENDENT (§4, §5).
- The odds are re-registered (§6).
- A ctrl-world stability check runs before the freeze (§3).
- The rank-controlled top-k readout is dropped (§8).
- The 15–65° camera band is kept (§8).*

*Owner decisions, 2026-11-17:
- §6 stands as written.
- The RISES bar stays at Δ ≥ 0.10, and loud seeds go from 10 to 20 per
  arm, so that a NULL means no effect of the bar's size, not too few
  seeds to tell (§5). The reachability flag is closed.
- Readouts encode on MLX's CPU stream, exact float32 (frozen, §4). The G3
  baseline is re-measured on CPU (§2, §5).
- A descriptive co-report reads friction under 10⁻⁴ relative input noise
  (§4).
- The rank account of the fixed-α penalty is withdrawn as a mechanism.
  Rank stays a co-reported description (§4).*

*Correction, 2026-11-14 (§1–§2 rewritten). The draft as pushed through
`c08b173` said in §1 that the camera fix raised "what survives the
average" (A: 0.177 → 0.408) and that the photos carry the same friction
on both layouts. It cited the cloud's pixel-mean numbers 0.089 and 0.227.
Those came from an over-regularized readout, which the cloud retracted in
cloud-notes `f2c5e25`. The Mac's per-view readout (`9b7066e`, `08ed667`,
`98e87b4`) shows that what survives the view average depends on how much
spatial layout each view keeps. A's rise is a property of A's features.
An intermediate reading, that this pointed away from pooling and at the
per-view encoder, was also wrong: the encoder pools each view over
H × W before the views are combined, so the coarse rows are its analogue
(§1).*

## 1. Why this, why now

The encoder reduces each view to one vector, then averages the 16 vectors
into z (`models/mlx_net.py:63-65`: a global average over H × W, a ReLU
projection, then the mean over views; mirrored at `numpy_net.py:126` and
`torch_net.py:95`). Anything carried by how the views *differ* is
discarded before z exists, and so is each view's spatial layout.

Friction reaches the photos only through roughness, rendered as Phong
shininess (`shininess = 1 − roughness`): the tightness of the specular
highlight. A highlight appears only where camera, light and surface line
up, so it is a cross-view signal.

What the average costs, on the above-floor photos (§2):

- **Fixed cross-view statistics** read friction at 0.547–0.565 (D, D′, H).
  The same statistics averaged over views read 0.408 (A). The room over
  the view average on fixed features is about 0.14–0.16. On the old
  cameras it was 0.46 (0.636 vs 0.177).
- **What survives the view average depends on how much spatial layout
  each view keeps** (per-view pixel readout, §2). With views reduced to
  8 × 8 block means, the 16-view average reads friction at 0.005 and the
  16 views side by side at 0.630. At 16 × 16 the average keeps most of it
  (0.597 vs 0.616). The encoder keeps no layout at all: it averages each
  view over H × W first. So it sits on the coarse side, unless its 256
  nonlinear channels carry the highlight in a form that survives spatial
  pooling. D1 (§4) reads exactly that.

Two further facts make pooling the suspect:

- z is already supervised on friction directly: the essence head (weight
  0.3, `mlx_net.loss_fn`) regresses all three essence axes from z. Under
  that direct supervision, mean-pooled z holds friction at 0.402 on the
  above-floor worlds (nested readout; 0.379 at fixed α; 50 epochs, 5
  seeds; CPU-encoded, §2; 0.398 / 0.378 GPU-encoded). That is below the fixed cross-view features and below the 16
  views side by side (0.616, a lower bound). Training to 150 epochs does
  not raise it. The target is not what is missing.
- The project's "pixel ceiling" (0.167, F19 pixel features) is computed on
  the averaged view, on the old cameras. The F28 and F31a legibility
  arguments rest on it.

The alternative explanation is upstream of pooling: the per-view CNN
(stride-2 convolutions, then a global average over H × W) may already have
smeared the highlight before the views are combined. Diagnostic D1 (§4)
separates the two.

This is a question about **appearance** (the F13–F31 line). It is separate
from F35 (`docs/CROSS_INTERACTION.md` on `claude/f35-cross-interaction-design`),
which keeps appearance fixed and reads physics from motion.

## 2. Baselines measured before this draft (disclosed)

**Worlds.** `pm_f31fix_{loud,ctrl}_cam`: the F27b recipe (512 scenes, 16
views, 128 px, seed 1234, boxes; loud coupling α 1, gain 8; ctrl α 0, gain
1) with `--camera-layout above`, generated from `claude/camera-above-floor`
@ `b7a62e2`. The layout puts the 16 cameras at elevations 15.0° → 65.0° in
index order (steps of 3.3°), golden-angle azimuths, radius 2.6 from the
look-at point. Checked on both worlds: the direction of camera i is the
same for all 512 scenes (maximum difference 0) and the stored frame order
is 0–15. `scripts/compare_worlds.py` (`7f9fc18`) confirms each world
matches its old-camera counterpart in every stored field (54,272 values
per world) except camera positions and stored trajectories (the new
worlds were generated without `--keep-trajectory`; the joint arm does not
read them);
roughness ↔ friction r = 0.9922 (loud) and 0.0078 (ctrl), unchanged.

**Cross-view pilot**, post hoc, no training (`scripts/crossview_pilot.py`,
unchanged since `eab1bcb`): nested 5-fold ridge R² (α chosen by inner
4-fold CV), mean of 3 fold seeds (range in brackets), plus ridge on the
train split scored on the 20 held-out (corner) scenes. Old-camera numbers
were reproduced exactly on the Mac before the new ones were read.

Loud world, friction:

| set | dim | old cameras | above-floor | held-out, old → above |
|---|---|---|---|---|
| A mean-view pixels | 198 | 0.177 (0.169–0.186) | **0.408** (0.405–0.410) | 0.321 → **0.752** |
| C mean + std across views | 26 | 0.309 (0.300–0.327) | 0.037 (0.023–0.055) | 0.406 → 0.560 |
| D \|DFT over view index\| | 117 | 0.636 (0.622–0.644) | **0.547** (0.543–0.556) | 0.678 → 0.675 |
| D′ same, views shuffled first | 117 | 0.613 (0.595–0.628) | **0.565** (0.554–0.572) | 0.678 → 0.647 |
| G per-view stats sorted across views | 208 | 0.469 (0.437–0.489) | 0.475 (0.452–0.507) | 0.615 → 0.679 |
| H mean + \|deviation\| quantiles | 65 | 0.491 (0.478–0.501) | **0.554** (0.541–0.567) | 0.662 → 0.668 |

Loud world, the other two axes, old → above:

| set | density | restitution |
|---|---|---|
| A | 0.233 → 0.010 | 0.946 → 0.939 |
| C | 0.553 → −0.020 | 0.975 → 0.967 |
| D | 0.657 → 0.060 | 0.979 → 0.967 |
| D′ | 0.647 → 0.068 | 0.980 → 0.967 |
| G | 0.563 → 0.004 | 0.980 → 0.975 |
| H | 0.656 → 0.079 | 0.980 → 0.964 |

- **Controls.** Every feature set reads friction at or below zero on
  `pm_f31fix_ctrl_cam` (−0.054 to −0.015; held-out −0.074 to +0.013), as
  on `pm_f31fix_ctrl` (−0.08 to −0.01) and on old-camera `pm_big` (−0.06 to
  −0.02). The pilot was not run on an above-floor `pm_big`.
- **The cross-view advantage shrank.** D − A: 0.459 → 0.139. H − A:
  0.314 → 0.146. D′ − A: 0.436 → 0.157.
- **Held-out reverses it.** On the 20 corner scenes of the above-floor
  world the averaged view (A, 0.752) beats every cross-view set
  (0.560–0.679). On the old cameras it was the other way round. Twenty
  scenes; disclosed, not interpreted.
- **C collapses** (0.309 → 0.037) while A rises. Mechanism not measured.
- **D ≈ D′:** shuffling the view order costs nothing (0.547 vs 0.565, |D − D′|
  = 0.018). The Fourier structure is not doing the work; |DFT| acts as a set
  of cross-view energy features. No Fourier arm is preregistered (§8).
- **Density falls on every set** with the above-floor cameras (0.00–0.08).
  The per-view readout below shows the photos still carry some, and that
  the view average loses it. Density numbers from the old worlds do not
  carry over.
- These are fixed hand-made features with a linear readout. They bound
  what the photos contain from below. They are not what a trained encoder
  will learn.

**Per-view pixel readout**, post hoc, no training
(`scripts/perview_readout.py`, `9b7066e`; `--block` `08ed667`;
`--log-density` `98e87b4`). Each view is reduced to g × g × 3 block-mean
pixels. Three readouts: each view alone, the 16-view mean, and the 16
views concatenated. Nested 5-fold ridge on column-standardized features,
grid 10⁻³ … 10⁶, mean of 3 fold seeds. The 16 × 16 predictions were
registered before the run (Mac `runs/postfix/PREDICTIONS_cam.md` Step 8,
hash timestamped in cloud-notes `31cf59a`). 5 of the 8 were correct.
One miss was narrow (old minus above-floor concatenated density 0.291
against a 0.30 bar). Two were anchored on the cloud's since-retracted
pooled numbers. The block-size and log-density rows are unregistered
checks.

Loud world, friction (density in the last two rows):

| | old cameras | above-floor |
|---|---|---|
| 8 × 8 blocks: 16-view mean / concatenated | 0.107 / 0.717 | 0.005 / 0.630 |
| 16 × 16: mean / concatenated | 0.612 / 0.760 | 0.597 / 0.616 |
| 32 × 32: mean / concatenated | 0.696 / 0.747 | 0.579 / 0.629 |
| 16 × 16, best single views | 6 0.626, 8 0.589 | 2 0.572, 0 0.551, 14 0.518 |
| log density, 16 × 16: mean / concatenated | 0.952 / 0.955 | 0.328 / 0.527 |
| log density, best single views | 7 0.506, 10 0.426 | 15 0.596, 14 0.527, 0 0.454 |

- **What the view average keeps depends on resolution.** Concatenated, friction is about
  resolution-independent (0.62–0.76). Averaged, it is nearly gone at 8 × 8
  and mostly kept at 16 × 16 and finer.
- **Density.** The above-floor cameras lose about 40 % of the photos' density
  (concatenated, log scale, 0.955 → 0.53–0.58; best of grid 0.577). Unlike
  friction, the 16 × 16 average also loses it (0.328 vs the best single
  view's 0.596). The best density views on this layout are the highest two
  (views 15 and 14, 65° and 62°) and the lowest (view 0, 15°). On the
  linear axis the levels are lower (concatenated 0.648 → 0.357) but the
  ordering is the same.
- Restitution reads ≥ 0.91 from every single view on both layouts. Ctrl
  reads ≤ −0.004 everywhere.
- The concatenation has p ≫ n (12,288 dims at 16 × 16, 512 scenes), so it
  is a lower bound on what the photos carry. The averaged sets and the
  informative single views pick the grid's lower edge, so they may
  under-read.
- **Replication.** The cloud session's MuJoCo 3.17 re-renders of the same
  materials agree with these photos once readout and target match:
  friction within 0.02, log density within 0.005 (cloud-notes `f2c5e25`,
  `1c761f9`). Its earlier pooled numbers came from a penalty scaled by n
  (λ·n·I, weakest effective α ≈ 0.41) and are retracted.

**Trained baseline**, MLX, mean pooling, F27b recipe (lr 2e-4, batch 16,
default `ModelConfig`), joint arm, trainer at `main` `7bb27ed`, readout
`scripts/f31_measure_mlx.py` @ `e5cbc19` (on `claude/f31a-mlx-replication`).
The above-floor runs were registered before training (predictions T0–T6,
hash recorded in cloud-notes `51d9240`). Constant LR and deterministic
training make each 150-epoch run an exact continuation of the 50-epoch run
with the same seed, so 150 vs 50 is a paired comparison.

Loud joint, seeds 0–4, friction z R²:

| | per seed | mean (sd) | held-out per seed | held-out mean (sd) |
|---|---|---|---|---|
| old cameras, 50 ep (F31a replication) | .319 .361 .286 .181 .187 | 0.267 (0.080) | .624 .646 .510 .659 .553 | 0.598 (0.064) |
| above-floor, 50 ep | .435 .453 .243 .349 .411 | **0.378** (0.085) | .382 .605 .551 .566 .490 | 0.519 (0.087) |
| above-floor, 150 ep | .373 .447 .214 .326 .346 | 0.341 (0.085) | .532 .523 .699 .604 .512 | 0.574 (0.079) |

The per-seed values above use the fixed α = 10⁻³ readout. On the
frozen-grid nested readout, now primary (§4), the above-floor 50-epoch
seeds read .457 .449 .330 .351 .402, mean **0.398** (0.057). Ctrl reads
−0.011 / −0.014 / −0.015 (`runs/xv_cam/readout_alpha_frozen_*.json`).

**Re-measured on CPU encodings (2026-11-17, `crossview_measure` @
`931adb4`, `runs/xv_cam/baseline_cpu/`).** Every MLX number in this
section was GPU-encoded (§4, encode device). The same checkpoints on both
devices, with predictions set in `931adb4` before the run:

| above-floor, 50 ep | nested, mean (sd) | fixed α 10⁻³ | held-out | ctrl nested | ctrl fixed α |
|---|---|---|---|---|---|
| GPU | 0.398 (0.057) | 0.378 (0.085) | 0.519 | −0.011 to −0.015 | −0.09 to −0.22 |
| CPU (frozen) | **0.402** (0.048) | 0.379 (0.084) | 0.512 | −0.012 to −0.018 | −0.09 to −0.11 |

- **Apparatus check.** The GPU reproduces 0.398 / 0.378, which is right.
- **Mean.** The CPU means are within 0.02 of the GPU's, as predicted.
- **Per run, one prediction failed.** Fixed moves by at most 0.007. Nested
  moves by −0.014 to +0.032, and seed 3's +0.032 is over the predicted
  0.02.
- **Ctrl.** Ctrl stays far below the 0.10 guard. Its fixed-α penalty
  shrinks on CPU (seed 1: −0.223 → −0.090), so part of that penalty was
  GPU rounding in near-null directions.
- **Noise co-report.** Under 10⁻⁴ input noise, loud nested reads 0.402,
  within 0.006 of exact on every run, as predicted. This joint z holds
  its friction robustly.

- **Camera effect at 50 epochs:** +0.111, Welch t = 2.14 (below the
  project's 2.5). Held-out moves the other way (−0.079, t = −1.65). Not
  established at n = 5.
- **Budget effect, 150 vs 50:** the 5-fold readout falls in 5/5 seeds
  (paired −0.038, sd 0.026, paired t = −3.26, df 4; Welch −0.70); the
  held-out readout rises in 4/5 (paired +0.055, t = +1.27). The two
  readouts move in opposite directions. Neither shows a gain from training
  three times longer. Over the same 100 epochs the rank of standardized z
  rose in every seed (102–115 → 124–142; see the next block), which raises
  a near-least-squares readout's out-of-fold variance penalty, so the fall
  need not mean less friction in z. That account is not supported either:
  tuning α does not remove the fall (extended-grid nested readout −0.061,
  down in 5/5, t = −2.84), and it weakens (t ≈ −1.6) only under ridges
  strong enough to discard most of the friction. A rank-controlled readout
  (top-k principal directions at fixed k) would separate the two. It is
  not run: with the budget fixed at 50, the question no longer bears on a
  decision (owner, 2026-11-14). Each run is read at its final checkpoint
  only; epoch-to-epoch jitter of the z readout is not measured.
- Registered predictions that failed: 150 epochs would add ≥ 0.05 (it
  lost 0.038); trained z would reach 0.408 at 150 epochs (0.341); at least
  3 of 5 runs would escape (gain ≥ 2) by 150 epochs (0 of 5; gain
  1.06–1.09).
- **Ctrl joint** (seeds 0–2): old 50 ep −0.204 / −0.171 / −0.216; above
  50 ep −0.092 / −0.223 / −0.171; above 150 ep −1.260 / −0.165 / −0.130.
  Held-out at or below +0.046.
- Descriptive, above-floor loud: density z R² −0.06 to +0.02 at 50 epochs
  and −0.13 to −0.03 at 150 (old cameras 0.58–0.64); restitution
  0.975–0.987; participation 47–69 (50) and 50–73 (150).

The held-out readout and the 5-fold readout disagree in level (0.52 vs 0.38
for the same z) and, between budgets, in direction. Both are reported. The
5-fold nested readout is primary because it uses all 512 scenes; the
fixed-α 5-fold readout, which matches F31a's H1, is co-reported.

**How the fixed-α readout depends on its ridge strength** (diagnostic, not
registered; `scripts/z_readout_alpha.py`, `fdb2147`, same z and folds as
the fixed readout; GPU-encoded, so the joint z's numbers below are within
about 0.03 of the CPU's, by the re-measure above). The fixed readout, primary until this revision and
F31a's, is a ridge at α = 10⁻³ on column-standardized z, close to least
squares. This diagnostic is why the nested readout is now primary (§4).

| above-floor, mean over seeds | α 10⁻³ (fixed) | α 0.1 | α 1 | α 10 | α 100 | nested α, pilot grid | std. rank of z |
|---|---|---|---|---|---|---|---|
| loud, 50 ep | 0.378 | 0.242 | 0.117 | 0.025 | −0.005 | 0.378 | 102–115 |
| loud, 150 ep | 0.341 | 0.200 | 0.087 | 0.008 | −0.012 | 0.341 | 124–142 |
| ctrl, 50 ep | −0.162 | −0.100 | −0.054 | −0.023 | −0.011 | −0.013 | 95–96 |
| ctrl, 150 ep | −0.518 | −0.065 | −0.039 | −0.020 | −0.009 | −0.009 | 75–95 |

- **Friction is linearly readable from z only along low-variance
  directions.** At α = 10, which halves any direction of standardized z
  holding under about 2.4 % of an average dimension's variance, the loud
  readout falls from 0.378 to 0.025.
- **The fixed α is near, not at, the optimum.** On the pilot's grid the
  nested readout picks 10⁻³, the grid's lowest value, on every loud run.
  With the grid extended to 0 (the minimum-norm least-squares fit) and
  10⁻⁵ … 10 (`fac9c7d`), it picks 10⁻⁴ or 10⁻⁵ at 50 epochs and mostly
  10⁻³ at 150. The unregularized fit is worse everywhere (0.258 at 50
  epochs; it blows up on two 150-epoch runs). Nested friction R², loud:
  0.398 (sd 0.057) at 50 epochs, 0.337 (sd 0.085) at 150, against the
  fixed α's 0.378 and 0.341. The frozen grid (10⁻⁵ … 10², §4) gives the
  same 0.398 at 50 epochs.
- **On ctrl the fixed α is what makes the readout negative.** Nested α
  reads ctrl at −0.006 to −0.015, i.e. no friction, on every run. The
  fixed-α values (−0.09 to −0.22, apart from the outlier below) are its
  variance penalty.
- **The ctrl seed 0, 150-epoch value (−1.260) is a readout outlier, not a
  health failure.** One fold reads −4.10, and its three worst scenes carry
  78 % of that fold's squared error; at α ≥ 0.1 the run reads −0.04 to
  −0.006, and its final PR is 84.8 (G1 needs ≥ 8). Standardized z has 120
  active dims but rank 78 there, so near-collinear directions get almost
  no shrinkage.

## 3. Arms

MLX (canonical), F27b recipe: lr 2e-4, batch 16, 128 px, 16 views, **50
epochs**. **Joint arm only: no trajectory head** (the F31a head collapsed
on both backends). Worlds `data/pm_f31fix_loud_cam` (seeds 0–19) and
`data/pm_f31fix_ctrl_cam` (seeds 0–2). `pm_big` is not run: the
old-camera pilot found nothing there to recover, and the above-floor
pilot was not run on `pm_big`.

**Epoch budget: 50 (owner's decision, 2026-11-14).** Measured (§2): 150
epochs lowers the friction readout, fixed α and nested, in 5 of 5 seeds (whether through z's
higher rank or less friction is unresolved) and raises held-out in 4 of 5,
both modestly, at three times the cost (about 3 min per 50-epoch run,
about 9 min per 150-epoch run on the Mac; at the 10 seeds then planned,
65 runs ≈ 3.5 h vs ≈ 10 h; at 20 seeds, 115 runs ≈ 6 h vs ≈ 17 h; P2 and
P2′ somewhat more). Not chosen: 150, whose case was a separate,
unpublished observation that on `pm_big` the held-out behavior gain for
density kept rising to epoch 150; that readout is not primary here.

Per-view latent v_i = relu(proj(GAP(CNN(image_i)))) ∈ ℝ²⁵⁶, unchanged. Only
the step from {v_1 … v_16} to z changes, selected by a new
`ModelConfig.view_pool` (default `"mean"`, so every existing config is
unchanged):

| arm | `view_pool` | z = | new parameters |
|---|---|---|---|
| P0 | `mean` | mean_i v_i (today's encoder, rerun under the new code) | 0 |
| P1 | `stats` | relu(W [mean_i v_i ; std_i v_i]), W: 512 → 256 | 131 k |
| P1′ | `stats_ctrl` | relu(W [mean_i v_i ; mean_i v_i]): P1's layer and ReLU, no cross-view information | 131 k |
| P2 | `attn` | mean_i of one pre-LayerNorm transformer block over the 16 tokens v_i + E·d_i (4 heads, ReLU FFN 512, residual), where d_i is camera i's unit direction and E: 3 → 256 is learned | ≈ 0.53 M |
| P2′ | `attn_ctrl` | P2's block with a diagonal (self-only) attention mask: same parameters and depth, no information passes between views | ≈ 0.53 M |

**Camera direction d_i** = (position − look_at) / ‖position − look_at‖,
read from each world's stored frames (`sample.json`,
`appearance.frames[i]`), never from a pose function: on the above-floor
worlds `samples.fibonacci_sphere_poses` gives the old, partly below-floor
directions. The implementation adds a test that d_i matches the stored
camera positions on a `_cam` world. The direction is the same for every
scene and the frame order is fixed (§2), so d_i carries the same
information as the view index, stated physically.

P2 is the principled mixer: a highlight depends on where each camera sits
relative to the light, so each view token carries its camera direction.

**Implementation** (`ModelConfig.view_pool`; `tests/test_view_pool.py`):
- std_i v_i = √(var_i v_i + 10⁻⁵), population variance. Dead ReLU units
  are identical across views, where a plain square root has an infinite
  gradient.
- The attn block uses LayerNorm ε 10⁻⁵ and scores scaled by 1/√64.
  `attn_ctrl` adds −10⁹ to every cross-view score, so the softmax puts all
  weight on self. Its query and key layers then get no gradient; the
  parameters and depth still match P2.
- The directions come from `PseudoMarbleDataset.camera_directions`,
  checked equal across every scene. They are stored in the run's config,
  not in the weights.
- The pooling layer is built after the heads, so at a given seed all five
  arms start from the same CNN and head weights.
- The numpy forward is the reference, and MLX matches it to 10⁻⁵ on its CPU
  stream. On the Metal GPU a float32 matmul carries about 10⁻³ relative
  error on this Mac (a plain product shows it). Every arm and every earlier
  MLX run shares it in training. Readouts therefore encode on the CPU
  (§4, encode device).
- The torch parity tests run where torch is installed; the Mac's project
  venv has none.
- Every arm trains on a fixed random batch: the loss falls, and the
  gradients reach the pooling layer.

**Matched controls.** P1 and P2 add capacity (a layer, a ReLU, depth), not
only mixing. P1′ and P2′ keep the added capacity and remove the mixing, so
the primary hypotheses (§5) compare each mixer with its own control and
test the mixing itself. Comparisons with P0 are reported as well.

115 runs in all: 5 arms × (20 loud + 3 ctrl). Ctrl stays at 3 seeds.
G2 requires every ctrl run to pass, so more ctrl runs would add chances
of a false failure without adding power to H1 or H2.

**Stability check, before the freeze (owner, 2026-11-14).** The mixer arms
have never been trained. Once `view_pool` is implemented, one run each of
P1, P1′, P2 and P2′ trains on `data/pm_f31fix_ctrl_cam` at seed 100. Seed
100 is not one of the experiment's seeds, and these runs are not among the
115. On ctrl the photos carry no material, so these runs can show nothing
about the hypotheses. From them, only these are read:
- the training-loss curve;
- the per-epoch latent PR;
- any non-finite value.

No essence, behavior or held-out readout is computed from them.

An arm fails the check if it diverges, goes non-finite, or ends with
PR < 8. A failed arm's recipe is revised before the freeze, and the
revision is disclosed here.

**Result (2026-11-17, code `77e6af3`): all four pass, so no recipe is
revised.** The values below are read from each run's own `metrics.json`
(`runs/xv_cam/stability/`); nothing else was computed.

| arm | loss, epoch 0 → 49 | highest loss after epoch 5 | PR, min–max | final PR | non-finite |
|---|---|---|---|---|---|
| P1 `stats` | 0.261 → 0.153 | 0.155 | 40.9–44.8 | 44.8 | none |
| P1′ `stats_ctrl` | 0.258 → 0.153 | 0.155 | 46.5–48.7 | 48.7 | none |
| P2 `attn` | 0.221 → 0.153 | 0.155 | 41.4–66.3 | 66.3 | none |
| P2′ `attn_ctrl` | 0.220 → 0.153 | 0.155 | 13.7–55.4 | 55.4 | none |

- P2′'s PR dipped to 13.7 during training before recovering to 55.4. That
  is above the bar of 8, but no other arm dipped that far. It is recorded
  here so it is no surprise later, and nothing is changed for it (cloud
  review, `2291c16`).
- Each run took about 3 min on the Mac (P2 and P2′ about 3.1 min).
- The 115 runs therefore come to about 6 h.

## 4. Readouts

**Encode device (owner, 2026-11-17).** Every checkpoint is encoded on
MLX's CPU stream, in exact float32. That is `crossview_measure`'s frozen
default, and `crossview_verdicts` refuses a report encoded elsewhere.
Training stays on the Metal GPU.

The GPU's float32 matmul rounds its inputs, with relative error about
7 × 10⁻⁴. At the projection this erased much of the F31a trajectory arm's
friction in z:
- trajectory arm, nested readout: 0.405 on CPU, 0.233 on GPU (5 seeds);
- joint arm: moved by at most 0.012.

The source is Mac `runs/postfix/RESULTS.md` Step 10 and PR #52. Torch
reproduces the site (cloud-notes `3b53754`). Every earlier MLX readout in
this document was GPU-encoded. The baseline that G3 compares against is
re-measured on CPU (§2, §5).

From `scripts/crossview_measure.py`, per checkpoint. It reuses
`f31_measure`'s fixed-α and held-out readouts and `z_readout_alpha`'s
nested readout and rank, so `f31_measure_mlx.py` (branch
`claude/f31a-mlx-replication`, no PR; GPU-encoded) does not
need to land here:

- **friction z R², nested (primary)**: ridge from column-standardized z
  (all 512 scenes) to normalized friction.
  - Outer folds: 5, the same as the fixed readout's.
  - α picked per outer fold by inner 4-fold CV on the frozen grid 10⁻⁵, 10⁻⁴, …, 10² (8 values).
  - Code: `scripts/z_readout_alpha.nested_grid`.
- **friction z R², fixed α = 10⁻³**: the same folds at F31a's α
  (`probe_appearance.kfold_r2`). Co-reported.
- **rank of standardized z** (`z_readout_alpha.std_rank`). Co-reported.
- **held-out friction**: ridge (fixed α = 10⁻³) on train-split z, R² on the
  20 corner scenes. Co-reported, and a condition for RISES (§5).
- **PR**: final latent participation ratio.
- **Descriptive:** density and restitution z R²; held-out behavior gain.
- **Descriptive, noise co-report:** friction nested and fixed on z, encoded
  with relative noise 10⁻⁴ on the input of every Linear and Conv2d
  (`crossview_measure.input_noise`, seed 0).
  - It shows whether an arm's friction survives a reader of that precision.
    It grades nothing.
  - On one run of torch F31a's trajectory arm, noise of this size drops
    friction from 0.478 to 0.138 (cloud-notes `3b53754`).

**Why nested is primary (owner, 2026-11-14).** The fixed α is close to
least squares (§2). How much it under-reads depends on where in z's
spectrum friction sits. P1, P1′, P2 and P2′ change what produces z, so a
fixed α can put a readout penalty into Δ.

This has been seen on both backends. The table gives traj − joint on exact
float32 encodings of F31a's joint and trajectory arms, 5 + 5 seeds:

| | fixed α 10⁻³ | nested |
|---|---|---|
| torch (cloud-notes `343ecea`) | +0.133 (t 2.85) | +0.063 (t 1.86) |
| MLX, CPU-encoded (Mac Step 10, PR #52) | +0.100 (t 1.94) | +0.148 (t 3.24) |

- The two readouts differ in Δ by 0.05–0.07, in opposite directions on the
  two backends.
- On torch the fixed α under-reads joint by 0.111 and traj by 0.041.

*Correction, 2026-11-17.* This paragraph said the penalty tracks the rank
of standardized z. It cited the Mac's MLX F31a replication: ranks 89 vs
136, H1 NULL on both readouts, and +0.108 at α 0.1. Both claims are
withdrawn:
- **The MLX numbers were GPU-encoded.** The GPU erases much of the
  trajectory arm's friction (see "Encode device" above).
- **The rank account fails in direction on torch.** `std_rank` gives joint
  the lower rank (113 vs 121, t 1.64), yet joint is the arm under-read
  more (cloud-notes `f87a210`).
- **A post-hoc band split locates the gain.** It sits in weak but real
  directions of standardized z, with sd below about 1.5 × 10⁻³
  (cloud-notes `cccb82d`, `1f0a43b`).

Rank is co-reported as a description, not as the mechanism.

The nested readout adapts α to each z. It reduces the penalty but does
not remove it, hence the readout-dependent rule (§5).

On the above-floor baseline, CPU-encoded, the nested readout reads 0.402
(sd 0.048) and the fixed α 0.379 (0.084). GPU-encoded, the values were
0.398 and 0.378. Its picks on loud runs are 10⁻⁵ to 10⁻³ on both devices.
- **Lower edge.** A pick at 10⁻⁵ is bracketed, not an under-read. The
  excluded α = 0 (the minimum-norm fit) reads below 10⁻⁵ on every
  50-epoch baseline run (0.168–0.324 vs 0.333–0.450).
- **Upper edge.** Ctrl runs pick 10 or 10² and read about −0.01. A pick at
  10² does not threaten G2: as α grows the prediction tends to the
  training mean, whose out-of-fold R² sits just below 0, so a wider grid
  cannot carry a ctrl run toward the 0.10 guard.

A rank-controlled top-k readout (cloud-notes `93024de`) is not added (§8).

**Density is compared only within the above-floor runs.** Its legibility
on fixed features fell from 0.66 to 0.06 with the camera change alone, and
P0-equivalent z reads it at about zero (§2). A density drop against any
old-world number is expected from the cameras and is not an arm effect.

**Diagnostic D1 (where friction is lost):** for every P0 checkpoint, the
per-view latents v_1 … v_16 are read out by the primary readout's nested
5-fold ridge on the same frozen grid (`z_readout_alpha.nested_grid`,
computed in dual form by `perview_readout.nested`, which a test shows
equal; the pilot's grid stops at 10⁻³ and under-reads low-variance
friction, §2),
through five constructions: the plain concatenation
(4096 dims, linear in v), and the pilot's nonlinear cross-view features C,
D, G and H (§2) computed on v instead of on hand-made image statistics.
A linear readout of the concatenation cannot compute a spread or a
quantile, so the nonlinear constructions are what can show the information
is present per view. (D here is a fixed readout feature, not an FNet model.)
**D1** = the best of the five, mean over P0 seeds; **z₀** = P0's mean
friction z R² on the primary (nested) readout.
- D1 high: D1 ≥ z₀ + 0.15. The information survives per view; the
  average discards it.
- D1 ≈ z: D1 ≤ z₀ + 0.05. The per-view CNN has already lost it.
- In between: unresolved.

D1 reads v_1 … v_16 through `encode_views`, which all three backends now
provide.

## 5. Gates, hypotheses and frozen rules

Every Welch t is two-sample over the listed seeds. Every bar is read on
the primary (nested) readout unless stated.

**Power disclosure,** recomputed on the above-floor P0-equivalent runs.
Friction sd is 0.085 on the fixed-α readout (MLX, 5 seeds, at both 50 and
150 epochs) and 0.057 on the nested one (50 epochs), GPU-encoded. On the
frozen CPU encoding the 50-epoch sds are 0.084 and 0.048, so 0.085 stays
the conservative figure.

The power figures use the larger sd, 0.085, because the variance ratio
(2.2 on 4 and 4 df) is not established on 5 seeds. At that sd the smallest
difference reaching t = 2.5 is about 0.134 at n = 5, 0.095 at n = 10 and
0.067 at n = 20 per arm (0.045 at n = 20 at sd 0.057). At 20 loud seeds
the t condition binds below the 0.10 bar, so RISES turns on Δ ≥ 0.10
itself.

**Why 20 loud seeds (owner, 2026-11-17).** NULL is graded on the point
estimate (|Δ| < 0.05). The one-sided 95 % upper bound on Δ for any NULL
is at most 0.05 + t₀.₉₅·SE, at sd 0.085:

| loud seeds per arm | NULL's largest upper bound | excludes Δ = 0.10 |
|---|---|---|
| 10 | 0.116 | no |
| 15 | 0.103 | no |
| 17 | 0.099 | yes (the minimum) |
| 20 | 0.095 | yes |

So at n = 20 every NULL also excludes an effect of the bar's size. That
holds while the realised sd stays ≤ 0.094. `crossview_verdicts.py` reports
the realised bound with every NULL, so a wider spread in the mixers shows.
The realised bound is |Δ| + t₀.₉₅·SE, with the Welch SE and df on the
primary readout: the far end of the 90 % interval, either sign. Each
arm's realised sd is reported beside it, so a reader can see why a bound
moved.

**If a realised bound reaches 0.10** (the arms spread wider than
assumed), the verdict stays NULL and is reported with its bound, labelled
**NULL, bound above bar**. It does not exclude an effect of the bar's size.
No seeds are added after the runs: a seed count chosen after seeing the
spread would make the design depend on the data. (Cloud review, cloud-notes
`e06f63f`.) Other consequences, at sd 0.085:
- A true effect of 0.15 reaches Δ ≥ 0.10 with probability about 0.97
  (0.91 at n = 10). Δ_h > 0 is a further condition.
- A true effect of exactly 0.10 still reaches the bar only about half the
  time, since the bar is on the point estimate.
- Effects between 0.05 and 0.10 read INCONCLUSIVE.

The figures assume P0's spread carries over to the mixers and their
matched controls; that is not measured.

**Gates (read first). A G1 or G2 failure withholds every comparison that
includes the failing arm.** For G1 this rule is new in this draft
(2026-11-17), adopted from F31a ("any failure withholds the dependent
verdicts") before the freeze; drafts 1–3 gave G1 no stated consequence.
- G1 health: every run's final PR ≥ 8 (a non-finite PR fails).
- G2 control, **the memorisation guard**: every ctrl run of an arm has
  friction z R² < 0.10 (nested; on the above-floor baseline ctrl reads
  −0.012 to −0.018 CPU-encoded, against −0.09 to −0.11 at the fixed α,
  whose negative values are its variance penalty, §2). The essence head regresses friction from z on the
  training scenes, and the 5-fold readout covers those same scenes, so a
  higher-capacity mixer could raise the readout by fitting that target
  rather than by reading highlights. On ctrl the pixels carry no material,
  so the same fitting would show there. A G2 failure withholds that arm's
  verdict.
- G3 apparatus (reported, does not withhold): P0's loud mean is within
  ±0.10 of the above-floor 50-epoch baseline, **0.402** nested (0.379 at
  the fixed α, co-reported). Both are re-measured on the frozen CPU
  encoding (§2, 2026-11-17); GPU-encoded they were 0.398 / 0.378.
  Verdicts compare against the P0 rerun either way.

**H1 — stats mixing (P1 vs P1′)** and **H2 — attention mixing (P2 vs
P2′)**, primary, each graded on Δ = mean friction z R²(mixer) − mean(its
matched control), with Δ_h the same difference on the held-out readout:
- RISES: Δ ≥ +0.10, t ≥ 2.5, **and Δ_h > 0** (the 20 corner scenes are
  never trained on, so memorisation cannot raise them; F31a Amendment 2's
  co-report, made a condition)
- NULL: |Δ| < 0.05
- FALLS: Δ ≤ −0.10 and t ≤ −2.5
- otherwise INCONCLUSIVE

**Readout-dependent rule (owner, 2026-11-14).** Each comparison is also
graded, by the same bars, on the fixed-α readout. If either readout grades
RISES or FALLS and the other does not grade the same, the verdict is
**READOUT-DEPENDENT**: both gradings are reported and no RISES or FALLS is
claimed. For example, nested INCONCLUSIVE with fixed-α RISES is
READOUT-DEPENDENT, not INCONCLUSIVE. Only when both grade NULL or
INCONCLUSIVE does the primary's grade stand, with both reported. The two Δ's and the two arms' mean standardized
ranks are reported with every verdict.

**Reachability of RISES: closed (owner, 2026-11-17); bar kept at 0.10,
seeds added.** The bar was set when the best fixed cross-view features
exceeded mean-pooled z by 0.37 (0.636 vs 0.267). On the above-floor worlds
the margin is 0.15–0.16 over nested z (D 0.547, H 0.554, D′ 0.565 vs
0.402 CPU-encoded; 0.17–0.19 over the fixed α's 0.379). RISES needs a
mixer to recover 61–69 % of that margin over its control. At n = 10 the detectable
difference (0.095 at the conservative sd) sat close to the 0.10 bar; at
n = 20 it is 0.067, below it. Separately, on fixed features the held-out
sign favours the average on these worlds (§2), so Δ_h > 0 is not assured
even when the 5-fold readout rises.

Two primary comparisons. At df ≈ 38, t ≥ 2.5 is about p = 0.008 one-sided
per test, so about 0.017 family-wise; disclosed, not corrected.

**Reported against P0, secondary:** P1, P1′, P2, P2′ each vs P0 (is any
arm better than today's encoder, whatever the reason).

**H3 — mechanism (P2 vs P1)**, secondary: does direction-aware mixing
beat order-free statistics?

The secondary comparisons use the same bars (Δ_h > 0 included), the same
readout-dependent rule and the same NULL bound as H1 and H2. Every arm is
scored on the same 20 held-out scenes, so Δ_h is correlated across
comparisons. When several comparisons pass or fail the Δ_h condition
together, that is reported once, as shared, not as independent
confirmations.
`scripts/crossview_verdicts.py` applies every rule in this section, and
`tests/test_crossview_verdicts.py` tests each one on synthetic reports.

## 6. Predictions (registered at real odds)

Re-registered for the above-floor worlds (owner, 2026-11-14). They replace
draft 2's (P1 45 %, P2 30 %), whose rationales rested on old-camera
numbers. They are hashed in the freeze commit.

- **P1: H2 (attention vs its matched control) RISES 35 %**; NULL or
  INCONCLUSIVE 55 %, READOUT-DEPENDENT 5 %, FALLS 5 %.
  - For: on the above-floor photos, fixed cross-view statistics beat the
    averaged set by +0.14 to +0.16 (+0.15 to +0.17 over P0-equivalent
    nested z).
  - For: on spatially coarse pixel features the view average loses nearly
    all friction while the 16 views side by side keep it (8 × 8: 0.005 vs
    0.630). The encoder pools each view over H × W first, which is the
    coarse case (§1).
  - Against: that room is only 1.5–1.7 times the 0.10 bar, so the mixer must
    realise most of it over a control with the same capacity.
  - Against: on fixed features the held-out sign favours the average.
  - Against: the per-view CNN may already have lost the highlight (D1).
- **P2: H1 (stats vs its matched control) RISES 20 %**; NULL or
  INCONCLUSIVE 70 %, READOUT-DEPENDENT 5 %, FALLS 5 %.
  - For: the spread of v_i across views is cross-view information the mean
    lacks.
  - Against: on fixed features the same construction (C, mean plus spread)
    reads 0.037, far below the averaged set A (0.408). The mechanism is not
    measured, but the pilot gives a stats mixer no support.
- **P3: G2 passes for every arm, 90 %.** On the nested readout the
  P0-equivalent ctrl runs read about −0.01; the risk is a higher-capacity
  mixer fitting the essence target.
- **P4 (descriptive):** restitution z R² stays ≥ 0.97 in every arm
  (brightness reads the same from every view; nothing to gain).
  Above-floor P0-equivalent z reads 0.975–0.987; fixed features
  0.939–0.975.
- **P5: D1 reads "high" (≥ z₀ + 0.15), 55 %.** Kept from draft 2.
  Thresholds are relative to z₀, now both on the nested readout.

## 7. What each outcome means

- **H2 or H1 RISES:** the view average was a bottleneck *on the
  above-floor cameras*. The verdict is relative to the mixer's matched
  control and to P0 on the same worlds, never to an old-camera number.
  The F28/F31a legibility arguments, built on the old-camera averaged
  view, need restating either way. Adopting a pooling variant as the
  default is a separate decision.
- **READOUT-DEPENDENT:** whether the mixer helps depends on how z is read.
  Typically that is because the arms' z hold friction in different parts
  of their spectrum; rank alone does not account for it (§4). Not
  established either way. The ranks, both Δ's and the noise co-report
  are the finding.
- **NULL, D1 high:** the information survives per view but the mixer does
  not learn to use it under this loss. The next question is the training
  signal, not the architecture.
- **NULL, D1 ≈ z:** the per-view CNN loses friction before pooling, as far
  as these five readouts can see. The next lever is the per-view encoder
  (resolution, no global average over H × W), not the pooling.
- **Mixer and its matched control both beat P0, with H1/H2 NULL:** the gain
  is capacity, not cross-view information.
- **G2 failure in an arm:** that mixer reads something other than the
  material from the ctrl photos; its loud result is not interpretable.

## 8. What this does not test

- **No Fourier (FNet) arm.** Owner decision, 2026-10-27: no FNet testing on
  this project. The pilot shows the view order barely matters on both
  camera layouts (|D − D′| = 0.023 old, 0.018 above-floor), so an FFT over
  the view index has no principled content here.
- No trajectory head (F31a collapse), no `pm_big` (old-camera pilot found
  nothing; above-floor pilot not run), no unity measurement.
- **Density.** The 15–65° band costs about 40 % of the photos' density
  (§2). The owner keeps the band for this preregistration (2026-11-14). It
  makes no density claim. Cameras chosen for density belong in a separate
  world design.
- **No rank-controlled (top-k) readout.** Dropped (owner, 2026-11-14).
  Its main question, whether 150 epochs scored lower through rank or
  through less friction, stopped bearing on a decision once the budget
  was fixed at 50. The rank report and the readout-dependent rule (§4–§5)
  cover the arms.
- The pilot's features and the per-view pixel readout are post hoc. They
  motivate the arms but are not graded.
- The cloud's MuJoCo 3.17 re-renders agree with the Mac's photos (§2) but
  carry no weight in any bar.

## 9. Reproduce

Worlds (generated from `claude/camera-above-floor` @ `b7a62e2`; the layout is on `main` since PR #48):

    for spec in "loud 1 8" "ctrl 0 1"; do set -- $spec
      python -m pseudomarble.data.generate_mujoco --output data/pm_f31fix_$1_cam \
        --num-scenes 512 --views 16 --resolution 128 --seed 1234 --shapes box \
        --coupling-alpha $2 --coupling-gain $3 --camera-layout above
    done

Gate (`scripts/compare_worlds.py`, commit `7f9fc18`, on `main` since PR #48
merged):

    python scripts/compare_worlds.py data/pm_f31fix_loud data/pm_f31fix_loud_cam
    python scripts/compare_worlds.py data/pm_f31fix_ctrl data/pm_f31fix_ctrl_cam

Pilot:

    python scripts/crossview_pilot.py --data data/pm_f31fix_loud_cam data/pm_f31fix_ctrl_cam
    python scripts/crossview_pilot.py --data data/pm_f31fix_loud data/pm_f31fix_ctrl data/pm_big

Trained baseline (trainer at `main` `7bb27ed`; loud seeds 0–4, ctrl 0–2;
the frozen budget is 50; §2's 150-epoch comparison used `--epochs 150`):

    python -m pseudomarble.models.train --data data/pm_f31fix_loud_cam \
      --epochs 50 --lr 2e-4 --batch-size 16 --seed $S --out runs/xv_cam/loud_joint_e50_s$S
    python scripts/f31_measure_mlx.py --data data/pm_f31fix_loud_cam \
      --runs runs/xv_cam/loud_joint_e50_s* --out runs/xv_cam/loud_joint_e50.json

Per-view pixel readout (§2):

    python scripts/perview_readout.py --data data/pm_f31fix_loud data/pm_f31fix_loud_cam \
      data/pm_f31fix_ctrl data/pm_f31fix_ctrl_cam --out runs/postfix/perview.json
    # block grid: add --block 8 or --block 32; density on a log scale: add --log-density

Readout-strength diagnostic (§2; needs MLX):

    python scripts/z_readout_alpha.py --data data/pm_f31fix_loud_cam \
      --runs runs/xv_cam/loud_joint_e50_s* runs/xv_cam/loud_joint_e150_s* \
      --out runs/xv_cam/readout_alpha_loud.json
    # extended grid (fac9c7d): add --alphas 0 1e-5 1e-4 1e-3 1e-2 1e-1 1 10
    # frozen primary grid (§4): --alphas 1e-5 1e-4 1e-3 1e-2 1e-1 1 10 100

Baseline on the frozen CPU encoding (§2). Add `--device gpu` for the
GPU column:

    python scripts/crossview_measure.py --data data/pm_f31fix_loud_cam \
      --runs runs/xv_cam/loud_joint_e50_s{0..4} --out runs/xv_cam/baseline_cpu/loud_cpu.json
    python scripts/crossview_measure.py --data data/pm_f31fix_ctrl_cam \
      --runs runs/xv_cam/ctrl_joint_e50_s{0..2} --out runs/xv_cam/baseline_cpu/ctrl_cpu.json

Stability check (§3; ctrl world, seed 100, no essence readout):

    for arm in stats stats_ctrl attn attn_ctrl; do
      python -m pseudomarble.models.train --data data/pm_f31fix_ctrl_cam \
        --epochs 50 --lr 2e-4 --batch-size 16 --seed 100 --view-pool $arm \
        --out runs/xv_cam/stability/${arm}_s100
    done

The experiment (after the freeze). Arms P0 P1 P1c P2 P2c are view_pool mean,
stats, stats_ctrl, attn and attn_ctrl. Loud seeds 0–19, ctrl seeds 0–2.
`crossview_measure` encodes on the CPU (frozen default) and adds the 10⁻⁴
noise co-report:

    python -m pseudomarble.models.train --data data/pm_f31fix_${W}_cam \
      --epochs 50 --lr 2e-4 --batch-size 16 --seed $S --view-pool $POOL \
      --out runs/xv/${ARM}_${W}_s$S
    python scripts/crossview_measure.py --data data/pm_f31fix_${W}_cam \
      --runs runs/xv/${ARM}_${W}_s* --out runs/xv/${ARM}_${W}.json   # add --d1 for P0 loud
    python scripts/crossview_verdicts.py --root runs/xv
