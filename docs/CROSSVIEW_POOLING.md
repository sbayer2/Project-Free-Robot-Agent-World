# Cross-view pooling: does replacing the view average let z hold friction?

**Status: DRAFT preregistration, 2026-10-27; draft 3, 2026-11-14. Not
frozen.** It freezes in the commit that adds the implementation (all three
backends, with cross-backend tests, per CLAUDE.md) and
`scripts/crossview_verdicts.py`, before any pooling-variant model is
trained. Until then every number, arm and bar here can change. F-number:
unassigned (the owner assigns it at the freeze).

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
and predictions that rest on old-camera numbers
are **flagged, not re-derived**: they are marked ⚑ and must be settled
before the freeze. Open owner decisions are marked OPEN.*

## 1. Why this, why now

The encoder averages its 16 per-view latents into z (`models/mlx_net.py`,
`Encoder.__call__`, last line; mirrored at `numpy_net.py:126` and
`torch_net.py:95`). Anything carried by how the views *differ* is discarded
before z exists.

Friction reaches the photos only through roughness, rendered as Phong
shininess (`shininess = 1 − roughness`): the tightness of the specular
highlight. A highlight appears only where camera, light and surface line
up, so it is a cross-view signal.

What the average costs, on the above-floor photos (§2):

- Fixed cross-view image statistics read friction at 0.547–0.565 (D, D′,
  H); the same statistics averaged over views read 0.408 (A). The room
  over the view average on fixed features is about 0.14–0.16. On the old
  cameras it was 0.46 (0.636 vs 0.177).
- The camera fix raised **what survives the average** (A: 0.177 → 0.408).
  It did not raise what the photos carry. An unreproduced cloud diagnostic
  on MuJoCo 3.17 re-renders of the same 512 materials (cloud-notes
  `19a367f`) reads friction from concatenated per-view pixels at 0.649
  (old layout) and 0.616 (above-floor), against 0.089 and 0.227 from the
  pixel mean. The gap the mixer could close has narrowed from the mean's
  side, not disappeared.

Two further facts make pooling the suspect:

- z is already supervised on friction directly: the essence head (weight
  0.3, `mlx_net.loss_fn`) regresses all three essence axes from z. Under
  that direct supervision, mean-pooled z holds friction at 0.378 on the
  above-floor worlds (50 epochs, 5 seeds), close to the averaged-view
  pixel readout (0.408) and below the fixed cross-view features. Training
  to 150 epochs does not raise it (0.341). The target is not what is
  missing.
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
- **Density is camera-limited on the above-floor worlds**, not only pooled
  away: every set falls to 0.00–0.08. Unreproduced cloud diagnostics on
  MuJoCo 3.17 re-renders put photo-level density content (concatenated
  per-view pixels) at 0.953 old vs 0.311 above-floor (a lower bound;
  `19a367f`), and find density visible mainly at low elevations
  (`3e51da7`, preliminary). Density numbers from the old worlds do not
  carry over.
- These are fixed hand-made features with a linear readout. They bound
  what the photos contain from below. They are not what a trained encoder
  will learn.

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
  need not mean less friction in z. That account fits the sign and rough
  size; it is not measured. Each run is read at its final checkpoint only;
  epoch-to-epoch jitter of the z readout is not measured.
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
for the same z) and, between budgets, in direction. Both are reported; the
5-fold one is primary because it uses all 512 scenes and matches F31a's H1.

**How the primary readout depends on its ridge strength** (diagnostic, not
registered; `scripts/z_readout_alpha.py`, `fdb2147`, same z and folds as
the primary readout). The primary readout is a ridge at a fixed α = 10⁻³
on column-standardized z, close to least squares.

| above-floor, mean over seeds | α 10⁻³ (primary) | α 0.1 | α 1 | α 10 | α 100 | nested α | std. rank of z |
|---|---|---|---|---|---|---|---|
| loud, 50 ep | 0.378 | 0.242 | 0.117 | 0.025 | −0.005 | 0.378 | 102–115 |
| loud, 150 ep | 0.341 | 0.200 | 0.087 | 0.008 | −0.012 | 0.341 | 124–142 |
| ctrl, 50 ep | −0.162 | −0.100 | −0.054 | −0.023 | −0.011 | −0.013 | 95–96 |
| ctrl, 150 ep | −0.518 | −0.065 | −0.039 | −0.020 | −0.009 | −0.009 | 75–95 |

- **Friction is linearly readable from z only along low-variance
  directions.** At α = 10, which halves any direction of standardized z
  holding under about 2.4 % of an average dimension's variance, the loud
  readout falls from 0.378 to 0.025. The inner-CV choice of α (the pilot's
  nested readout) picks 10⁻³ for every loud run, so the primary readout is
  the one a tuned readout picks there.
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
on both backends). Worlds `data/pm_f31fix_loud_cam` (seeds 0–9) and
`data/pm_f31fix_ctrl_cam` (seeds 0–2). `pm_big` is not run: the
old-camera pilot found nothing there to recover, and the above-floor
pilot was not run on `pm_big`.

**Epoch budget, OPEN (owner's decision at the freeze).** This draft
writes 50. Measured (§2): 150 epochs lowers the primary readout in 5 of 5
seeds (possibly through z's higher rank rather than less friction) and
raises held-out in 4 of 5, both modestly, at three times the cost (about 3 min per 50-epoch run, about 9 min per 150-epoch run on the
Mac; 65 runs ≈ 3.5 h vs ≈ 10 h, P2 and P2′ somewhat more). The case for
150 is a separate, unpublished observation: on `pm_big` the held-out
behavior gain for density kept rising to epoch 150; that readout is not
primary here.

Per-view latent v_i = relu(proj(GAP(CNN(image_i)))) ∈ ℝ²⁵⁶, unchanged. Only
the step from {v_1 … v_16} to z changes, selected by a new
`ModelConfig.view_pool` (default `"mean"`, so every existing config is
unchanged):

| arm | `view_pool` | z = | new parameters |
|---|---|---|---|
| P0 | `mean` | mean_i v_i (today's encoder, rerun under the new code) | 0 |
| P1 | `stats` | relu(W [mean_i v_i ; std_i v_i]), W: 512 → 256 | 131 k |
| P1′ | `stats_ctrl` | relu(W [mean_i v_i ; mean_i v_i]): P1's layer and ReLU, no cross-view information | 131 k |
| P2 | `attn` | mean_i of one pre-LayerNorm transformer block over the 16 tokens v_i + E·d_i (4 heads, FFN 512, residual), where d_i is camera i's unit direction and E: 3 → 256 is learned | ≈ 0.53 M |
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

**Matched controls.** P1 and P2 add capacity (a layer, a ReLU, depth), not
only mixing. P1′ and P2′ keep the added capacity and remove the mixing, so
the primary hypotheses (§5) compare each mixer with its own control and
test the mixing itself. Comparisons with P0 are reported as well.

65 runs in all (5 arms × 13).

## 4. Readouts

From `scripts/f31_measure_mlx.py` (on `claude/f31a-mlx-replication`; it
lands on this branch with the freeze), per checkpoint:

- **friction z R²**: 5-fold ridge from z (all 512 scenes) to normalized
  friction. Primary.
- **held-out friction**: ridge on train-split z, R² on the 20 corner scenes.
  Co-reported.
- **PR**: final latent participation ratio.
- **Descriptive:** density and restitution z R²; held-out behavior gain.

⚑ **Readout strength, flagged, readout not changed.** The primary readout
is a fixed, near-least-squares ridge (§2), and its out-of-fold R² falls as
the rank of standardized z rises, whatever z holds. P1, P1′, P2 and P2′
change what produces z and may change its rank, so part of a Δ could be
the readout's variance penalty, not friction. Draft 3 adds two
co-reported quantities per checkpoint: the rank of standardized z, and
friction R² under the nested readout (`scripts/z_readout_alpha.py`).
Whether the nested readout should replace the fixed one as primary, or
G2 be stated on it, is decided at the freeze.

**Density is compared only within the above-floor runs.** Its legibility
on fixed features fell from 0.66 to 0.06 with the camera change alone, and
P0-equivalent z reads it at about zero (§2). A density drop against any
old-world number is expected from the cameras and is not an arm effect.

**Diagnostic D1 (where friction is lost):** for every P0 checkpoint, the
per-view latents v_1 … v_16 are read out by nested 5-fold ridge (the
pilot's `nested_r2`) through five constructions: the plain concatenation
(4096 dims, linear in v), and the pilot's nonlinear cross-view features C,
D, G and H (§2) computed on v instead of on hand-made image statistics.
A linear readout of the concatenation cannot compute a spread or a
quantile, so the nonlinear constructions are what can show the information
is present per view. (D here is a fixed readout feature, not an FNet model.)
**D1** = the best of the five, mean over P0 seeds; **z₀** = P0's mean
friction z R².
- D1 high: D1 ≥ z₀ + 0.15. The information survives per view; the
  average discards it.
- D1 ≈ z: D1 ≤ z₀ + 0.05. The per-view CNN has already lost it.
- In between: unresolved.

D1 needs a per-view encode helper (added with the freeze).

## 5. Gates, hypotheses and frozen rules

Every Welch t is two-sample over the listed seeds. Power disclosure,
recomputed on the above-floor P0-equivalent runs: friction sd 0.085 (MLX,
5 seeds, at both 50 and 150 epochs). The smallest difference reaching
t = 2.5 is about 0.134 at n = 5, 0.106 at n = 8 and 0.095 at n = 10 per
arm; hence 10 loud seeds. Effects between 0.05 and 0.095 will read
INCONCLUSIVE. The figures assume P0's spread carries over to the mixers
and their matched controls; that is not measured.

**Gates (read first):**
- G1 health: every run's final PR ≥ 8.
- G2 control, **the memorisation guard**: every ctrl run of an arm has
  friction z R² < 0.10. The essence head regresses friction from z on the
  training scenes, and the 5-fold readout covers those same scenes, so a
  higher-capacity mixer could raise the readout by fitting that target
  rather than by reading highlights. On ctrl the pixels carry no material,
  so the same fitting would show there. A G2 failure withholds that arm's
  verdict.
- G3 apparatus (reported, does not withhold): P0's loud mean is within
  ±0.10 of the above-floor baseline at the chosen budget (0.378 at 50
  epochs; 0.341 if the owner picks 150). Verdicts compare against the P0
  rerun either way.

**H1 — stats mixing (P1 vs P1′)** and **H2 — attention mixing (P2 vs
P2′)**, primary, each graded on Δ = mean friction z R²(mixer) − mean(its
matched control), with Δ_h the same difference on the held-out readout:
- RISES: Δ ≥ +0.10, t ≥ 2.5, **and Δ_h > 0** (the 20 corner scenes are
  never trained on, so memorisation cannot raise them; F31a Amendment 2's
  co-report, made a condition)
- NULL: |Δ| < 0.05
- FALLS: Δ ≤ −0.10 and t ≤ −2.5
- otherwise INCONCLUSIVE

⚑ **Reachability of RISES, flagged, bar not changed.** The bar was set
when the best fixed cross-view features exceeded mean-pooled z by 0.37
(0.636 vs 0.267). On the above-floor worlds the margin is 0.17–0.19
(D 0.547, H 0.554, D′ 0.565 vs 0.378). RISES now needs a mixer to recover
more than half of that margin over its control, and the detectable
difference at n = 10 (0.095) sits close to the 0.10 bar. Separately, on
fixed features the held-out sign favours the average on these worlds
(§2), so Δ_h > 0 is not assured even when the 5-fold readout rises.
Whoever freezes decides whether to keep the bar, justify a new one from
the new margin, or add seeds.

Two primary comparisons. At df ≈ 18, t ≥ 2.5 is about p = 0.011 one-sided
per test, so about 0.02 family-wise; disclosed, not corrected.

**Reported against P0, same bars, secondary:** P1, P1′, P2, P2′ each vs
P0 (is any arm better than today's encoder, whatever the reason).

**H3 — mechanism (P2 vs P1)**, same bars, secondary: does direction-aware
mixing beat order-free statistics?

## 6. Predictions (registered at real odds)

⚑ **Draft 3: the probabilities below are draft 2's, not revised.** Their
rationales rest on old-camera numbers as noted under each. They are
re-registered, with new odds, in the freeze commit.

- **P1: H2 (attention vs its matched control) RISES 45 %**, NULL or
  INCONCLUSIVE 50 %, FALLS 5 %. The pilot shows cross-view structure worth
  about +0.3 to +0.45 over the averaged view on fixed features. Against it:
  the per-view CNN's global average may already have flattened the
  highlight, which D1 will show.
  ⚑ On the above-floor worlds that structure is worth +0.14 to +0.16 over
  the averaged view (+0.17 to +0.19 over P0-equivalent z), and held-out
  favours the average.
- **P2: H1 (stats vs its matched control) RISES 30 %.** The pilot's C
  (mean + std) gained only +0.13 over A on fixed features.
  ⚑ On the above-floor worlds C reads 0.371 *below* A (0.037 vs 0.408):
  the rationale's sign flips.
- **P3: G2 passes for every arm, 90 %.**
- **P4 (descriptive):** restitution z R² stays ≥ 0.97 in every arm
  (brightness reads the same from every view; nothing to gain).
  ⚑ Old-world-derived. Above-floor P0-equivalent z reads 0.975–0.987;
  fixed features 0.939–0.975.
- **P5: D1 reads "high" (≥ z₀ + 0.15), 55 %.** Thresholds are relative to
  z₀, so the switch does not change the rule; the odds are draft 2's.

## 7. What each outcome means

- **H2 or H1 RISES:** the view average was a bottleneck *on the
  above-floor cameras*, beyond what the camera fix already recovered (A
  0.177 → 0.408, §2). The verdict is relative to the above-floor average
  and to P0 on the same worlds; the camera fix's share is not credited to
  the mixer. The F28/F31a legibility arguments, built on the old-camera
  averaged view, need restating either way. Adopting a pooling variant as
  the default is a separate decision.
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
- **Density.** The 15–65° band costs most of the photos' density content
  (§2). Whether to change the band is an owner decision, OPEN at this
  draft; this preregistration takes the band as given and makes no
  density claim.
- The pilot's features are post hoc, chosen after seeing the
  mean-view result; they motivate the arms but are not graded.
- The cloud diagnostics cited in §1–§2 (MuJoCo 3.17 re-renders) are not
  reproduced on the Mac's photos and carry no weight in any bar.

## 9. Reproduce

Worlds (from `claude/camera-above-floor` @ `b7a62e2`):

    for spec in "loud 1 8" "ctrl 0 1"; do set -- $spec
      python -m pseudomarble.data.generate_mujoco --output data/pm_f31fix_$1_cam \
        --num-scenes 512 --views 16 --resolution 128 --seed 1234 --shapes box \
        --coupling-alpha $2 --coupling-gain $3 --camera-layout above
    done

Gate (`scripts/compare_worlds.py`, commit `7f9fc18`; ⚑ local only at this
draft, must be on GitHub before the freeze):

    python scripts/compare_worlds.py data/pm_f31fix_loud data/pm_f31fix_loud_cam
    python scripts/compare_worlds.py data/pm_f31fix_ctrl data/pm_f31fix_ctrl_cam

Pilot:

    python scripts/crossview_pilot.py --data data/pm_f31fix_loud_cam data/pm_f31fix_ctrl_cam
    python scripts/crossview_pilot.py --data data/pm_f31fix_loud data/pm_f31fix_ctrl data/pm_big

Trained baseline (trainer at `main` `7bb27ed`; loud seeds 0–4, ctrl 0–2;
`--epochs 150` for the longer budget):

    python -m pseudomarble.models.train --data data/pm_f31fix_loud_cam \
      --epochs 50 --lr 2e-4 --batch-size 16 --seed $S --out runs/xv_cam/loud_joint_e50_s$S
    python scripts/f31_measure_mlx.py --data data/pm_f31fix_loud_cam \
      --runs runs/xv_cam/loud_joint_e50_s* --out runs/xv_cam/loud_joint_e50.json

Readout-strength diagnostic (§2; needs MLX):

    python scripts/z_readout_alpha.py --data data/pm_f31fix_loud_cam \
      --runs runs/xv_cam/loud_joint_e50_s* runs/xv_cam/loud_joint_e150_s* \
      --out runs/xv_cam/readout_alpha_loud.json
