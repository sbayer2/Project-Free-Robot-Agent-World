# F35 (provisional) — cross-interaction prediction: watch one interaction, predict another

**Status: DESIGN NOTE, 2026-10-26. Not a preregistration.** Nothing here is
frozen. It sets the question, the instrument, and the gates a preregistration
will need, and records the zero-training evidence that shaped them. The
F-number is provisional.

## 1. The question

Can a latent formed by **watching one interaction** with an object predict
the object's behavior in a **different interaction**, when the only thing the
two interactions share is the object's material?

Example: watch the object fall and bounce; predict how far it slides down a
ramp. Nothing in the drop footage is the slide's answer. The only route from
one to the other runs through density, friction and restitution, so a model
that succeeds must have recovered them, or something equivalent.

Why this is the project's strongest available test of "one essence":

- **No authored coupling.** F13–F31 asked whether *appearance* predicts
  behavior, and appearance only carries physics because `materials.py`
  writes it in (the F22a coupling dial). In motion, the simulator writes the
  physics itself: friction *is* how far the box slides.
- **Operational, not geometric.** F27 showed magnitude coherence certifies
  substrate entanglement and F28 showed directional coherence reads zero.
  This test scores a prediction on an interaction the input never showed,
  like F29's external yardstick, so it cannot be passed by latent geometry.
- **Not circular.** "Watch the slide, predict the slide" puts the answer in
  the input. Cross-interaction pairs keep input and target disjoint.

## 2. The framing: physics is the encoder

A signal-processing view, made precise enough to design with:

| Coding term | Here |
|---|---|
| message | the material: (density, friction, restitution), 3 numbers |
| encoder | the interaction (probe): spreads the message over thousands of position × time values |
| channel | sensing: camera, resolution, frame rate, noise, compression |
| decoder | the learned model |
| code design | **probe design**, which we control |

Three consequences:

1. **The code is very low rate** (3 numbers spread over thousands of
   samples), so noise is rarely the limit. The real risk is that the
   encoder fails to write part of the message at all: an *erasure*. §4 finds
   several.
2. **Reed–Solomon codes are not applied to raw data.** A Reed–Solomon
   decoder only recovers messages that were encoded with that code, assumes
   a symbol is either exactly right or wrong (sensor noise touches every
   pixel a little), and *adds* redundancy where this goal is to keep only
   the message. What carries over is the **erasure question**: Reed–Solomon
   recovers the message from *any* k of n symbols. For a learned decoder that
   becomes a measurement: from which subsets of frames is each axis
   recoverable?
3. **Keyframes and in-betweens** (the owner's animation analogy). An
   animator fixes keyframes and draws in-betweens the viewer's own sense of
   motion will accept. In physics, the in-betweens are the frames the prior
   predicts without knowing the material, such as free flight under gravity,
   and they carry **no** material information. The keyframes are the
   contacts, where the material changes what happens. A frame's information
   is how far it departs from what the prior would draw. §4 measures this
   directly. Unlike Reed–Solomon, physics is not "any k of n": the frames are
   not interchangeable.

## 3. What "video" means in this world

The signal chain, and the choices it forces:

- **State (tier T0).** The simulator's own trajectory: position and up
  vector per frame (`--keep-trajectory`, 125 samples over 2 s). No pixels.
  The oracle tier: anything a camera could recover is bounded by this.
- **Raw frames (tier T1).** Rendered frames kept as numeric arrays, never
  passed through a codec. A synthetic camera is a perfect global shutter:
  each frame is the exact state at one instant, and MuJoCo steps at 500 Hz
  (timestep 0.002 s), so frame rate is a free choice. MuJoCo's renderer
  returns 8-bit RGB that is **linear in light**: no display curve is
  applied. Measured 2026-10-27 on MuJoCo 3.9.0: a grey plane (albedo 0.5)
  under a headlight of 0.05–1.2 renders at 6, 13, 26, 51, 102, 153, exactly
  0.5 × light × 255. What it lacks against a CMOS raw is bit depth (8 bits,
  clipped at 1.0) and sensor physics (no shot or read noise, no mosaic).
  Because the output is linear, rendering each frame at two light levels
  and merging them extends it to about 12 bits; a sensor model then adds
  the noise. §10 specifies this.
  *Correction:* an earlier version of this note said the renderer returns
  "8-bit display RGB, not linear raw" and proposed undoing a display curve.
  That was not checked and is wrong; there is no display curve to undo.
- **Compressed video (tier T2): not on the main path.** Owner decision,
  2026-10-27: no codec and no text anywhere in F35. F33's clips went through
  ffmpeg at 4 fps, where a bounce is invisible; codecs quantize away exactly
  the small, fast contact events §4 finds the information in.

**A noise model is mandatory.** The simulator is noiseless, so without
noise a decoder can read sub-millimetre differences perfectly. The pilot hit
this: a "free fall" window that ended 2 ms into contact read restitution at
R² 1.000, because contact stiffness depends on restitution. The noise model
*is* the channel, and every claim must name its noise level.

## 4. Evidence so far (zero training)

`scripts/f35_code_table.py` on `data/pm_f31fix_loud` (512 boxes, real 20°
ramp, state trajectories resampled to 64 frames, inverse R² best of ridge and
kNN-5, 5-fold).

### 4.1 Code table: which interaction encodes which axis

| probe | noise | density | friction | restitution |
|---|---|---|---|---|
| drop | 0 mm | −0.07 | +0.82 | +1.00 |
| drop | 5 mm | −0.19 | −0.24 | +0.99 |
| ramp (20°) | 0 mm | −0.19 | +0.92 | +0.98 |
| ramp (20°) | 5 mm | −0.16 | +0.86 | +0.81 |
| push (1.5 N·s) | 0 mm | +0.01 | +0.71 | +0.97 |
| push (1.5 N·s) | 5 mm | −0.08 | −0.17 | −0.17 |

At any realistic noise:

- **Density is erased by the whole battery.** Drop and ramp are
  gravity-driven, so mass cancels (Galileo). The push gives every box the
  same 1.5 N·s, and boxes run 1.4–216 kg, so most barely move (the class page
  reported 84 % under 1 cm).
- **The push is erased** at 1 mm noise.
- **The drop carries restitution only.** Its friction signal exists only
  noiseless (see 4.2).
- **The ramp carries friction and restitution**, but friction only partly:
  70 % of boxes have μ > tan 20° = 0.364. They only skid from landing (median
  22 cm), and their slide barely tracks μ (r = −0.21). Below the threshold the
  slide tracks μ almost perfectly (r = −0.99). The 20° ramp erases most of
  the friction range.

**So "watch the drop, predict the slide", as literally stated, can only
transfer restitution.** The battery has to be redesigned before the
experiment means anything (§5).

### 4.2 Keyframes: where the information lives (drop)

| frames (contact at t = 0.350 s, frame 11) | noise | friction | restitution |
|---|---|---|---|
| free fall (0–10) | 0 mm | −0.00 | −0.00 |
| first 4 frames of contact (11–14, ~64 ms) | 0 mm | +0.95 | +1.00 |
| first 4 frames of contact | 1 mm | +0.09 | +0.99 |
| after contact (15–63) | 5 mm | −0.23 | +1.00 |
| same random 4 of 64 frames, every scene | 5 mm | −0.07 | +0.88 |
| same random 8 of 64 frames, every scene | 5 mm | −0.15 | +0.99 |

- **Free flight carries nothing**, exactly as the in-betweens should.
- **Restitution is written redundantly**, re-encoded at every bounce. Any 8
  frames recover it. This is the closest physics comes to "any k of n".
- **The drop's friction is one fragile keyframe**: sub-millimetre motion in
  the first ~64 ms of contact, erased by 1 mm of noise.

## 5. Probe battery redesign (the encoder we control)

Gate before any training: **every axis must be shared by at least one
ordered pair of probe families**, where a pair shares an axis when both
families read it at inverse R² ≥ 0.8 at the preregistered noise
(`scripts/f35_code_table.py` prints the pair matrix and the gate). Why this
is the bar: an axis read by only one family can only ever appear in a
non-shared (target-side) cell, so whether it transfers would be untestable by
design. A shared pair is the minimum for every axis to have a positive
cell. The same matrix defines the control cells of §6. It is equivalent to
"at least two families read the axis", but stated as pairs because the test
is scored on pairs. It is not sufficient for the third-interaction readout,
which needs the trained input family to carry whatever the third
interaction depends on; that is checked per pair from the same table.

Every box is the same 0.3 m cube, so mass = density × 0.027 m³
(1.35–216 kg): the density column reads mass up to a constant.

- **Friction: a thermometer ramp.** Several ramp angles θ_k with tan θ_k
  tiling the friction range (0.05–1.10). Which ramps the box slides on
  places μ in a band, and slide distance on those that slide refines it. A
  unary (thermometer) code. Candidate angles: 5.7°, 11.3°, 16.7°, 24.2°,
  31.0°, 38.7° (μ = 0.1, 0.2, 0.3, 0.45, 0.6, 0.8). The box starts **at rest
  on the surface**, not 5 cm above as now, so the slide/no-slide threshold
  is not blurred by a landing skid. MuJoCo has one friction coefficient (no
  static/kinetic split), so the threshold should sit near tan θ = μ, but
  contact softness may creep: measure it. Stay below the cube's 45° tipping
  angle; μ above about 0.84 falls in the top "never slides" band. Boxes only:
  a sphere rolls at any angle, and its slip threshold is different.
- **Density: a momentum probe.** Gravity cannot reveal mass. Options: a
  reference sphere of known mass and speed strikes the object, and the
  rebound of both encodes the mass ratio (with restitution, which the drop
  gives independently); or graded impulses with log-spaced magnitudes and
  log-displacement outcomes (the "graded-push probe" already on the open
  list).
- **Restitution: the drop**, unchanged. Already robust.

Re-run `f35_code_table.py` on the new battery. Proceed only if the gate passes.
The battery as built, iteration by iteration, is in §9.

## 6. Experiment sketch

- **World.** Boxes only (no shape-carried variation, the F26/F27 lesson),
  and **one fixed appearance** for every object. Material must be
  unreadable from looks, so any success comes from motion. Fixed ramp
  geometry (post-#45).
- **Input.** One probe's recording: T0 state first, then T1 frames.
- **Target.** Another probe's outcomes, per-field standardized or with
  log slide distance. Never a raw MSE in which one field dominates (the F31a
  head-collapse lesson). The head-collapse diagnostic is mandatory.
- **Pairs.** The full input × target matrix, excluding the circular
  diagonal.
- **Built-in controls.**
  - *Shared-axis cells vs non-shared cells.* From the gated code table, each
    pair has axes both probes encode (transfer predicted) and axes only the
    target encodes (transfer must be at predict-the-mean). A "transfer" on a
    non-shared axis is a leak or shortcut.
  - *Shuffled pairing.* Input from object i, target from object j: must fail.
  - *Oracle floor (F34 discipline).* Ridge or kNN from the input's T0 state
    to the target outcomes. A learned model must match it before it is
    credited with anything.
- **The unity test proper.** Train the encoder on A → B, freeze it, then
  fit a linear readout from its latent to an interaction C it never trained
  on. One essence predicts a third interaction it was not built for.
- **The erasure measurement**, for each trained decoder: recoverability vs
  frame subset (free flight vs contact windows; fixed random subsets of size
  k), at the preregistered noise. Prediction from §4.2: flight frames are
  removable at no cost, contact frames are not, and restitution degrades
  far more gracefully than friction.

## 7. Relation to existing work

- **F31** made the trajectory a *target*; F35 makes it an *input*, from a
  different interaction than the target.
- **F33** rendered rollout clips for an LLM baseline; its renderer
  (`scripts/f33_render_clips.py`) is the starting point for T1, without the
  4 fps codec chain.
- **Impact-audio proposal** (ai-research vault): a tap as input has the
  same structure. Its timing channel is simulated (like T0); its timbre
  channel is authored (like appearance) and must be labeled as such.
- **F29**: the operational yardstick this inherits.
- **The static question stays separate.** The 2026-10-26 pooling pilot
  (friction R² 0.18 from the averaged view vs 0.64 from cross-view features,
  controls at or below zero) is about appearance and belongs to the F13–F31
  line. It is not part of this design. Its draft preregistration is
  `docs/CROSSVIEW_POOLING.md` on `claude/crossview-pooling-prereg`.

## 8. Open decisions

1. Tier order: T0 first (recommended; it is the bound), then T1. §10 needs
   only the T0 *oracle floor* (a ridge/kNN fit), not a trained T0 model.
2. Noise level(s) to preregister, in mm for T0 and as a sensor model for T1.
   T1 sensor model proposed in §10 (DRAFT).
3. Battery: the gate passes at D4 (§9). Decide whether the unity test needs
   a multi-axis input family (§9, D3 consequence), then freeze the battery
   and the gate in one commit before any training.
4. Frame rate and window for T1: contact keyframes last ~64 ms. Proposed in
   §10: 125 fps over the full 2 s.
5. ~~Inverse display curve or linear HDR renderer~~ Resolved: MuJoCo's
   output is already linear (§3); §10 uses exposure bracketing plus a
   sensor model. Blender remains an option for a photoreal follow-up.
6. F-number assignment.
7. Architecture for T1: owner decision 2026-10-27, an attention transformer
   over frame tokens; no FNet arm (§10.9).

Reproduce §4: `python scripts/f35_code_table.py --data data/pm_f31fix_loud`
(needs a `--keep-trajectory` world generated after #45).

## 9. Design iteration log

**Every battery in this section is DESIGN, not evidence.** The gate is being
set after its first results were seen, which is legitimate for an
instrument gate (redesigning a battery until it can read the axes is
design work) on two conditions: the battery and gate are frozen in one
commit before any F35 training, and no F35 hypothesis is scored on a
battery tuned against it. Numbers below: `data/pm_f35_battery_sampled`
(`--sample 512 --seed 1234`), inverse R², 5 mm position noise, families read
whole; the `--source data/pm_f31fix_loud` world agrees to about ±0.05.

**Modeling convention (strike, tap):** the reference ball's contact
parameters carry the box's restitution, so each contact's restitution is
attributed to the box. Physically it belongs to the pair; this is authored.

| family | probe | D | density | friction | restitution |
|---|---|---|---|---|---|
| drop | 0.6 m drop (existing) | D1 | −0.23 | −0.17 | **0.99** |
| ramp | thermometer ramp, 6 angles 5.7–38.7°, box at rest | D1 | −0.16 | **0.93** | 0.24 |
| push | graded centre push, 15–3840 N × 0.1 s | D1 | 0.25 | 0.44 | 0.00 |
| strike | 17 kg ball at 4 m/s into the resting box | D1 | **0.88** | 0.36 | 0.79 |
| scale | box on a spring plate (k 5300 N/m, c 1500 N·s/m); stiff plate contact from D4 | D2, D4 | **1.00** | −0.26 | −0.21 |
| tap | 0.2 kg ball dropped 0.3 m onto the box | D2 | −0.17 | −0.17 | **0.94** |
| kick | box on flat ground given 1 and 3 m/s at t = 0 | D3 | −0.20 | **0.97** | 0.20 |

**D1** (`7c14978`, read by `888d741`). Gate: one family per axis, all three
FAIL. Lessons: the ramp lost restitution because the box now starts at rest
(no landing bounce); the graded push is friction-gated, since the box moves
only when F > μMg and its displacement then mixes mass with friction, so
the logarithmic force ladder does not rescue density; soft contacts let
boxes creep below tan θ = μ, so the thermometer is blurred.

**D2** (`c2b02e5`) adds two friction-free probes. Gate: density PASS
(strike, scale), restitution PASS (drop, tap), **friction FAIL** (ramp only).
Shared pairs: strike↔scale (density), drop↔tap (restitution). D1
recordings are byte-identical to D1's, so cross-platform reference values
taken on D1 still hold.

**D3** (`b85bda4`) adds the mass-free friction probe D2 called for: the
resting box is given a horizontal velocity on flat ground (a "curling
stone" kick). Under Coulomb friction x(t) = v₀t − μgt²/2 until it stops at
v₀²/(2μg), with no mass in it. Smoke test: slides match that within a few
percent at 3 m/s and do not depend on mass (μ ≈ 0.45 at 1.7 kg and 14.9 kg:
0.118 vs 0.117 m). At 1 m/s high-μ boxes slide 0.07–0.14 m against
≈ 0.056 m predicted, so the 1 m/s kick alone reads friction at only 0.49;
3 m/s reads 0.98. 13 of 512 boxes tip at 3 m/s, all with μ ≥ 1.07 (a
sliding cube tips forward when μ > 1). D1 and D2 recordings are
byte-identical.

**Gate: PASS on all three axes** (both worlds): density via strike and
scale, friction via ramp and kick (0.97, and 0.96 even at 20 mm), and
restitution via drop and tap. Six of 42 ordered pairs share an axis.

**D4** (this commit) fixes a defect the cloud session's cross-version
reproduction found. In D2/D3 the box-plate contact used the box's own
restitution-derived contact parameters (soft, and barely damped for bouncy
boxes) against a 2 cm plate. In 24 of 512 sampled scenes, all with
restitution ≥ 0.53, the box tunnelled through the plate, ending up to
0.69 m below it; even passing scenes penetrated up to 4 cm. Tunnelling is
chaotic, so those scenes differed between MuJoCo builds (11 of 13
reference recordings matched to 1e-6 across machines on 3.9.0; only the
two scale recordings did not). It also leaked restitution into the scale
(R² 0.68 noiseless). Fix: the plate takes contact priority with stiff,
critically damped parameters (solref 0.01 1) and is 10 cm thick. Maximum
penetration is now 3.3 mm, no scene exceeds 1 cm, and the plate's sink
matches (M + 1)g/k within 1.3 mm (r 0.99999). The scale now reads density
at 1.00 and restitution at −0.01 noiseless. Only `traj_scale` and
`traj_scale_plate` changed; every other recording is byte-identical. Gate
unchanged: PASS on all three axes. Lesson: a probe that adds a second
contact surface inherits the box's contact softness unless it overrides it,
and a soft contact against a thin geom can tunnel.

**Consequence for the experiment (open, not a design fix):** every family
reads exactly one axis at the bar. The strike comes closest to two
(density 0.88, restitution 0.79 / 0.73). So every positive cell transfers a
single axis, and the unity readout of §6 (train A → B, freeze, read a third
interaction C) only has something to find when C depends on the same axis.
A test of *one essence*, rather than three separate one-axis channels,
needs at least one input that carries several axes at once: the strike
pushed over the bar on restitution, a composite interaction (e.g. a drop
onto the ramp, which mixes bounce and slide), or the input taken as several
families watched together. That choice belongs to the preregistration,
before the freeze.


## 10. Amendment R1 (DRAFT): the T1 pixel arm, an attention transformer on uncompressed frame arrays

**Status: DRAFT, 2026-10-27. Not frozen.** R1 freezes together with the
battery and gate (§8 item 3), in one commit that also adds the frame
generator, the model and `scripts/f35_r1_verdicts.py`, before any R1 model is
trained. Until then everything here can change.

**Owner decisions recorded (2026-10-27):**
- The model reads rendered frames stored as numeric arrays (`.npz`). No
  PNG, no video codec, no text anywhere: no captions, prompts or language
  model. Physics → frame arrays → model → numbers.
- One architecture: an attention transformer over frame tokens. FNet is not
  tested (§10.9).

### 10.1 The question R1 adds

T0 asks what the exact state carries. R1 asks whether a model that sees
only pixels, through a realistic sensor, recovers the same transfer: watch
interaction A as raw frames, predict interaction B. The bar is the T0
oracle floor at the matching precision (§10.5), so R1 measures what the
camera costs, not whether physics transfers at all.

### 10.2 Frames (the channel)

- **Scene.** Every box has the same albedo and finish (one fixed
  appearance, §6), so material is unreadable from any single frame.
  Lighting fixed.
- **Camera.** One fixed camera per input family, never moving. Framing is
  set from the family's T0 extents over the whole world plus a margin, so
  the full 2 s path stays in view; the metres per pixel is written to the
  manifest.
- **Timing.** 125 fps (one frame every 4 simulator steps of 2 ms), the full
  2 s window: 250 frames per recording. A contact keyframe (~64 ms, §4.2)
  spans 8 frames. Each frame is the exact state at one instant (zero
  exposure time, no motion blur; a disclosed idealization). T0 for R1 is
  recorded at the same 125 fps so frames and states align one to one.
- **Resolution.** 96 × 96 (disk: §10.8).
- **Linear exposure pair.** Each frame is rendered twice, with every light
  scaled by 1 and by 1/16. MuJoCo's output is linear in light (§3), so
  merging the pair (use the bright frame where it is below 250, otherwise
  16 × the dim frame) gives linear radiance over about 12 bits with no
  clipping below 16× the nominal white. The freeze commit includes a test
  that the merge is linear across the seam.
- **Sensor model**, applied in numpy from a per-scene seed at load time, so
  disk holds only the noiseless exposure pair:
  1. scale linear radiance so the merged range maps to a full well of
     10 000 e⁻;
  2. Poisson shot noise;
  3. Gaussian read noise, σ = 3 e⁻;
  4. RGGB Bayer mosaic: one colour sample per pixel, one channel;
  5. 12-bit ADC, 0–4095, clipped.
  The model's input is that single-channel 12-bit mosaic: the closest a
  simulator gets to the "raw binary" of a global-shutter CMOS sensor.
- **What stays synthetic:** the 8-bit quantization of each exposure (fine
  steps after merging, but not a true analog signal), MuJoCo's lighting
  model, and zero exposure time. Blender (`generate_blender.py`, linear
  float output) is the route to photoreal raw if R1 is worth extending.

### 10.3 Model

New module (one module per concept), under the repo's three-backend rule:
MLX canonical, torch mirror with a parity test, numpy forward for CI.

- **Frame encoder**, shared across frames. Conv 32/64/128, kernel 3, the
  first with stride 2 so each step covers one 2 × 2 Bayer cell. Then
  **no global average pool**: averaging over the image throws away where
  the box is, and position is the signal. Instead a 1 × 1 conv to 32
  keypoint maps, a spatial soft-argmax per map (x, y), and a linear layer
  from the 64 coordinates to a 256-dim frame token (deep spatial
  autoencoder style, Finn et al. 2016).
- **Temporal transformer.** 250 frame tokens plus a learned time embedding;
  4 pre-LayerNorm blocks, width 256, 4 heads, FFN 512, dropout 0.1; mean
  over tokens → z (256) → MLP head → target outcomes.
- **Frame dropout in training.** Each frame token is replaced by a learned
  mask token with p = 0.25, identically in every run, so the erasure tests
  (§10.6) are in distribution at evaluation.
- **Loss.** MSE on the target family's outcomes, each field standardized on
  train statistics, never a raw MSE in which one field dominates (the F31a
  head-collapse lesson). Outcome fields per family are defined in the freeze
  commit.

### 10.4 Pairs, splits, seeds

- **Pairs:** the frozen battery's shared pairs. At D4: kick → ramp
  (friction, primary: the faintest keyframe), drop → tap (restitution),
  strike → scale (density). The multi-axis input question (§9) may add one
  pair at the freeze.
- **Split:** one fixed iid 80/20 split by scene (seed 0), the same for
  every pair. **Seeds:** 0–4 per pair.

### 10.5 The bar: the T0 oracle floor at pixel precision

For each pair, **F_px** = best of ridge and kNN-5 from the input family's
T0 trajectory (125 fps, position noise σ = 0.5 px × that camera's metres per
pixel) to the target outcomes, fit on the train split, scored on the test
split. F_5mm (σ = 5 mm, the §5 gate noise) is co-reported. A pixel model
cannot beat exact state, except through noise; F_px is what a perfect
tracker at the camera's resolution would get with a linear or
nearest-neighbour readout.

### 10.6 Gates, hypotheses and frozen rules

**Gates:**
- G1 apparatus: the box centroid from a segmentation render matches the
  projected T0 position within 0.5 px RMS over all frames of every
  recording (checks frame timing and camera).
- G2 head health, per run: train gain ≥ 1.5 and the spread of predictions
  across scenes ≥ 0.5 × the target's. A failing run is reported and
  withheld (F31a's heads collapsed without anyone checking).
- G3 controls, per pair: shuffled pairing (input from scene i, target from
  scene π(i)) held-out R² ≤ 0.05; first frame only (the box at rest, before
  any interaction) held-out R² ≤ 0.05. A G3 failure withholds that pair.

**H1 — transfer reaches the floor (primary on kick → ramp; secondary on
the other pairs).** S = mean over seeds of held-out R² of the target
outcomes (mean over standardized fields).
- REACHES: S ≥ F_px − 0.05
- SHORT: S ≤ F_px − 0.20
- otherwise PARTIAL

**H2 — erasure (the keyframe analogy, §2 item 3 and §4.2)**, evaluated on
H1's trained models by masking frames at test time:
- (a) Drop → tap: masking every frame before the first contact costs ≤ 0.02
  in S. In-betweens carry nothing.
- (b) Keeping a fixed random 25 % of frames: the restitution pair loses less
  S than the friction pair. Restitution is redrawn at every bounce;
  friction is one faint keyframe.

**Descriptive (no verdict):** the same model on noiseless frames (the leak
the noise model exists to prevent); z → each essence axis, 5-fold ridge;
full erasure curves (pre-contact, contact-onset, motion, rest windows; k
of n random subsets).

### 10.7 Predictions (registered at real odds, before any frame exists)

- **Friction pair (kick → ramp) REACHES 45 %, PARTIAL 35 %, SHORT 20 %.**
  Low-μ kicks slide metres, so framing for them makes a pixel several
  centimetres; the slide-and-stop is large but its fine detail is
  sub-pixel. F_px prices that in, but a learned tracker may still fall short
  of a perfect one.
- **Restitution pair REACHES 65 %**: bounce heights are large and redundant.
- **Density pair REACHES 50 %**: the strike's rebound is fast and brief.
- **G2 passes in every run, 70 %**, given the F31a history.
- **H2 (a) holds 75 %; H2 (b) holds 70 %.**

### 10.8 Compute and disk

Per input family: 512 scenes × 250 frames × 96 × 96 × 2 exposures of uint8
≈ 2.4 GB before compression; the background is static and noise is added at
load, so `.npz` compression should cut this substantially. The Mac has
about 58 GB free (2026-10-27). Run time per model is unknown: one
single-seed smoke run, labeled as such and not graded, measures it before
the freeze. If one run takes over 30 minutes, R1 is cut to the friction
pair.

### 10.9 Not tested, and why

- **FNet** (fixed Fourier token mixing), by owner decision. The reasoning,
  recorded: the format of the input (`.npz` vs a lossless PNG) does not
  favour either architecture; what was decisive is that a Fourier basis is
  global in time while contact keyframes are brief, and that attention is
  the stronger default (FNet reached 92–97 % of attention's accuracy on
  language). A learned spectral filter (GFNet / FNO style) is likewise not
  tested.
- **Codec or text inputs** (§3, owner decision).
- **Photoreal rendering** (Blender): only if R1 is worth extending.
