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
- **Raw frames (tier T1).** Rendered frames with no codec. A synthetic
  camera is a perfect global shutter: each frame is the exact state at one
  instant, and MuJoCo steps at 500 Hz (timestep 0.002 s), so frame rate is a
  free choice. MuJoCo's renderer returns 8-bit display RGB, not linear raw.
  Options for "raw linear": (i) undo the display curve and add a sensor model
  (shot noise, read noise, Bayer mosaic, 12-bit depth), which cannot recover
  clipped highlights; (ii) a linear HDR renderer (the repo has
  `generate_blender.py`, not yet used for dynamics).
- **Compressed video (tier T2, later, as a degradation arm only).** F33's
  clips went through ffmpeg at 4 fps, where a bounce is invisible. Codecs
  quantize away exactly the small, fast contact events §4 finds the
  information in.

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
  line. It is not part of this design.

## 8. Open decisions

1. Tier order: T0 first (recommended; it is the bound), then T1.
2. Noise level(s) to preregister, in mm for T0 and as a sensor model for T1.
3. Battery: a second friction family (§9, D2 result), then freeze the
   battery and the gate in one commit before any training.
4. Frame rate and window for T1: contact keyframes last ~64 ms.
5. Whether the T1 "raw linear" arm uses an inverse-display-curve sensor
   model on MuJoCo frames or a linear HDR renderer.
6. F-number assignment.

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
| scale | box on a spring plate (k 5300 N/m, c 1500 N·s/m) | D2 | **0.99** | −0.28 | −0.17 |
| tap | 0.2 kg ball dropped 0.3 m onto the box | D2 | −0.17 | −0.17 | **0.94** |

**D1** (`7c14978`, read by `888d741`). Gate: one family per axis, all three
FAIL. Lessons: the ramp lost restitution because the box now starts at rest
(no landing bounce); the graded push is friction-gated, since the box moves
only when F > μMg and its displacement then mixes mass with friction, so
the logarithmic force ladder does not rescue density; soft contacts let
boxes creep below tan θ = μ, so the thermometer is blurred.

**D2** (this commit) adds two friction-free probes. Gate: density PASS
(strike, scale), restitution PASS (drop, tap), **friction FAIL** (ramp only).
Shared pairs: strike↔scale (density), drop↔tap (restitution). D1
recordings are byte-identical to D1's, so cross-platform reference values
taken on D1 still hold.

**Next (D3): a second friction family.** Friction is read only through the
ramp, and every other probe mixes it with mass or force. A mass-free
friction probe: give the resting box a fixed initial horizontal velocity
(a "curling stone" kick on flat ground); its slide distance v₀²/(2μg) does
not depend on mass.

