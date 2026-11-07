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
  and merging them extends its range above the clip point (two stops with
  §10's × 1/4 bracket); a sensor model then adds the noise and digitizes at
  12 bits. §10 specifies this.
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
   T1 sensor model proposed in §10 (DRAFT): SNR ≈ 50 at nominal white.
3. ~~Battery and gate~~ **Frozen 2026-11-04** (§9, FREEZE): battery D7,
   outcome fields D7.1 (no masks), bar 0.8 at 5 mm on one designated pair
   per axis, canonical MuJoCo 3.9.0. R1 is not frozen.
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

**D5** (this commit) fixes a convention leak and the readout, and moves
to R1's frame rate. Three changes:

1. *Ball–ground leak.* The reference ball's geom-level solref (the box's
   restitution, by the convention above) applied to **every** contact the
   ball made, including the ground, whose friction is the box's own
   (`build_mjcf`). After rebounding off the box, the strike ball drops
   about 10 cm and bounced on the ground with the box's restitution and
   friction, the box not involved: the label replayed against the floor.
   An explicit `<pair>` (ball, ground) now gives those contacts fixed,
   material-independent parameters (friction 0.5, solref `0.01 1`); the
   convention remains only on ball–box contacts. Verified: one
   ball–ground contact per step, with the fixed parameters whatever the
   material. The tap ball never touches the ground in any of the 512
   scenes (its recordings are byte-identical to D4's), so the tap was never
   affected.
2. *Readout.* The ridge half of the inverse R² used a fixed α = 1, which
   overfits once a family's columns outnumber its scenes: D4's strike
   family read friction at −0.4 to −1.0, and reading the same ball at 125
   rather than 64 frames dropped its restitution from 0.89 to 0.78. α is
   now chosen per axis by inner 4-fold CV (1e-2 … 1e5, dual form so wide
   families stay cheap); kNN-5 is unchanged. Families are still read whole.
3. *125 fps* (one frame every 4 steps, 250 frames), R1's rate. Every
   recording but the strike's equals D4's at the shared frames.

Strike family, restitution, 5 mm, sampled world, separating the two fixes:

| battery | readout | frames | density | friction | restitution |
|---|---|---|---|---|---|
| D4 (leak) | α = 1 | 64 | 0.88 | 0.36 | 0.79 |
| D4 (leak) | CV α | 64 | 0.88 | 0.54 | **0.88** |
| D5 (fixed) | CV α | 64 | 0.81 | 0.52 | 0.71 |
| D5 (fixed) | CV α | 250 (125 fps) | 0.83 | 0.55 | **0.74** |
| D5, impact only (before the ball first touches the ground, frames 0–17) | CV α | 250 | 0.65 | 0.26 | 0.70 |

The readout fix alone would have passed the strike on two axes, on the
leaked signal. Leak-free, the box–ball impact reads restitution at
0.70–0.74, below the bar at 125 fps; the source world agrees (0.74). The
strike's friction (0.55) is genuine: the box slides after the impact.

D5 family table (5 mm, CV α, 125 fps; sampled world, source world within
±0.04):

| family | density | friction | restitution |
|---|---|---|---|
| drop | −0.01 | −0.00 | **0.98** |
| ramp | −0.01 | **0.94** | 0.37 |
| push | 0.37 | 0.60 | 0.13 |
| strike | **0.83** | 0.55 | 0.74 |
| scale | **1.00** | −0.01 | −0.01 |
| tap | −0.01 | −0.02 | **0.95** |
| kick | −0.01 | **0.98** | 0.31 |

**Gate: PASS on all three axes**, the same six shared pairs as D4:
strike ↔ scale (density), ramp ↔ kick (friction), drop ↔ tap
(restitution). With CV α, readouts that carry nothing sit at 0 instead of
going negative (D4's −0.2 density cells).

**Consequence for the experiment (open, not a design fix):** every family
reads exactly one axis at the bar. The strike comes closest to two
(density 0.83, restitution 0.74 leak-free); D4's 0.79–0.89 was partly the
ball–ground leak. So every positive cell transfers a
single axis, and the unity readout of §6 (train A → B, freeze, read a third
interaction C) only has something to find when C depends on the same axis.
A test of *one essence*, rather than three separate one-axis channels,
needs at least one input that carries several axes at once: the strike
pushed over the bar on restitution, a composite interaction (e.g. a drop
onto the ramp, which mixes bounce and slide), or the input taken as several
families watched together. That choice belongs to the preregistration,
before the freeze.

**D6** (`61f005f`) takes the composite route and levels the strike ball.

1. *Drop onto the ramp* (`dropramp38.7`): the drop's 0.6 m fall onto the
   steepest ramp, box flush with the plane. It bounces off the slope
   (restitution), then slides or stops (friction). Per recording at 5 mm:
   density −0.00, friction **0.87**, restitution **0.98**, the first single
   input over the bar on two axes. 78 of 512 boxes tumble; under the repo
   contact model the tumblers are the grippy ones (r(tumble, μ) +0.56).
2. *Strike ball under gravity compensation*, so it flies level instead of
   landing about 70 ms after the impact. Strike family 0.87 / 0.69 / 0.71.
   Caveat (cloud cross-check): the ball flies level in only 215 of 512
   scenes; in 171 it touches the ground after the impact (heavy boxes) and
   in 152 it climbs over light boxes (35 rise above 50 cm, out of a tight
   camera window). The box tips under ground friction when struck, so the
   ball's deflection depends on friction and mass, which is probably part
   of why the strike's friction rose from 0.55 to 0.69.
3. *Outcome fields* (`scripts/f35_outcomes.py`, the targets a model
   predicts; §10.3a): drop/tap e1 0.99, scale 1.00, strike v_ball 0.89 and
   v_box 0.84, kick3.0 0.96, dropramp e1 0.99 and log_slide 0.97 (field
   from the essence).

Gate PASS on all three axes, 14 of 56 ordered pairs. The cloud reproduced
every number on MuJoCo 3.9.0 and 3.17.0 (only the 0 mm rows and two cells
differ between versions, by 0.01). **Found in D6: restitution leaks into the
friction targets.** Read from their outcome fields, the ramp family carries
restitution at 0.86 and the kick at 0.77. Physics, not field definition:
under the repo contact model friction is a soft constraint sharing the
box's restitution-derived solref, so boxes that should hold a 38.7° slope
creep down it, 151 mm median and 589 mm maximum (n = 96, raw μ > tan 38.7°
+ 0.1), with r 0.87 against restitution and −0.02 against friction.

**D7 pilot** (`2dcd620`, `3cbfa25`, `c3b2e22`, `2df55d1`;
`scripts/f35_contact_pilot.py`; seed 1234, one noise draw). Contact
variants at 38.7°, held-box creep median / max, r with restitution:

| contact | creep mm | r(e) | kick1.0 friction (per recording) |
|---|---|---|---|
| repo (D6) | 151 / 589 | +0.87 | 0.63 |
| elliptic, impratio 1 | 52 / 390 | +0.78 | 0.65 |
| elliptic, impratio 10 | 21 / 295 | +0.70 | 0.65 |
| elliptic, impratio 100 | 18 / 280 | +0.70 | 0.66 |
| elliptic + box–ground pair, solreffriction `0.01 1` | **4 / 14** | +0.45 | **0.96** |

impratio alone does not fix the leak; a fixed friction solref does. Under
it, the remaining outcome leak is field definition: ramp slides floored at
1 cm (the existing "still" threshold; 2 and 5 cm give the same) take the
ramp family's restitution from 0.70 to −0.00 with friction 0.98, and
dropping kick `stop_s` (it repeats `log_stop`'s stop event in time) takes
the kick's from 0.79 to −0.01 with friction 1.00. Under the repo model the
same changes leave 0.85 and 0.66: there it is physics.

The fix breaks the 38.7° drop onto the ramp (432 of 512 tumble; its
rebound tracks the flat drop's only at r 0.34). Angle sweep, then drop
height at 20°:

| 20°, drop gap | tumbled | r(tumble, e) | per recording f / e | slide inverse R² f / e | r(dropramp.e1, drop.e1) |
|---|---|---|---|---|---|
| 0.60 m | 144 (28%) | +0.63 | 0.85 / 0.93 | 0.70 / 0.36 | 0.91 |
| **0.30 m** | **75 (15%)** | **+0.50** | **0.82 / 0.92** | 0.65 / 0.55 | 0.91 |
| 0.15 m | 38 (7%) | +0.36 | 0.79 / 0.90 | 0.66 / 0.59 | 0.91 |
| 0.08 m | 30 (6%) | +0.33 | 0.75 / 0.84 | 0.65 / 0.55 | 0.90 |
| 0.04 m | 23 (4%) | +0.29 | 0.69 / 0.73 | 0.64 / 0.58 | 0.88 |
| 0.02 m | 23 (4%) | +0.28 | 0.63 / 0.55 | 0.67 / 0.56 | 0.87 |

Three readings. (1) The contact model decides who tumbles: grippy boxes
under repo, bouncy boxes under the fix (r(tumble, e) +0.44 to +0.67 from
10° to 30°). The mechanism (a stiff friction grip catches a bouncing edge)
is post hoc and untested. Masking tumblers would therefore grade friction
on a restitution-selected subset. (2) A lower drop shrinks that selection
but floors near 4% with r ≈ +0.3, and costs friction information below
0.3 m. (3) The dropramp slide carries restitution under every contact
model and angle tried (repo: 0.76 at 10°, 0.15 at 38.7°), so it is a
composite field, never a pure friction target. Where the slide is measured
from (release point or rest pose) changes nothing.

Cross-check (cloud, notes on `claude/cloud-notes`): every pilot number
reproduces on MuJoCo 3.9.0 (arm and x86) and 3.17.0 (x86). The only
differences are the MuJoCo version, not the platform: one borderline
tumbler at 0.08 and 0.04 m, and at 0.02 m the 3.17.0 dropramp restitution
falls to 0.38 (a few-mm bounce near the 5 mm noise). At 0.3 m both
versions agree in every column. Repo tumbles nothing at ≤ 30°.

**D7** (`aacd3bc`, `2d7a519`; owner decision 2026-11-03). Battery: the
elliptic cone with a box–ground pair, solreffriction `0.01 1`, for every
recording; the drop onto the ramp at 20° from 0.3 m. `--design D6`
rebuilds D6 bit for bit. Outcome fields (`FIELDS_VERSION` D7): ramp slides
floored at 1 cm; no kick `stop_s`; `dropramp.log_slide` defined for every
scene, with tumbling saved as a covariate (never a field, never a mask);
kick tipping stays a mask but is saved and reported the same way.

Five seeds (1234, 1–4; 512 sampled materials each; `scripts/f35_seeds.py`),
mean ± sd, D7 against D6 on the same D7 fields:

| | D6 | D7 |
|---|---|---|
| gate passes (density / friction / restitution) | 5 / 5 / 5 | 5 / 5 / 5 |
| shared ordered pairs | 14 of 56 | 14 of 56 |
| held-box creep at 38.7°, median | 134 ± 10 mm | **4.4 ± 0.0 mm** |
| creep r with restitution | +0.86 ± 0.01 | +0.39 ± 0.05 |
| ramp outcome family, restitution | 0.85 ± 0.02 | **−0.00 ± 0.02** (friction 0.98) |
| kick outcome family, restitution | 0.65 ± 0.03 | **−0.01 ± 0.01** (friction 1.00) |
| push outcome family, restitution | 0.44 ± 0.01 | −0.01 ± 0.01 |
| ramp trajectories (5 mm family), restitution | 0.37 ± 0.03 | 0.18 ± 0.04 |
| kick trajectories (5 mm family), restitution | 0.34 ± 0.06 | 0.13 ± 0.02 |
| kick1.0 recording, friction | 0.67 ± 0.03 | **0.96 ± 0.00** |
| dropramp recording, friction / restitution | 0.88 / 0.98 (38.7°, 0.6 m) | 0.82 ± 0.01 / 0.92 ± 0.00 (20°, 0.3 m) |
| dropramp outcome family, friction / restitution | 0.95 / 1.00 | 0.88 / 0.98 |
| dropramp tumbled; r with μ / e | 90 ± 8; +0.57 / −0.16 | 69 ± 5; −0.17 / **+0.46** |
| kick3.0 tipped; r with μ / e | 17 ± 3; +0.31 / −0.11 | **44 ± 4**; +0.34 / +0.09 |
| strike family (5 mm) | 0.87 / 0.67 / 0.70 | 0.86 / 0.73 / 0.69 |

What D7 buys: the friction targets carry no restitution (outcome level),
and the 1 m/s kick reads friction (0.96, from 0.67). The input trajectories
still carry some (ramp 0.18, kick 0.13 at the family level): residual creep
of 4 mm median that the dead zone hides from the targets only. Harmless
while no target field carries it; stated so it is not forgotten.

What D7 costs, open: (1) the dropramp recording reads friction at
0.82 ± 0.01, a 0.02 margin over a bar that is not frozen; (2) the firmer
grip tips 44 boxes in the 3 m/s kick (17 under D6), grippy ones (r +0.34
with μ), and `kick3.0.log_stop` is masked there, so the kick's friction
field loses the top of the μ range (the same selection problem as the
dropramp, on the target's own axis); (3) D7 changes every recording's
contact, so all reference values taken on D1–D6 are superseded.

**Correction (Mac session, 2026-11-03).** During the pilot the Mac session
withdrew "ramp restitution 0.37 → 0.19, kick 0.31 → 0.13" as unreproducible.
They reproduce exactly at seed 1234: they are the 5 mm family readouts of
the ramp and kick *trajectories* (the code table), not of their outcome
fields. The withdrawal was wrong; the numbers stand with that label.

### FREEZE (2026-11-04): battery D7, outcome fields D7.1, the gate

**Frozen (owner decision 2026-11-04, "apply the recommendation, freeze").**
Code at `0260217`; this entry is the freeze commit. Changing anything
below after this commit is a dated amendment, never a tweak, and no F35
hypothesis may be scored on a battery or gate changed after its result
was seen.

- **Battery:** design D7 (`--design D7`, the generator default): elliptic
  cone, box–ground pair solreffriction `0.01 1`, impratio 1; the drop onto
  the ramp at 20° from 0.3 m; every other probe as in D6.
- **Canonical build: MuJoCo 3.9.0** (the Mac; numpy 2.4.6, Python 3.13);
  `pip install -e ".[f35]"` pins it. Canonical means the MuJoCo version
  only: numpy, Python and platform are recorded, not required (the cloud's
  3.9.0 / x86 / Python 3.11 rerun matches the reference in every cell). Every battery's `manifest.json` now
  records `provenance` (MuJoCo, numpy, Python, platform, git commit and a
  dirty flag), copied into `code_table.json` and `outcomes.json`. D7 is
  the most version-sensitive battery so far: ramp-family sums move by up
  to 49 between 3.9.0 and 3.17.0. The cloud session reproduced every cell
  of the five-seed table below at two decimals on 3.17.0 / x86 and on
  3.9.0 / x86 (note `2026-11-04-0003-d7-build-seeds.md`), so no gate cell
  and no reported mean depends on the version; a 3.17.0 battery is still
  not the canonical one.
- **Outcome fields D7.1** (`FIELDS_VERSION`, 25 fields): drop.e1, drop.e2,
  drop.bounces, drop.settle_s; ramp{05.7…38.7}.log_slide (1 cm dead zone),
  ramp.tan_steepest_still; push{0015…3840}.log_disp; strike.v_ball,
  strike.v_box; scale.sag; tap.e1, tap.e2; kick1.0.log_stop,
  kick3.0.log_stop; dropramp.e1, dropramp.log_slide. **No field is ever
  masked:** tumbling (dropramp) and tipping (kick) are covariates saved in
  `outcomes.npz` with their r against each essence axis, never masks.
- **Gate:** inverse R² (best of CV-α ridge and kNN-5, 5-fold), families
  read whole, 5 mm position noise (0.01 on the up vector), **bar 0.8**.
  Each axis's verdict rests on **one designated pair**, both families
  reading the axis: density **strike + scale**, friction **ramp + kick**,
  restitution **drop + tap** (`VERDICT_PAIRS` in `f35_code_table.py`). The
  other shared pairs, the drop onto the ramp's above all, are reported and
  never load-bearing. The any-pair gate stays as a descriptive readout.
- **Seeds:** 1234, 1, 2, 3, 4; 512 MaterialSampler scenes each
  (`scripts/f35_seeds.py --design D7`).

**The three open items, settled:**
1. *The 3 m/s kick's tip mask* is removed (fields D7.1). The mask cut
   44 ± 4 boxes, the grippy ones (r +0.34 with μ; 32 of the 44 in the top
   friction quartile), so `kick3.0.log_stop` lost the top of the μ range,
   the selection the drop onto the ramp had under D6. The stop distance of
   a tipped box still tracks friction (r −0.95 with μ over all 512 scenes),
   and the kick family reads friction at 0.997 ± 0.001 unmasked, the same
   as masked. Both kick speeds are target fields; the 1 m/s kick never
   tips (0 of 512 on every seed). The cloud's checklist read the
   recommendation as "kick1.0 is the target, kick3.0 stays masked and
   descriptive"; the owner approved the unmasked version, which keeps both
   speeds and drops the last mask.
2. *The dropramp's 0.02 friction margin* no longer decides anything: the
   friction verdict rests on ramp + kick (margin +0.15). **The thinnest
   designated margin is density's**: strike reads it at 0.86 ± 0.02,
   +0.043 over the bar on the worst seed. That margin is stated here
   because it is the cell most likely to fail a future rerun.
3. *The residual restitution in the ramp and kick trajectories* (family
   0.18 / 0.13, from ~4 mm of creep) is disclosed and carried by no target
   field (outcome-field restitution −0.00 / −0.01).

**Frozen reference values** (canonical build, `0260217`, clean tree;
mean ± sd over the five seeds; batteries bit-identical to the `ad82352`
D7 seeds). A rerun of the frozen battery must give the same verdicts and
match every cell at two decimals within ±0.01.

| | density | friction | restitution |
|---|---|---|---|
| **gate verdict**, designated pair | strike+scale 5/5 | ramp+kick 5/5 | drop+tap 5/5 |
| margin over 0.8 (worst seed) | +0.06 ± 0.02 (+0.043) | +0.15 ± 0.00 (+0.152) | +0.15 ± 0.00 (+0.147) |
| *trajectory families, 5 mm:* drop | −0.01 | −0.02 | 0.98 |
| ramp | −0.01 | 0.95 | 0.18 ± 0.04 |
| push | 0.57 ± 0.03 | 0.61 ± 0.03 | 0.08 ± 0.02 |
| strike | 0.86 ± 0.02 | 0.73 ± 0.03 | 0.69 ± 0.03 |
| scale | 1.00 | −0.01 | −0.01 |
| tap | −0.01 | −0.01 | 0.95 |
| kick | −0.01 | 0.99 | 0.13 ± 0.02 |
| dropramp | −0.02 | 0.82 ± 0.01 | 0.92 |
| *outcome families (essence ← fields):* drop | −0.01 | −0.01 | 1.00 |
| ramp | −0.00 | 0.98 | −0.00 ± 0.02 |
| push | 0.94 ± 0.01 | 0.87 ± 0.02 | −0.01 |
| strike | 0.80 ± 0.03 | 0.34 ± 0.05 | 0.26 ± 0.07 |
| scale | 1.00 | −0.01 | −0.01 |
| tap | −0.01 | −0.01 | 1.00 |
| kick | −0.01 | 1.00 | −0.01 |
| dropramp | −0.01 | 0.88 ± 0.01 | 0.98 |

Shared ordered pairs: 14 of 56 on every seed (the same 14). Covariates:
dropramp tumbled 69 ± 5 (r with μ −0.17, e +0.46); kick3.0 tipped 44 ± 4
(μ +0.34, e +0.09); kick1.0 tipped 0. Held-box creep at 38.7°: median
4.38 ± 0.04 mm, max 23 ± 11 mm, r with e +0.39 ± 0.05. Sds under 0.01
are left out of the table.

**Cross-check of the freeze** (cloud session, note
`2026-11-04-2304-freeze-crosscheck.md`, at `6aa0746`): reproduces on
MuJoCo 3.9.0 and 3.17.0 (x86). Same verdicts and margins, every cell
within ±0.01 (identical at two decimals but for one dropramp tumbler and
the creep-max sd), the same set of 14 shared pairs, batteries bit-identical
to its `ad82352` D7 seeds, 28 F35 tests pass on 3.17.0, and the §9 wording
matches the code.

**Not frozen:** R1 (§10), the T1 pixel arm. Its open items stay open: the
framing rule (the W ladder, logged as a D-iteration before computing;
F_px chooses W, F_track grades; the tie-break rule), §10.2's exposure
items and §10.4's pair list. R1 freezes later, in its own commit, on top
of this frozen battery and before any R1 model is trained.


## 10. Amendment R1 (DRAFT): the T1 pixel arm, an attention transformer on uncompressed frame arrays

**Status: DRAFT, 2026-10-27. Not frozen.** ~~R1 freezes together with the
battery and gate (§8 item 3)~~ *Amended 2026-11-04:* the battery and gate
froze first, on their own (§9, FREEZE), which is stricter: R1 can no longer
tune the battery. R1 freezes later, in one commit that adds the frame
generator, the model and `scripts/f35_r1_verdicts.py`, before any R1 model
is trained. Until then everything in §10 can change except the battery,
the outcome fields and the gate.

*Revision 2026-10-27 (draft 2), from the cloud session's review: explicit
exposure and SNR (and a × 1/4 bracket in place of × 1/16); angular noise in
F_px; span masking in training; H2 (b) defined as relative retention; G2
per field; corner split co-reported; mosaic and headlight notes.*

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
  manifest. **Open before the freeze (cloud review, 2026-10-27):** at 96 px,
  whole-path framing leaves the strike ball under 1 px and the box 4–5 px
  on the steep ramp and the 3 m/s kick. The likely fix is a camera that
  covers only the window each target field needs; to be settled with the
  outcome fields. *Superseded by the framing rule, §10.2a (draft
  2026-11-06).*
- **Timing.** 125 fps (one frame every 4 simulator steps of 2 ms), the full
  2 s window: 250 frames per recording. A contact keyframe (~64 ms, §4.2)
  spans 8 frames. Each frame is the exact state at one instant (zero
  exposure time, no motion blur; a disclosed idealization). T0 for R1 is
  recorded at the same 125 fps so frames and states align one to one.
- **Resolution.** 96 × 96 (disk: §10.8).
- **Linear exposure pair.** Each frame is rendered twice, with every light
  scaled by 1 and by 1/4: the headlight's ambient, diffuse and specular
  included, no emissive materials, no skybox. MuJoCo's output is linear in
  light and clips at 255 (§3; confirmed on 3.9.0 and 3.17.0, GPU and
  software renderers alike), so merging the pair (the bright frame where it
  is below 250, otherwise 4 × the dim frame) gives linear radiance up to
  4 × nominal white. The freeze commit includes a test that the merge is
  linear across the seam, with the headlight scaled.
- **Exposure and noise, stated explicitly** (the noise level is the
  channel, §3, so it is a preregistered number, not a by-product):
  - nominal white (255 in the × 1 exposure) = 2 500 e⁻, so **SNR at white
    ≈ 50** (shot-limited); a surface of albedo a sits at a × 2 500 e⁻, and
    the box albedo fixed at the freeze sets the box's SNR, which the
    manifest records;
  - full well 10 000 e⁻ = 4 × nominal white (two stops of highlight
    headroom), clipped above;
  - read noise σ = 3 e⁻; 12-bit ADC over the full well (≈ 2.4 e⁻ per step).
  Values of the order of current global-shutter CMOS. One exposure level
  of the × 1 frame is ≈ 9.8 e⁻, about the shot noise at 100 e⁻, so 8-bit
  quantization is visible only in deep shadow (disclosed).
- **Sensor model**, applied in numpy from a per-scene seed at load time, so
  disk holds only the noiseless exposure pair:
  1. map merged linear radiance to electrons as above;
  2. Poisson shot noise;
  3. Gaussian read noise;
  4. RGGB Bayer mosaic: one colour sample per pixel, one channel;
  5. 12-bit ADC, 0–4095, clipped.
  The model's input is that single-channel 12-bit mosaic: the closest a
  simulator gets to the "raw binary" of a global-shutter CMOS sensor. With
  a grey scene under white light the three channels are equal, so the
  mosaic changes nothing but the format; it is kept for fidelity and
  costs nothing.
- **What stays synthetic:** the 8-bit quantization of each exposure (fine
  steps after merging, but not a true analog signal), MuJoCo's lighting
  model, and zero exposure time. Blender (`generate_blender.py`, linear
  float output) is the route to photoreal raw if R1 is worth extending.

### 10.2a Framing rule (design iteration R1-W1, DRAFT 2026-11-06)

**Logged before any F_px is computed** (the cloud's condition, note
`2026-11-01-d6-crosscheck.md` (c)). Nothing below has been run against a
floor; the only numbers used are the T0 path extents of the frozen battery.

**Why a rule is needed.** T0 path extents, frozen D7 battery, seed 1234,
all 512 scenes, test split included (no outcome and no floor was used, so
this does not peek; disclosed). Per scene, the **extreme displacement from
frame 0 along each axis over the recording** (not the final
displacement); median / p99 / max across scenes:

| family | motion plane | extent |
|---|---|---|
| kick1.0 | x–z | x 0.09 / 0.83 / 0.97 m |
| kick3.0 | x–z | x 0.80 / 4.78 / 4.95 m |
| drop | z only | z −0.66 m in every scene (0.6 m gap, then bounces) |
| strike, box | x–z | x 1.42 / 9.19 / 10.97 m; z up to 0.99 m |
| strike, ball | x–z | x −5.17 … +7.00 m; z up to 2.23 m |
| dropramp | y–z | y −0.26 / −5.28 max downhill; z −0.39 / −2.21 m |

At 96 px a window of W metres gives W / 96 m per pixel: the 0.3 m box spans
28.8 / W px and the 0.1 m strike ball 9.6 / W px. Whole-path windows
(about 6 m for the kick, 17 m for the strike) make the ball sub-pixel.

**The rule:**
1. **Camera.** One fixed camera per input family, square 96 × 96, looking
   along the normal of the family's motion plane (along y for kick, drop
   and strike; along x for the drop onto the ramp), never moving. W is
   the window's width at the motion plane.
2. **Placement, fixed per family before W is chosen.** The box's start
   position sits at 15 % of W from the trailing edge along the motion
   direction (+x for kick and strike, downhill for the drop onto the
   ramp) and centred for the drop. Vertically, the flat ground line sits
   at 15 % of W from the bottom edge; for the drop onto the ramp, which
   descends, the box's start sits at 15 % of W from the top edge.
3. **Ladder.** W ∈ {0.5, 1, 2, 4, 8, 16} m. A rung is admissible only if
   every tracked body spans at least 2 px (box: W ≤ 14.4 m; strike ball:
   W ≤ 4.8 m), so the strike's ladder stops at 4 m.
4. **What leaves the frame is lost.** Per frame and per body, the F_px
   features are the in-plane position (u, v), the up vector and a
   **visibility bit** (centroid inside the window). In view: the §10.5
   noise is added (σ = 0.5 px × W / 96 on u and v; the angular term
   σ_θ = 0.5 px × (W / 96) / 0.15 m as a random rotation of the up
   vector). Out of view: u and v are clipped to the window edge, the up
   vector is zero, the bit is 0, and **no noise** is added (an empty frame
   has no jitter). The bit removes the ambiguity between "left the frame"
   and "sits at the edge" while keeping one array shape per scene.
   Approximation, disclosed: a body is treated as fully known while its
   centroid is inside, though near the edge the frames show only part of
   it. Expected, not tested: σ_θ grows with W (about 0.28 rad at 8 m), so
   readings from boxes that tumble (the drop onto the ramp, the 3 m/s
   kick) fall at large W.
5. **Choosing W without the test split.** For each input family, W* is
   the admissible rung with the highest F_px **by 5-fold CV inside the
   train split**: the §10.4 split seed 0 (iid 80 %: 410 train / 102 test
   scenes) on **battery seed 1234**, the frozen reference battery. The
   test split is never used to choose W; F_px on the test split is then
   reported at W* only. The readout (ridge with α by inner CV, or kNN-5)
   is also chosen inside the train split, then refit on all of it.
6. **Ties and several pairs.** An input family that feeds several R1
   pairs uses the rung that maximises the **minimum** F_px across those
   pairs, and each pair's own F_px at W* is reported too, so a pair whose
   floor falls because of the shared W is visible. Rungs within 0.01 of
   the best count as tied, and the **largest** tied W wins: less
   clipping, so less dependence on the 512 sampled materials.
7. **One W per input family**, the same for the iid and the corner split
   and for battery seeds 1–4, written to the frame manifest with its m/px
   and **each body's size in pixels at W\***. The density H1 report states
   the strike ball's pixel size, so a SHORT on density can be read as
   resolution-limited or not (the strike ball carries density, the
   thinnest frozen margin).

**Predictions for W\*, registered before step (b) computes anything:**
drop 1 m (the 0.75 m release clips slightly, but the bounce apexes stay in
view and the pixel shrinks); kick 2 m (low-friction boxes leave the frame,
but a log-distance target needs their early deceleration more than their
stop); strike 4 m, the largest admissible rung (the cloud also expects the
strike's W\* above the kick's: the ball's fast rebound leaves small
windows within a few frames). Rough confidence: each exact rung 40 %;
strike's W\* ≥ kick's 70 %.

**The graded bar (cloud proposal; owner decision at R1's freeze).** F_px
assumes a 0.5 px tracker; a centroid tracker on a blob 7–58 px across does
much better (the cloud estimates ~0.03 px), so F_px likely understates
what the frames carry. Proposal: **F_px chooses W; F_track grades H1.**
F_track is a classical pipeline on the actual sensor-model frames at W*:
background subtraction against the empty-scene frames, a 5 σ threshold
with σ the analytic per-pixel noise of the empty scene (shot noise at its
expected signal plus 3 e⁻ read noise, in ADC units), connected components,
per body the area centroid (every box has one albedo, so an intensity
weighting adds nothing) and, for the box, the in-plane angle from second
moments; then the same ridge /
kNN-5 readout to the target outcomes. F_track sees only the in-plane
projection, so it can fall below F_px where the box tumbles out of plane;
which way it moves is not predicted. If adopted, H1's thresholds read
against F_track with F_px and F_5mm co-reported, and the §10.7 odds are
re-registered against the new bar at R1's freeze, before any frame
exists.

*Review (cloud, note `2026-11-06-1654-review-r1-w1.md`): sound, no
blocking issue; its answers are folded into rules 4–7 and the F_track
definition above.*

**Order of work:** (a) review of this rule (cloud), done; (b) F_px on the ladder
from T0 only, no frames; (c) the frame generator, with G1 at W*;
(d) F_track; (e) R1's freeze commit.

**R1-W1 results, step (b)** (2026-11-06; `scripts/f35_r1_floor.py`, T0
only, no frames; frozen D7 battery seed 1234, canonical build; CV inside
the 410-scene train split, noise seed 1; the min over pairs is the
dropramp's):

| input → target | CV F_px by W (0.5 / 1 / 2 / 4 / 8 m) | W\* | test F_px at W\* | F_5mm |
|---|---|---|---|---|
| kick → ramp | .936 / .917 / .935 / .920 / .920 | 2 m | 0.946 | 0.950 |
| drop → tap | .931 / .988 / .987 / .981 / .974 | 4 m | 0.985 | 0.991 |
| strike → scale | .850 / .889 / .871 / .879 / — | 1 m | 0.878 | 0.850 |
| dropramp → 4 targets (min) | .805 / .781 / .775 / .707 / .661 | 0.5 m | 0.82–0.97 | 0.79–0.91 |

**Predictions scored:** kick 2 m hit; drop 1 m missed (1 m had the best
CV score, but 4 m is within the 0.01 tie and the largest tied rung wins);
strike 4 m missed (1 m); strike's W\* ≥ kick's (70 %) missed.

**Three problems found, none a code bug:**
1. **W\* depends on the noise draw.** Repeating the choice with noise
   seeds 1–5 (train-split CV only): drop 4 m in 5 of 5, kick 2 m in 4 of 5,
   strike 1 / 4 / 4 / 2 / 4 m, dropramp 0.5 m in 3 of 5. The strike's
   rungs sit within ~0.02 of each other, the size of one noise draw's
   effect, so the rule picks among near-equals by chance.
2. **The corner split's per-field R² is undefined for the ramp target.**
   In the top 20 % of friction, four of the six ramp slides (05.7–24.2°)
   have zero variance (every grippy box sits in the 1 cm dead zone) and
   two are near zero (31.0°: 0.053 × the full-population sd; 38.7°:
   0.026). *Corrected 2026-11-06: the first version of this entry said
   five zero and one at 0.05, a misreading of my own diagnostic; the
   cloud's count is right.* R² on the corner's own variance divides
   by ~0, giving −189 (kick → ramp) and −508 / −698 (dropramp → ramp /
   drop). H1's corner co-report would hit the same problem for the model.
3. **A noiseless reference is not meaningful with standardized columns:**
   kick → ramp read 0.33 with no noise against 0.95 at 5 mm, because
   standardizing near-constant columns lets sub-micron variation dominate
   kNN. Dropped from the script; F_5mm is the reference.

**Proposed revision R1-W2, for review (not adopted):** (a) choose W\* on
the CV score averaged over five noise draws (seeds 1–5) instead of one;
(b) score the corner split per field against the **train split's**
variance of that field (equivalently, MSE in train-standardized units),
for F_px and for the model alike, so a field that saturates in the corner
counts as easy, not as undefined. *Disclosed:* the test-split F_px at
R1-W1's W\* (the table) was seen before this proposal; no test score at
any other rung was computed, and both changes are motivated by
train-split CV and by the corner's target variances alone.

**Rule-4 leak, found by the cloud's code read and fixed** (note
`2026-11-06-1711-r1-floor-crosscheck-review.md`). Out of view, R1-W1's
code clipped each coordinate to the window, so the coordinate along the
edge the body did not cross stayed exact and noiseless: a box leaving
through the leading edge still reported its height in every later frame.
The frames show nothing once a body is gone, so this broke rule 4 as
written. Fix: out of view **both** coordinates are a constant (0), with
the bit, a zero up vector and no noise. This is a correctness fix, not a
tuning choice; the cloud tested it first, and the Mac reproduces its
numbers. Under R1-W1 (one noise draw) **no W\* changes** (kick 2, drop 4,
strike 1, dropramp 0.5 m), so the registered predictions stand as
scored. Test F_px at W\*, before → after: kick → ramp 0.946 → 0.953;
drop → tap 0.985 → 0.985; strike → scale 0.878 → 0.884; dropramp → ramp
0.851 → **0.823**, → kick 0.919 → 0.917, → drop 0.819 → **0.784**, → tap
0.966 → 0.964. The leak inflated the dropramp's floors and drove most of
the strike's W\* instability: with the fix, the strike's W\* over noise
seeds 1–5 is 1 / 1 / 1 / 2 / 1 m (was 1 / 4 / 4 / 2 / 4).

**Why the strike's F_px beats F_5mm (cloud ablation, not a leak):**
F_5mm puts noise on the out-of-plane coordinate, which for the strike is
nearly pure noise, and windowing drops noise from uninformative
out-of-view frames while the bit gives the exit timing, which a camera
also shows. F_5mm is therefore not a strict ceiling for F_px. The script
now co-reports **F_5mm in-plane** ((u, v) and up at 5 mm), the
like-for-like reference: strike → scale 0.856 against F_px 0.884.

**Cloud review of R1-W2:** adopt (a), keeping the 0.01 tie (per-rung sd
over five draws is 0.000–0.013, so a five-draw mean is good to about
0.002–0.006); prefer (b) over dropping saturated fields (makes the field
set split-dependent) or pooling R² (dominated by whichever fields still
vary), and call it "skill vs train variance", not R². **Averaging order (R1-W2 (a)):** each
pair's CV score is averaged over the five draws first, then the minimum
across pairs is taken: the min of five-draw estimates, not the mean of
per-draw minima, which is biased low (the cloud's per-draw order gave
0.790 for the dropramp at 0.5 m against 0.792). **R1-W2 with the
fix, as it would choose:** kick 2 m, drop 4 m, strike 1 m (0.887 vs
0.875 at 2 m), dropramp **1 m** (0.792 vs 0.784 at 1 m: inside the tie,
the larger rung wins); test F_px at those W\*: 0.953, 0.985, 0.884, and
dropramp 0.879 / 0.933 / 0.777 / 0.958. **Status: implemented as an
option (`--noise-seeds 1 2 3 4 5`), not adopted; the owner decides.**

**New, from the corner scored as skill vs train variance** (the
co-report R1-W2 (b) proposes; it is now computed beside the R² either
way): defined everywhere, as intended. The grippy ramp slides that
saturate score about 1.0, which is the honest reading. Two cells show the
floor itself failing to extrapolate: **strike → scale −18.4** (the
densest 20 % lie far outside the training range of the scale's sag, and
neither ridge nor kNN on the strike's trajectory extrapolates there; the
recoil-speed nonlinearity is a likely cause, post hoc, not measured), and
the 38.7° ramp slide at −0.98. A model's corner score on the density
pair cannot be read against a floor that is itself at −18; this has to
be settled before R1's freeze.

**Density corner diagnosed** (cloud note `2026-11-06-1717-rule4-fix-
density-corner.md`; the Mac reproduces the load-bearing numbers): **both
range and missing information.** The densest 20 % weigh 70–216 kg against
1.4–69.7 kg in training, and their mean sag sits 6.3 train-sd above the
training mean. Every readout trained on the light 80 % misses by 3.7–4.2
train-sd of bias, with ridge, kNN, a log-sag target or the outcome fields
alike. And the information thins out: with the 17 kg reference ball, a
heavy box barely recoils (v_box 0–0.32 m/s in the corner against up to
1.71 m/s below it; r(v_box, 1/(17 + M)) falls from 0.94 overall to 0.48
in the corner), so even trained on heavy boxes, 5-fold CV inside the
corner reads sag at only 0.35 at pixel precision (kNN; ridge 0.30) and
0.64 at 5 mm. The 1/(M + m) guess above is borne out. This is a property
of the **frozen battery's density carrier**, not of R1; a heavier
reference ball or a second strike speed would move the saturation, for
any future battery only.

**Proposed corner co-report rule (cloud; for the owner, not adopted):**
always report the floor's corner skill (vs train variance) beside the
model's for every pair; read a corner *verdict* only where the floor's
corner skill is ≥ 0.5, and label other pairs **floor-limited**, with the
model's number descriptive only; for readable pairs, report model minus
floor in skill units. The threshold and the W\* rule are fixed before the
first frame. At R1-W1's W\*: readable kick → ramp (0.66), drop → tap
(0.74), dropramp → kick (0.51) and → tap (0.81); floor-limited strike →
scale (−18.4), dropramp → ramp (0.02) and → drop (−0.72). Consequence:
**density has no readable corner pair at R1**; its corner question is
answered "floor-limited", not "fails". Disclosed: the 0.5 threshold is
proposed after these floor values were seen, and one readable pair
(dropramp → kick, 0.51) sits just above it; if adopted, the doc should
say the threshold was chosen with the floor in view.

**Step (c): the frame generator and G1** (2026-11-07;
`scripts/f35_r1_frames.py`, commit 6316b22; frozen D7 battery seed 1234,
all 512 scenes, canonical build; the drop onto the ramp at both candidate
W\*, 0.5 m under R1-W1 and 1 m under R1-W2, since that choice is open).
The generator re-simulates each R1 input recording with the battery's own
`run_*` code, unchanged: two hooks add a `<camera>` (no body, no physics)
and make the recorder render at every recorded step. Choices made here,
all reviewable before R1's freeze:
- **Orthographic camera**, `fovy` = W, looking along the motion plane's
  normal from 20 m. A perspective camera would move a body that leaves
  the plane (the strike ball's small y drift, a tumbling box's far
  corner); orthographic makes the window exactly the rule's W at every
  depth.
- **Anchor:** each family's nominal start position. The floor anchors on
  each scene's frame-0 position; on the anchoring axes (x for the flat
  families, whose height is absolute; y and z for the ramp) the two
  differ by 0 for the flat families and at most 0.0003 mm for the drop
  onto the ramp (its first 2 ms of fall), far under 0.001 px.
- **One box finish:** albedo 0.5, matte (specular, shininess and
  reflectance 0), set on the compiled model, so physics cannot change.
  The re-simulated T0 is **bit-identical** to the stored battery in all
  6 recordings × 512 scenes. The repo's lighting is kept (headlight
  0.6 + ambient 0.4, a spotlight from 3 m overhead).
- **What the frames look like.** Motion stays in the camera's plane and
  the box rotates about the view axis, so the camera always sees the same
  face, lit head-on: every box pixel is 128 in the × 1 exposure, a flat
  silhouette whose outline carries position and in-plane angle. The
  ground is seen edge-on, so it is invisible, and the background is
  black. The strike ball (albedo 0.2, the battery's) shows a small
  highlight. **No pixel in any frame reaches the 250 knee**, so the HDR
  merge never uses the dim exposure in this battery; the merge is still
  tested on a synthetic sweep across the seam (below). Box SNR is
  therefore ≈ 35 (1 255 e⁻).
- **Storage:** one grey channel per exposure (R = G = B is checked on
  every frame; the RGGB mosaic of a grey scene is the identity), uint8,
  `.npz` compressed: about 100 MB for all six family × W sets (flat grey
  on black compresses very well; §10.8's 2.4 GB per family was the
  uncompressed size). The empty scene per recording, for F_track's
  background, is stored with the frames (identical across scenes,
  checked). Rendering took 5.5 min on 16 processes.

**G1, defined for partial views.** §10.6 states G1 as "box centroid
from a segmentation render vs projected T0 within 0.5 px RMS over all
frames". Two things made "all frames" impossible as written: the rule's
15 % placement puts the box's centre 0.15 × W from the edge, which is
inside the box's own half-width (0.15 m) whenever W < 2 m, so at 1 m and
0.5 m the box starts partly out of frame; and during the strike, the
soft contact lets the 17 kg ball sink up to ~8 cm into the box, which
hides it. G1 therefore scores the segmentation centroid against the
centroid of the **projected silhouette clipped to the image**, which
equals the projected T0 position whenever the body is fully in view, on
frames where the body is not occluded and at least 25 % of it is
visible. Both definitions are reported. A diagnostic, not graded: the
same silhouette sampled at pixel centres against the render, which
separates pixelization from camera or timing error.

| family @ W | recording | G1 RMS px (max) | fully in view only | frames scored | raster RMS px | pixels differing |
|---|---|---|---|---|---|---|
| kick @ 2 m | kick1.0 | 0.182 (0.39) | 0.182 | 128 000 / 128 000 | 0.029 | 0.03 % |
| | kick3.0 | 0.162 (0.53) | 0.162 | 112 143 | 0.022 | 0.03 % |
| drop @ 4 m | drop | 0.186 (0.40) | 0.186 | 128 000 | 0.041 | 0.09 % |
| strike @ 1 m | box | 0.301 (0.61) | 0.309 | 69 262 | 0.007 | 0.01 % |
| | ball | 0.092 (0.30) | 0.090 | 28 089 (7 975 occluded) | 0.043 | 0.57 % |
| dropramp @ 0.5 m | box | 0.021 (0.33) | 0.027 (443 frames) | 103 410 | 0.004 | 0.00 % |
| dropramp @ 1 m | box | 0.036 (0.45) | 0.037 | 114 551 | 0.005 | 0.01 % |

**G1 passes in every recording.** The camera geometry is exact: sampled
at pixel centres, the predicted silhouette reproduces the render's
centroid to 0.004–0.043 px. The 0.16–0.30 px G1 residual is pixelization
of axis-aligned faces: a resting box keeps the same sub-pixel offset for
many frames, so the error does not average out (the strike box, at rest
for most of its in-view frames and 29 px across, has the largest). The
tumbling dropramp box, never axis-aligned, sits at 0.02–0.04 px. Three
strike-ball frames (of 128 000) show a presence mismatch: slivers under
2 px² at the image edge, where MuJoCo's faceted sphere and the 96-gon
prediction differ.

**New, and relevant to the open W choice for the drop onto the ramp:
how much of the box the frames show.** Rule 4 treats a body as fully
known while its centroid is in view (a disclosed approximation). Measured
on the frames, over frames whose visibility bit is 1:

| family @ W | box fully in view | visible area, median / p10 |
|---|---|---|
| kick @ 2 m (kick3.0) | 97.3 % | 1.00 / 1.00 |
| drop @ 4 m | 100 % | 1.00 / 1.00 |
| strike @ 1 m | 93.2 % | 1.00 / 1.00 |
| dropramp @ 0.5 m | **0.4 %** | **0.77 / 0.60** |
| dropramp @ 1 m | 74.8 % | 1.00 / 0.97 |

At 0.5 m the box is 58 px across, 60 % of the frame, and almost never
wholly in it: R1-W1's W\* = 0.5 m for the drop onto the ramp rests on
rule 4's approximation where the frames honour it least. This does not
choose between R1-W1 and R1-W2 by itself (F_px's number at 0.5 m is what
a tracker of the centre would get, and a tracker can infer the centre of
a cut square from its visible corners); F_track, step (d), measures it
directly at both rungs.

**Tests** (`tests/test_f35_r1_frames.py`, 15): the camera maps the
rule's window corners to the image corners and ignores out-of-plane
offsets; rule 2's placement; clipping, area, centroid and raster
geometry; a box silhouette's centroid is its projected centre under
in-plane and out-of-plane rotation; the merge rule; **the merge is linear
across the 250 seam with the headlight scaled** (a matte plate under a
light sweep from × 0.6 to × 2.2, the × 1 frame clipping above × 1.1: one
slope on both sides, residual ≤ 1 count below the seam and ≤ 4.5 above,
i.e. one 8-bit step of the dim frame; MuJoCo's 8-bit output is good to
about one step, not half); shot and read noise and the ADC step
reproduce the stated variance; full-well clipping; seeded determinism;
the grey check; G1's frame selection.

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
- **Masking in training**, identically in every run, so the erasure tests
  (§10.6) are in distribution at evaluation. Each training example gets,
  with equal probability, either i.i.d. frame masking (each frame token
  replaced by a learned mask token with p = 0.25) or one contiguous span of
  8–80 frames masked (length uniform). H2 (a) masks a contiguous block of
  about 44 frames, which i.i.d. masking alone would make out of
  distribution.
- **Loss.** MSE on the target family's outcomes, each field standardized on
  train statistics, never a raw MSE in which one field dominates (the F31a
  head-collapse lesson). Outcome fields per family are defined in the freeze
  commit.

### 10.4 Pairs, splits, seeds

- **Pairs:** the frozen battery's shared pairs. At D4: kick → ramp
  (friction, primary: the faintest keyframe), drop → tap (restitution),
  strike → scale (density). The multi-axis input question (§9) may add one
  pair at the freeze. *Note 2026-11-04:* the frozen battery (D7) shares 14
  ordered pairs, including the drop onto the ramp's six; the gate's
  designated pairs are the three above. R1's pair list is decided at R1's
  own freeze.
- **Split:** one fixed iid 80/20 split by scene (seed 0), the same for
  every pair; primary. R1's first question is whether pixels can match the
  state floor at all, which is an interpolation question. **Corner split,
  co-reported** (the project default, "splits with teeth"): per pair, the
  top 20 % of the shared axis held out, with its own F_px. A model that
  reaches the floor on iid and falls short on the corner has learned the
  training range, not the physics. **Seeds:** 0–4 per pair.

### 10.5 The bar: the T0 oracle floor at pixel precision

For each pair, **F_px** = best of ridge and kNN-5 from the input family's
T0 trajectory to the target outcomes, fit on the train split, scored on the
test split. T0 is degraded to what the camera can see: position noise
σ = 0.5 px × that camera's metres per pixel, and orientation noise of the
same pixel precision, σ_θ = 0.5 px × (m/px) / 0.15 m (the cube's half
size), applied as a random rotation of the up vector. Without the angular
term the bar would be inflated wherever the box tips or bounces. F_5mm (σ = 5 mm, the §5 gate noise) is co-reported. A pixel model
cannot beat exact state, except through noise; F_px is what a perfect
tracker at the camera's resolution would get with a linear or
nearest-neighbour readout.

### 10.6 Gates, hypotheses and frozen rules

**Gates:**
- G1 apparatus: the box centroid from a segmentation render matches the
  projected T0 position within 0.5 px RMS over all frames of every
  recording (checks frame timing and camera).
- G2 head health, per run and **per standardized field**: train gain
  ≥ 1.5 overall, and for every field the floor can predict (per-field F_px
  R² ≥ 0.3), the spread of the model's train predictions across scenes
  ≥ 0.5 × that field's target spread. A pooled check can pass while a
  small block is collapsed: F31a's drop block was, hidden under the
  dominant tilt block. A failing run is reported and withheld.
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
  in S. In-betweens carry nothing. Co-reported: the cost of masking the
  same number of i.i.d. random frames from the whole recording.
- (b) Keeping a fixed random 25 % of frames (the same frame indices for
  every scene), each pair's **retention** is ρ = S(25 % kept) / S(all
  frames), relative to its own full-frame S because the pairs start from
  different S. Holds if ρ(restitution) − ρ(friction) ≥ 0.10 on the seed
  mean and in at least 4 of 5 seeds. Restitution is redrawn at every
  bounce; friction is one faint keyframe.

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
