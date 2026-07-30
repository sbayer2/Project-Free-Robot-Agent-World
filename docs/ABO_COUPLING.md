# Preregistered: the coupling-strength ladder on real geometry (F22)

> Status: **PREREGISTERED 2026-07-30**, frozen before the runs.
> Amendments are dated additions, not edits. Staged by cost: a synthetic pilot
> runs first and gates whether the ABO arms are built at all.
>
> **This is not a reality test.** Physics is authored on top of real meshes and
> overrides ABO's listed weights. Nothing here licenses a claim about real-world
> materials. See "Honest limits".

---

## 1. Destination — the question

> **When a shared latent fails to route material essence into behavior, is the
> limit the world or the architecture?**

Every result from F18 to F21 is consistent with two mutually exclusive readings,
and the project has never separated them:

- **World reading.** The authored appearance↔physics coupling in `materials.py`
  is too faint to exploit. A stronger coupling would be learned. The instrument
  is fine; the substrate is impoverished.
- **Architecture reading.** The encoder→behavior-head path cannot route material
  essence at all, at any coupling strength. A stronger coupling would change
  nothing. The substrate is fine; the instrument cannot do this.

F21 is usually read as supporting the world reading, but it does not establish
it. F21 showed that making the *existing* coupling more available and more
legible does not move behavior. It never raised the coupling's **strength**, so
it cannot distinguish "too faint to exploit" from "cannot be exploited".

F22 turns coupling strength into a dial and reads the answer off the slope.

## 2. Why it matters

The two readings prescribe opposite next moves, and the project cannot choose
between them today:

| If the answer is | Then Option B (stronger synthetic world) is | And the honest headline becomes |
|---|---|---|
| **World** | worth building, with a *calibrated* target strength read off this ladder rather than guessed | "a shared latent learns essence only above coupling strength α\*, which natural worlds may not reach" |
| **Architecture** | a waste — a stronger world would be learned no better | "this shared-latent architecture cannot route essence to behavior, at any coupling strength" |

The architecture answer is the stronger scientific claim and the cheaper
outcome: it closes the project instead of opening a new dataset-building phase.
Option A (accept the F18–F21 negative) is currently stated as a claim about
*this world*; if the architecture reading holds, the same evidence supports a
much broader and more useful claim.

Neither reading can be reached by collecting more data or training longer. Only
by varying coupling strength while holding everything else fixed.

## 3. Map — where this sits in F1–F21

What is already established, and what each result does *not* settle:

- **F13** — a shared latent learns a small real coupling (+0.146 behavior
  coherence, 20 seeds). *Does not say* whether that coupling is essence or shape.
- **F17** — a 1-trit FSQ bottleneck nearly doubles learned coherence (+0.27 vs
  +0.146) with prediction intact: **scarcity manufactures unity**. *Establishes
  that architecture is a coupling lever* — measured only on synthetic data, and
  never tested for whether it raises prediction *gain* rather than coherence.
- **F18** — the gain is shape, not essence: model 1.413× vs shape-only oracle
  1.331× (CI [1.01, 1.79]), ceiling 2.311×; ~8% of reachable signal extracted.
  *Establishes the primary metric* used here: Δ = gain − shape-oracle gain.
- **F19/F20** — the latent retains color (R² ≈ 0.8) but not the contact-physics
  channels; roughness reachability 0.16, rising only to 0.37 under oblique
  256 px. Forcing channels into `z` moved behavior 1.37 → 1.46 only.
  *Establishes that retention is measurable per channel* — the basis for Arm R
  vs Arm U below.
- **F21** — neither availability (noise ↓) nor legibility (256 px + oblique)
  moves behavior off ~1.5; the oracle−model gap *widened* 0.80 → 1.01 when noise
  was removed. *Establishes that the pipeline is not the barrier* — and leaves
  exactly the world-vs-architecture ambiguity F22 attacks.
- **F14/F16** — the ABO reality arm is **VOID**, not negative: with friction and
  restitution assumed constant the probe battery is nearly mass-blind (free fall
  is mass-invariant), so the one real label barely enters the outcomes; held-out
  gain 0.96. *Establishes why ABO cannot simply be "re-pointed"* at the question
  without authored physics.

The route: F18 gives the metric, F19/F20 give the channel axis, F17 gives the
architecture axis, F21 gives the ambiguity, F14 gives the constraint that forces
authored physics. F22 is the composition of all five, not an extension of F21.

## 4. Home port — what we return to, and what is genuinely new

Reused unchanged (this is not a new apparatus):

- `sample.json` schema v2 and the probe battery — drop/tilt/push, 21 fields.
- `generate_abo.py` → `generate_gso.py` mesh path; the 437 prepared ABO objects
  already on disk (`data/pm_abo/manifest.json`: 437 scenes).
- `scripts/oracle_ceiling.py` for the shape-only oracle and the gain metric.
- `splits.py` extrapolation-corner logic; `models/train.py`; the FSQ bottleneck
  (`ModelConfig.latent_trits`, already in all three backends).

New, and the entire delta:

1. A function from **rendered appearance** to the hidden factors (see §7), so a
   coupling exists on meshes that cannot be repainted.
2. A strength dial `α` blending that function with randomness.
3. Per-object `friction`/`restitution` sampling — the fields already exist on
   `MeshAsset` (`generate_mujoco.py:132-133`); they are currently pinned to the
   constants `ASSUMED_FRICTION = 0.5` / `ASSUMED_RESTITUTION = 0.3`.

**The design trap, recorded so it is not re-entered.** The naive version of this
experiment — sample material properties randomly and assign them to real meshes
— produces **zero coupling by construction**. A scanned texture has no
relationship to a randomly drawn friction, so the model is asked to learn
something that does not exist, and the resulting null is uninterpretable. The
arrow must be inverted: read factors *out of* the appearance, then generate
physics from them. Any future variant of this experiment must preserve that
inversion.

## 5. Anchor — what holds this fixed

- **Primary metric is Δ = model_gain − shape_oracle_gain**, never raw gain.
  F18's lesson: raw gain is mostly shape, and on ABO's real geometry it will be
  more so (P5).
- **P1 is a hard gate.** If the α = 0 control shows signal, the descriptor
  derivation leaks and every downstream arm is void. Nothing is interpreted
  until P1 passes.
- **VOID discipline (from F14).** Coherence is interpreted only where prediction
  gain is established. An arm with Δ ≤ 0.15 yields no coherence claim — VOID,
  not negative.
- **Multi-seed only.** 3 seeds minimum; sign-consistency across seeds is
  required for any claim. Single-seed cells are smoke tests.
- **Falsifiers are stated per prediction below.** A prediction with no stated
  falsifier is not a prediction.
- **Provenance labels are mandatory.** Every `sample.json` records that physics
  is authored and that listed mass was overridden, retained as analysis-only
  metadata.

## 6. Alternative routes considered

Honest accounting: **most of F22's decisive power does not require ABO at all.**

| Route | Cost | What it answers | Why not (or: why also) |
|---|---|---|---|
| **A. Synthetic coupling ladder** — turn the α dial inside `materials.py` on `pm_big`, no ABO | Lowest. Regenerate + retrain only | World-vs-architecture, cleanly, with full control of appearance | **This is the pilot (§7.0), and it may settle the question alone.** It cannot address the 5-primitive shape confound, which is F18's biggest limitation |
| **B. FSQ sweep on existing `pm_big`** | Very low — checkpoints partly exist | Whether architecture raises *gain*, not just coherence (F17's untested half) | Necessary but not sufficient: without a strength dial it cannot separate the two readings |
| **C. WMGym-style regret protocol on existing checkpoints** | Low — no new data at all; the behavior head is already action-conditioned | Decision utility: can the latent *rank* interventions? | Answers a **different** question. Complementary, not a substitute. Worth doing regardless |
| **D. Mass-sensitive probe redesign** | High — new probe physics | Repairs F14's VOID for a genuine reality test | The right move only if the goal is reality. F22 explicitly is not |
| **E. Do nothing (Option A)** | Zero | — | Legitimate. But it leaves the world-vs-architecture ambiguity permanently unresolved, and states the negative more weakly than the evidence may support |

**Consequence for staging.** Route A is strictly cheaper than F22 and answers
the primary question. ABO buys exactly one thing: **real shape and appearance
diversity**, which matters for P5 and for the standing criticism that every
finding rests on five primitives. That is worth paying for *only if the
synthetic pilot leaves the question open*.

Therefore F22 runs in two stages, and **stage 2 is not authorized by this
document** — it requires a dated amendment recording the pilot's outcome.

## 7. Design

### 7.0 Stage 1 — synthetic pilot (runs first, gates everything)

On `pm_big`'s generator, with appearance fully under our control:

- **α ∈ {0, 0.67, 1.0}** on the existing `materials.py` factor→appearance map.
- **Bottleneck ∈ {continuous, FSQ k=1}**.
- 3 seeds. **12 runs.**

**Gate:** if Δ at (α = 1.0, FSQ k=1) is ≈ 0 — the architecture reading — the
question is answered and stage 2 does not run. Record as F22a and stop.

If Δ rises with α, the world reading holds; stage 2 then asks whether it
survives real geometry.

### 7.1 Stage 2 — ABO arms (only on a passing pilot)

**Coupling channel** — chosen by *measured* retention (F19/F20), not intuition:

- **Arm R (retained):** factors from mean luminance. The latent demonstrably
  carries this channel at R² ≈ 0.8.
- **Arm U (unretained):** factors from high-frequency texture energy, a
  roughness proxy. The latent carries this poorly (0.16–0.37).

**Factor derivation**, computed from rendered views only — nothing privileged:

| Factor | Arm R descriptor | Arm U descriptor |
|---|---|---|
| `heaviness` | mean luminance (darker → heavier) | HF energy, low band |
| `grip` | luminance variance | HF energy, mid band |
| `hardness` | highlight spread (p95 − median) | HF energy, high band |

Then `_physics_from_factors()` unchanged: density ← heaviness, friction ← grip,
restitution ← hardness.

**Strength dial:** `factor = α · f(appearance) + (1 − α) · random`,
α ∈ {0, 0.67, 1.0}.

**Bottleneck:** continuous vs FSQ k=1. **3 seeds → 36 runs.**

**Holdout:** the **factor corner** (extrapolation logic, `splits.py:158`), not
category holdout. F14 established that ABO category holdout does not generalize
from ~4 objects/category; since F22 authors the factors, the corner is available
and is the stronger test.

**Efficiency:** α and channel change *physics only*, never appearance. One
render pass serves all arms; only the MuJoCo probe pass is repeated.

### 7.2 The decisive cell

**Arm R, α = 1.0, FSQ k=1.** Physics fully determined by a channel the latent
provably retains, under the architecture that maximally forces cross-head
sharing. Every lever the project has, at maximum, simultaneously.

## 8. Predictions

Frozen before any run. Each states its falsifier.

**P1 — control behaves (HARD GATE).** At α = 0, |Δ| < 0.05 in both channels.
*Falsified if* Δ > 0.10, indicating the descriptors leak information that is
supposed to be random. **On falsification: stop; fix the derivation; nothing
downstream is interpretable.**

**P2 — retention predicts learnability.** At matched α, Δ(Arm R) > Δ(Arm U),
sign-consistent in ≥ 2 of 3 seeds.
*Falsified if* Arm U ≥ Arm R. That would contradict F19/F20's retention
measurements and would mean retention does not predict routability.

**P3 — the threshold is high.** Δ > 0.15 first appears at **α ≥ 0.67** in Arm R;
Arm U may never clear it.
*Rationale:* F21 established the existing (effectively low-α) coupling is
unlearnable.
*Falsified if* α = 0.33-equivalent or lower already clears 0.15 — which would be
good news, relocating F21's failure to something ABO-shape-specific.

**P4 — FSQ amplifies (replication + extension).** At matched (channel, α),
FSQ k=1 shows higher learned coherence than continuous, replicating F17
(+0.27 vs +0.146); **and** it raises Δ, which F17 never established.
*Falsified if* no coherence advantage appears — which bounds F17's generality to
synthetic worlds and is itself a publishable correction.

**P5 — the shape oracle rises (RISK prediction).** On 437 real ABO shapes,
`shape_oracle_gain` exceeds pm_big's **1.331×**.
*Rationale:* real product geometry carries far more outcome-relevant information
than 5 primitives (~1.6 bits of shape).
*Consequence if true:* essence headroom **shrinks** and detection gets harder.
Preregistering this forbids claiming surprise later.
*Falsified if* `shape_oracle_gain` < 1.331×. Note the sample is thin (437
objects) — report the CI, do not lean on the point estimate.

**P6 — coherence gate.** Coherence is measured only in arms where Δ > 0.15. If
no arm clears it, the coherence arm is **VOID**, not negative.

## 9. What each outcome means

| Result | Reading | Consequence |
|---|---|---|
| P1 fails | Instrument error | Fix descriptors; no claim |
| Δ ≈ 0 at the decisive cell | **Architecture limit** | Strongest available outcome. F18–F21 re-reads as an architecture result, not a world result. Option B is refuted before it is built; the project closes with a broader claim |
| Δ rises with α, threshold α ≥ 0.67 | **World limit, calibrated** | Option B becomes worth building *with a target strength*, not a guess |
| Δ rises only with FSQ k=1 | Unity must be architected | Confirms and extends F17; the design principle generalizes past synthetic data |
| P5 holds and Δ stays small | Shape swamps essence on real geometry | Argues for shape-controlled probes before any further essence work |

## 10. Cost

- Stage 1 pilot: 12 runs, ~87 s/seed at 128 px → well under an hour of training,
  plus regeneration of `pm_big` variants (physics-only re-sim where possible).
- Stage 2: 36 runs, ~52 min training; one render pass over 437 objects plus six
  probe passes. Measure the render pass before committing.

## 11. Honest limits (carried, not hidden)

- **Not a reality test.** Physics is authored and overrides ABO's listed weight.
  The real-object question raised in F14/F16 remains VOID and untouched by F22.
- **437 objects** is a thin sample for a 6-cell design; every claim carries its CI.
- **Arm U's descriptor is a proxy**, not a measurement of surface roughness.
  A null in Arm U is weak evidence about roughness specifically.
- **The coupling is still authored**, so this inherits the standing
  "Blender's eigenvector" limitation. What changes is that the coupling's
  *strength* is now a controlled variable rather than a fixed property.
- A stage-2 result cannot be compared directly to `pm_big` numbers: geometry,
  appearance, object count and holdout kind all differ.

## 12. Reproduce

To be filled in on execution, following the F21 pattern: exact generator
invocations per arm, checkpoint paths, and the eval script. Nothing in §8 may be
edited after the first run — amendments are dated additions.
