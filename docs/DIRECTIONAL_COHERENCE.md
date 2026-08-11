# F28 preregistration — directional (content-matched) coherence: do the z-directions that write material into the render move the matched physics readouts?

*Preregistered 2026-08-04, before any directional-coherence number has been
computed on any checkpoint, anywhere. Decision rules are committed as
`scripts/f28_verdicts.py` in the same commit as this document. This is a
**measurement-only** experiment: zero training, zero data generation — every
input already exists under `runs/f27b/` and `data/pm_f27_*`. The measure
driver (`scripts/f28_measure.py`) will be committed before any measurement
runs and may not alter the constants or rules frozen here.*

---

## 1. Why this experiment exists

F27b killed the magnitude-coherence metric's content-reading: +0.434 learned
coherence in a world where the two heads share **zero** world-content
(disjoint control +0.0001). Joint training on a shared substrate couples head
response *magnitudes* regardless of content, so every learned-coherence
number since F6 certifies substrate entanglement, not "one physical essence,
two projections."

F28 builds the successor recorded (not preregistered) in FINDINGS F27:
replace "do response magnitudes correlate over random z-directions" with "do
the **specific** z-directions that write a material channel into the render
also move the **matched** physics readout, with the committed sign." The
designed property: in the ctrl world (α = 0), appearance is decoy-driven, so
no z-direction can carry matched content — the metric should read ≈ 0 there
*by construction*, restoring the discriminative power the magnitude metric
never had. Whether it actually does is hypothesis H1, the kill switch:
substrate entanglement might contaminate directions too, and finding that out
is half the point.

## 2. The metric

For one trained model on one arm's dataset (encode convention identical to
the coherence harness; train/test splits as recorded in the dataset):

1. **Render-side directions** (global, one per committed channel). Encode
   the train split → z; the direction `d_c` for appearance channel *c* is the
   weight vector (intercept excluded, normalized to unit length) of
   `oracle_ceiling.ridge_fit_predict` (α = 1e-3) fit z → channel-*c* target
   on the train split. Probe quality is gated by train-split 5-fold CV R²
   (`probe_appearance.kfold_r2`).
2. **Physics-side sensitivities** (per scene). Jacobian rows of the trained
   **essence head**, ∇_z e_p for p ∈ (density, friction, restitution) — the
   F19 `ESSENCE_AXES` order — at each test-split z, via MLX `vjp`.
3. **Coordinates.** All vectors are expressed in the standardized
   coordinates of the space they inhabit: per-dim mean/std from that model's
   own train-split z. Ridge weights are already in standardized-X
   coordinates; raw-z Jacobians are converted by elementwise multiplication
   with the train-split per-dim std. A unit test in the measure commit must
   verify this composition on a synthetic case.
4. **Alignment matrix.** M[c,p] = mean over test scenes of the signed
   cosine cos(d_c, ∇_z e_p).
5. **Directional coherence.**

       D = (1/3) · Σ_matched  s_cp · M[c,p]

   over the three matched cells of §3, with the committed signs s_cp.
   The six mismatched cells are co-reported as `mismatched_abs`
   (mean |M[c,p]|), a diagnostic for subspace-sharing/entanglement — **not**
   part of D.

**Why matched-minus-nothing rather than matched-minus-mismatched:** any
content-free mechanism (shared active subspace, head entanglement) inflates
matched and mismatched cells *alike*, with signs that are arbitrary with
respect to the committed pairing — so under the content-free null,
E[D] = 0 without subtracting the mismatched term, while subtracting
mean |mismatched| would bias D negative under that same null. D is unbiased
where it needs to be; `mismatched_abs` carries the entanglement diagnosis
separately.

**Baseline discipline (unchanged from the program):** the reported quantity
is per-seed **learned_D = D_trained − mean(D_untrained)**, where the
untrained baseline is the same full procedure (probe fit + Jacobians) on 3
untrained builds (`mx.random.seed(1000+i)`, i ∈ {0,1,2},
`ModelConfig(image_size=128)`), measured per arm on that arm's data. F19
showed untrained z retains appearance linearly (JL), so untrained probes are
expected to fit; untrained essence heads are random, so D_untrained ≈ 0 is
expected — the subtraction guards instrument bias, not a known prior.

## 3. The channel-pairing commitment — frozen from `materials.py`, before any measurement

The pairing is read off the source of the appearance map
(`MaterialSampler._appearance_from_factors`, `materials.py:297–322`): the
historical couplings and the dedicated map the gain dial blends toward
(`metallic ← heaviness`, `roughness ← grip`, `val ← 1 − hardness`).

| render channel (probe target)             | physics factor | essence output | sign s_cp | secondary behavior field | sign |
|---|---|---|---|---|---|
| `metallic` (appearance_vector[5])         | heaviness      | density        | +         | `push.path_length`       | −    |
| `roughness` (appearance_vector[4])        | grip           | friction       | +         | `push.slid_distance`     | −    |
| value V = max(R,G,B) (appearance_vector[0:3]) | 1 − hardness | restitution   | **−**     | `drop.n_bounces`         | −    |

Behavior signs are the composition channel↔factor × factor↔field: density↑ →
less travel under a fixed impulse (−); friction↑ → less slide (−);
restitution↑ → more bounces (+), composed with V's − pairing → −. Behavior
fields are addressed by `behavior_field_names()` name, never by index; a
missing name fails loudly. The behavior endpoint (D_behavior, same
construction on the behavior head) is **secondary**: co-reported, cannot
rescue a failed primary. `transmission`, `ior`, and hue are excluded —
`clarity` never enters `_physics_from_factors`. No other pairing may be
substituted post hoc.

## 4. Inputs — all existing

Checkpoints `runs/f27b/` (F27b, all gates passed): per arm
{ctrl, base, g2, loud} — 3 shared seeds `<arm>_s{0,1,2}` plus the disjoint
pair `<arm>_renderonly` (`--behavior-weight 0 --essence-weight 0`; essence
head untrained) and `<arm>_behavonly` (`--render-weight 0`; essence head
**trained**). Datasets `data/pm_f27_{ctrl,base,g2,loud}`; measured r =
0.008 / 0.263 / 0.809 / 0.992. This is a re-measurement of F27b's artifacts
with a new instrument, not a replication.

## 5. Measurements

1. **learned_D per arm** (primary, essence-head endpoint): 3 seeds × 4 arms,
   untrained-corrected as in §2.
2. **learned_D_behavior** (secondary): same construction, committed behavior
   fields.
3. **Disjoint stitched reference** per arm: directions from the
   `renderonly` model's probes, transferred into the `behavonly` model's
   z-space through a linear stitch (same ridge, α = 1e-3), aligned against
   `behavonly`'s trained essence-head Jacobians. Stitch quality:
   variance-weighted R² on the test split; below the gate the cell is
   **VOID, not zero**.

   *Role — corrected from the session sketch that proposed this experiment,
   which predicted "stitched disjoint D ≈ 0 at every r." That was wrong: in
   coupled arms two separately trained models can align matched directions
   through genuine world correlation alone (renderonly learns appearance,
   behavonly learns physics, and the world ties them at r = 0.81/0.99). The
   disjoint arm is therefore the **content-without-substrate reference**:
   verdict-bearing only at ctrl (where no content exists), co-reported at
   coupled arms, where shared-vs-disjoint separates content alignment from
   unity beyond content.*
4. **Magnitude-coherence cross-reference** (no new measurement): the F27b
   values are reprinted beside learned_D for the dissociation read (H4).

## 6. Hypotheses and frozen decision rules

All arm-level statistics are the mean of per-seed learned_D (n = 3); Welch t
on per-seed values.

**H1 — substrate escape (the kill switch), on the ctrl arm:**

    CLEAN         |learned_D(ctrl)| < 0.05
    ENTANGLED     learned_D(ctrl) >= +0.10  → directionality inherits substrate
                  entanglement; H2/H2a are withheld for content-reading and the
                  finding is recorded as "directionality does not escape
                  entanglement"
    BIASED        learned_D(ctrl) <= −0.10  → instrument bias; run VOID
    INCONCLUSIVE  otherwise

**H2a — power anchor (readable only if H1 = CLEAN):** at loud the dedicated
map is 87.5%-blended — near-definitional content. If it cannot separate,
the metric lacks power and an H2 NULL is unreadable.

    POWERED       learned_D(loud) − learned_D(ctrl) >= +0.10 AND Welch t >= 2.5
    UNDERPOWERED  otherwise → H2 NULL is unreadable (CONTENT may still be read)

**H2 — the primary contrast: g2 vs ctrl (readable only if H1 = CLEAN).**
g2 is primary *by choice, as the harder test*: at g = 2 the dedicated map is
only 50%-blended (w = 1 − 1/g), so matched directions must read the authored,
confounded coupling — not the dedicated map's near-identity. A pass here is
evidence the metric reads content as authored, not as engineered.

    CONTENT       learned_D(g2) − learned_D(ctrl) >= +0.10 AND Welch t >= 2.5
    NULL          |learned_D(g2) − learned_D(ctrl)| < 0.05
                  (readable only if H2a = POWERED)
    INCONCLUSIVE  otherwise

Interpretations frozen with the rules: **CONTENT** → a coherence-family
metric finally discriminates content, and the F22–F27 lever arcs can be
re-asked against an instrument that can see the answer. **NULL beside
H2a = POWERED** → matched content at authored strength is invisible to
directional reads even where near-definitional content is visible; the
authored-coupling regime is below this instrument's floor.

**H3 — disjoint stitched reference:**

    ctrl (verdict-bearing):
      PASS          |D_disjoint(ctrl)| < 0.05
      FAIL          |D_disjoint(ctrl)| >= 0.10  → the statistic can be
                    manufactured without joint training OR content;
                    interpretation of all arms withheld pending diagnosis
      INCONCLUSIVE  otherwise
    coupled arms: co-reported, no verdict. Frozen interpretive notes:
      shared ≈ disjoint (|diff| < 0.05)  → directional alignment is
        world-content, achievable without a shared substrate; no unity evidence
      shared − disjoint >= +0.10         → joint training adds alignment
        beyond content + linear stitch — the first unity-shaped positive the
        program could report

**H4 — dissociation (descriptive, no verdict):** Spearman between per-arm
learned_D means and r is printed beside the known magnitude profile (flat
~0.42 across ctrl/base/g2, falling at loud). The dissociation claim needs
H1 = CLEAN plus a positive trend where magnitude coherence has none.

## 7. Gates — all must pass before any verdict is read

    apparatus   arm r values carried from the F27b report: |r(ctrl)| < 0.10,
                r monotone base < g2 < loud.
    collapse    re-encoded trained-z PR >= 8 per model (the F27 floor).
    probes      train-split 5-fold CV R² of each committed channel probe:
                >= 0.25 per trained model, >= 0.10 per untrained build.
                A failing channel VOIDs its cells in that model; any VOID
                committed channel in a verdict arm (ctrl, g2, loud) withholds
                the verdicts that read that arm.
    jacobians   median per-scene ||∇_z e_p|| (standardized coords) >= 1e-4
                per essence output per model — guards cosine degeneracy.
    stitch      per-arm stitch R² >= 0.5, else that arm's disjoint cell is
                VOID (ctrl VOID → H3's ctrl verdict is withheld).

## 8. Predictions — frozen before any data

- **P1 (70/30):** H1 → CLEAN. Directionality escapes substrate entanglement
  at ctrl. The 30 is real: F27b showed entangled *magnitudes*; entangled
  *directions* would be a stronger version of the same pathology and would
  kill this metric the way F27b killed the last one.
- **P2 (75/25):** H2a → POWERED.
- **P3 (60/40):** H2 → CONTENT at g2. The authored map at 50% blend is
  readable by linear probes (F19 precedent) — but this is the hard test by
  design, and 60/40 is honest.
- **P4:** H3 ctrl → PASS; at g2, D_disjoint > 0 (content without substrate)
  with shared ≥ disjoint. Lean, not odds: the shared−disjoint gap is the
  program's most genuinely open cell.
- **P5:** H4 Spearman > 0 while the magnitude profile stays flat/falling —
  the two metrics dissociate.

## 9. Cost

Measurement only. ~23 encode passes (12 shared + 8 disjoint + 3 untrained,
re-used across arms where the model is arm-independent) at the F24-measured
rate ≈ 4 min; ridge probes, stitches, and per-scene Jacobians on ~1M-param
heads are seconds each in MLX. **≈ 10–20 min wall-clock on the M5 Pro**, no
training, no new datasets, no GPU contention concerns at 128 px.

## 10. Honest limits

- n = 3 seeds per arm; only large effects clear t ≥ 2.5 (program-standard
  since F27, acknowledged again).
- The ±0.05 / ±0.10 bars are set a priori on a cosine-mean statistic whose
  null spread has never been piloted. The ctrl arm and the untrained
  baseline supply the empirical null at read time; if that null swamps the
  bars, the honest price is INCONCLUSIVE — the bars will not be re-tuned
  post hoc.
- Ridge directions are *linear* reads of possibly nonlinear channel
  encodings. A CLEAN-everywhere null could mean "linear directions cannot
  find the content," not "no content" — the probe-R² gate bounds but does
  not remove this.
- The stitch is linear, so the disjoint reference is a *lower bound* on
  content-without-substrate alignment.
- The essence-head endpoint measures head sensitivity to z, not behavioral
  realism; the behavior-head endpoint is secondary for exactly the outlier
  and chaos reasons documented at F22b/F8.
- One world class (shape-degenerate boxes, lr 2e-4 recipe) and F27b's exact
  checkpoints: any pathology in those artifacts propagates here unexamined.

## 11. Contamination disclosure

All F27b magnitude-coherence values, prediction gains, and the arm→r map
were seen before this design was written. No directional-coherence number
has ever been computed on any checkpoint. The pairing table is derived from
source code (`materials.py`), not from data. The session sketch that
preceded this document mispredicted the disjoint arm (≈ 0 at every r); the
correction is recorded in §5.3 and the corrected P4 stands as the frozen
prediction.

---

## Amendment 2 (2026-08-11, pre-run) — external context recorded; design unchanged; successor candidates registered

*Written with the measure driver commit, before any measurement has run.
Still true at this writing: no directional-coherence number has ever been
computed on any checkpoint. Nothing in §§1–11 is altered.*

**1. The Dyna-2 report post-dates this freeze and changes nothing here.**
Between the freeze (2026-08-04) and this amendment, Dyna Robotics published
"Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models" (2026-08-10;
single-lab tech report, not peer-reviewed), clipped and synthesized in the
research vault 2026-08-11. It is architecturally this program's design at
~1000× scale — a shared trunk under a video/world head and an action head,
the action head never reading the world latent (the F20 mechanism) — and two
of its results bear directly on F28's subject: its joint-training-alone
ablation does not scale (corroborating F27b's substrate-entanglement
reading), and its operational test for "the world model carries content" is
**held-out prediction on a held-out embodiment** — an external transfer
yardstick, not an internal scalar. This design was frozen before that report
existed and runs exactly as frozen. The transfer yardstick is deliberately
NOT adopted into F28: it addresses a different failure mode (F25's Goodhart
circularity — optimizing/reading an internal metric) than directional
coherence does (F27b's substrate entanglement — magnitude co-response
without content), the two instruments are complements rather than
substitutes, and H1's verdict is prerequisite knowledge for designing the
transfer experiment properly (an ENTANGLED H1 kills the internal-coherence
family outright and makes external transfer the only surviving
operationalization of unity).

**2. Successor candidates, registered by name before F28's numbers exist**
(so their motivation cannot be back-fit to this experiment's outcome; none
is designed or preregistered here, and none may inherit this document's
rules):

- **F29 candidate — held-out-transfer coherence.** Score unity externally:
  the *advantage of a jointly trained shared-latent model over disjoint
  single-head models* at prediction on a held-out world / shape-family
  neither was trained on. The disjoint training recipe already exists
  (F27b's `renderonly`/`behavonly` arms); the yardstick is Dyna-2's,
  transplanted; the Goodhart immunity comes from the target referencing the
  world, per F25's own lesson (constraints that reference the world
  converge; constraints that reference a metric diverge).
- **F30 candidate — render-only data scaling.** Fix the physics-labeled
  scenes, scale render-only data well beyond them, co-train, and measure
  whether transfer (F29's yardstick) moves where signal (F23), budget
  (F24/F25), and objective (F25) all failed. Dyna-2's central ablation —
  co-training helps only with extra content-bearing video, and the gap
  opens with scale — is the motivating datapoint; its absence at small
  scale is that report's own low-data regime, so a null here would not
  falsify the mechanism, only bound it below this instrument's scale.
- **F31 candidate — dynamics head.** The honest disanalogy: Dyna-2's world
  head predicts the *future*; this program's render head is static
  multi-view appearance (the one-step degenerate case). A
  next-observation-given-action head would test whether the scaling axis
  Dyna-2 found requires dynamics prediction — the sharpest reason a static
  render head might never organize the latent at any scale.

**3. Driver bookkeeping (documented in `scripts/f28_measure.py`, restated
here so the prereg is self-contained; decisions, not new rules).** The
arm-level trained-probe gate covers the three shared seeds — the models the
primary statistic reads; a `renderonly` model failing the same R² floor
voids its own disjoint cell (emitted as `null`, which the frozen verdicts
script renders VOID) rather than the arm's verdict, implementing §7's "a
failing channel VOIDs its cells in that model". The three untrained builds
are re-encoded per arm (§2's procedure: their z depends on the arm's
images even though their weights do not), so the encode count is 32 rather
than §9's 23-pass estimate; the 10–20 min wall-clock budget stands.

---

## Amendment 3 (2026-08-11, after pass 1 VOID) — the roughness↔friction cell is unmeasurable at this apparatus; the committed pairing shrinks to two cells

**Pass 1 record.** Run 2026-08-11 on the frozen design. **VOID by its own
§7 gates**: the trained-probe floor (R² ≥ 0.25 per committed channel)
failed in all four arms — min 0.148 (ctrl), 0.219 (base), 0.215 (g2),
0.248 (loud) — so no verdict was read. Aggregate observations recorded
behind the failed gates (not verdicts): learned_D ≈ 0 at every arm
(+0.0000 / +0.0001 / +0.0006 / +0.0031), untrained baselines ≈ 0,
mismatched_abs ≤ 0.008; the loud disjoint cell survived its own gates with
stitch R² 0.998 and D −0.0029. Report preserved at
`runs/f28/f28_report_pass1.json` (gitignored, regenerable).

**Diagnosis (post-void, measurement-only).** A per-channel breakdown on
the same encodes shows the failing minimum is the **roughness** column in
every arm (trained seeds 0.148–0.383; metallic 0.53–0.99 and value
0.82–0.98 clear the floor everywhere, including every `renderonly`
model). The cause is a render-configuration property measured three times
before on other worlds — F19 (roughness "untestable", pixel ceiling 0.035
at 128 px), F21 Arm 1 (untrained reach 0.18–0.33 across the noise
ladder), F21 Arm 2 (256 px oblique doubles reachability to only ~0.37) —
and now reproduced on the pm_f27 worlds (pixel-feature ceiling for
roughness 0.05–0.26 per arm): **128 px flat-lit renders barely vary with
roughness**. The trained latents often *exceed* the crude pixel-feature
ceiling, so the latent is not discarding roughness; there is almost
nothing to keep. The freeze-time error is §8/P3's appeal to "F19
precedent" for probe readability: F19's readable channel was color;
roughness was unmeasurable at this resolution in F19's own record. The
coupling dial cannot fix this — it changes what the appearance *params*
encode about physics, not how legibly the *render* exposes the params.

**The change (this amendment alters §3; nothing else).** The committed
pairing is reduced to the two cells whose render channels are measurable
at this apparatus:

| render channel | physics factor | essence output | sign | behavior field | sign |
|---|---|---|---|---|---|
| `metallic` (appearance_vector[5]) | heaviness | density | + | `push.path_length` | − |
| value V = max(R,G,B) (appearance_vector[0:3]) | 1 − hardness | restitution | **−** | `drop.n_bounces` | − |

- **D is the mean of the two matched signed cells.** The essence-head
  Jacobians are still computed for all three outputs (they cost nothing
  and their quality gate is unchanged), so the alignment matrix is 2 × 3
  and `mismatched_abs` now averages the four unmatched cells — the
  friction row survives as entanglement diagnostic.
- **The roughness↔friction cell is recorded as VOID-BY-LEGIBILITY at this
  apparatus, not NULL.** No statement about friction-content
  directionality is made or implied by pass 2. The escalation path if that
  cell is ever wanted is dataset regeneration at 256 px oblique lighting
  (the F21 Arm 2 recipe) plus retraining — hours of work for a channel
  F21 measured at only ~0.37 reachability even then; it is noted, not
  planned.
- All §6 hypotheses, decision rules, bars, §7 gates, and the §2 metric
  construction are **untouched**. `scripts/f28_verdicts.py` is unchanged,
  byte for byte. The driver and its pairing-pinning tests are updated in
  the same commit as this amendment to implement the two-cell table —
  the amendment changes the commitment; the driver follows it.

**Contamination disclosure.** Pass 1's aggregate D values (≈ 0
everywhere), gate minima, and the diagnostic's per-channel probe R² were
all seen before this amendment. Per-cell alignment values M[c,p] were
never stored or seen, so the two-cell D is not derivable from what was
seen. The channel-set change is legibility-motivated, and the seen
aggregates are null — no positive result exists to chase. The bars are
not re-tuned (§10's commitment stands).

**Pass-2 expectation, registered before the rerun.** Given the seen
pass-1 aggregates, the live prediction is now **(65/35) H1 CLEAN and H2a
UNDERPOWERED** — D ≈ 0 even at loud *with all gates passing*. That
outcome would be a genuine finding, not a failure: at r = 0.992 the
metallic probe reads the latent at R² ≥ 0.94 and the essence head extracts
density at gain 5.67, yet the z-direction that writes metal into the
render would share no orientation with the direction that moves the
density readout — the strongest available form of F27b's
substrate-entanglement-without-content-unity result, now measured
directionally with a valid instrument. If instead D(loud) − D(ctrl)
clears +0.10 (t ≥ 2.5), H2a is POWERED and H2 reads normally. Both
readings inherit §6's frozen rules unchanged.
