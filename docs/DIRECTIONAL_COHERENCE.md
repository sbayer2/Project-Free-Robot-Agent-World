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
