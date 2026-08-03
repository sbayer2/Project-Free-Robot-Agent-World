# F27 preregistration — the shape-degenerate world: with geometry removed, can material carry coherence at all?

*Preregistered 2026-08-02, before any F27 dataset was generated. Decision
rules are committed as `scripts/f27_verdicts.py` in the same commit as this
document. Zero new model or generator code: the world is built from existing
flags (`--shapes box` plus the F22 coupling dial); the only new files are
this document and the measure/verdict scripts.*

---

## 1. Why this experiment exists

F26 ended with a post-hoc synthesis and its falsifiable implication, recorded
in FINDINGS: the coherence this instrument measures (~0.08–0.19 in every
world, including zero coupling) is **shape-carried** — shape is the one
factor visible to the render head *and* predictive for the behavior head in
all nine worlds, which is why the coupling dial never moved the metric (F23,
F25, F26). The implication: *in a shape-degenerate world (one shape,
materials only), learned coherence should collapse toward zero at r = 0 and —
if material coupling can carry coherence at all — grow with r.* This is the
first world design in which the dial **should** move this metric.

The stakes, stated before the data: if material coherence rises here, unity
was never impossible — it was **shadowed** by a cheaper channel, and the
Marble property needs a world where geometry doesn't give co-movement away
for free. If it stays flat *while prediction rises*, that is the arc's most
damning negative: even in a world where the only learnable shared structure
is material, and the model demonstrably learns it, the two heads still do not
unify — unity is then a property this architecture does not express at any
coupling, in any world tried.

## 2. The world

`--shapes box` on the standard generator. `SHAPE_TO_GEOM` fixes each
primitive's dimensions, and primitives take no per-scene scale, so a
single-shape world has **zero geometric variation**: every scene is the same
0.15-half-height box; all render variation is material appearance, all
behavior variation is material physics. The extrapolation-corner holdout
(heavy+bouncy) is factor-based and carries over unchanged.

Four arms, seed 1234, 512 scenes, 16 views, 128 px:

| arm | flags | expected r |
|---|---|---|
| ctrl | `--coupling-alpha 0 --coupling-gain 1` | ≈ 0 |
| base | `--coupling-alpha 1 --coupling-gain 1` | ≈ 0.26 (historical) |
| g2 | `--coupling-alpha 1 --coupling-gain 2` | ≈ 0.81 |
| loud | `--coupling-alpha 1 --coupling-gain 8` | ≈ 0.99 |

Datasets `data/pm_f27_{ctrl,base,g2,loud}`. Coupling strength is reported as
the measured roughness↔friction correlation r per arm (the F22a convention),
not as g. Apparatus checks before any model is trained (part of the gates):
matched scene indices must carry identical physics across the four arms
(the F22a decoy-stream property, re-verified in the single-shape regime),
and every scene's `input.shape` must be `box`.

## 3. Arms and training

Per arm: 3 seeds of the standard shared model (lr 5e-4, 50 epochs, 128 px) →
`runs/f27/<arm>_s{0,1,2}`, plus one masked-loss control pair
(render-only / behavior-only, seed 0, the F23 recipe) →
`runs/f27/<arm>_{render,behav}only`. 20 training runs total; no new
architecture, no FSQ, no coherence objective — this is a *world* experiment.

## 4. Measurements

1. **Learned coherence** (primary, behavior target; essence co-reported):
   `scripts/run_coherence_experiment.py` per arm, 5 untrained inits, test
   split — identical to F23/F25/F26.
2. **Behavior gain** (the H3 sanity gate): held-out gain vs the
   train-mean baseline. With one shape the F18 shape-only oracle *is* the
   train mean (a constant one-hot column), so gain itself is the essence
   signal; no Δ subtraction is defined and none is reported. The appearance
   oracle (ridge on the recorded `appearance_params`, analysis-only) is
   computed as the reachable ceiling.
3. **Cross-seed learned kNN alignment** (secondary, no verdict): the F24
   harness per arm (k = 10, 512 scenes, 8-init untrained baseline per arm) —
   does individuation persist when geometry is gone?

## 5. Hypotheses and frozen decision rules

**H1 — the shape-channel test** (the F26 synthesis's direct audit):

    SHAPE-CARRIED CONFIRMED  learned_coh(ctrl) < 0.05
    REFUTED                  learned_coh(ctrl) >= 0.10
    INCONCLUSIVE             otherwise

The multi-shape zero-coupling world scored +0.187 (F26). If removing shape
removes the co-movement, the F26 synthesis survives its audit; if +0.10-level
coherence persists in a world with *no* shared world-factor at all, the
metric is detecting something architectural and the synthesis is wrong.

**H3 — the learnability gate (graded before H2; H2 is unreadable without
it):**

    MATERIAL LEARNED   gain(loud) − gain(ctrl) >= +0.20 AND Welch t >= 2.5
    NOT LEARNED        otherwise → H2 is withheld (a coherence null would be
                       uninformative if the model never learned the material)

**H2 — can material carry coherence?** (readable only if H3 = LEARNED):

    RISES  learned_coh(loud) − learned_coh(ctrl) >= +0.10 AND Welch t >= 2.5
    NULL   |learned_coh(loud) − learned_coh(ctrl)| < 0.05
    else   INCONCLUSIVE

Interpretations frozen with the rules: **RISES** → unity was shadowed, not
impossible; the Marble property needs a geometry-quiet world. **NULL beside
H3 = LEARNED** → the strongest negative of the program: no world
configuration tried can make this architecture unify its heads; the
"one latent, two projections" claim fails at the architecture, full stop
(at this scale and class — not a theorem).

**Statistics:** per-arm learned coherence with Welch t from summary stats
(the F23 repair); gain compared on per-seed values (n = 3 vs 3 — small, so
only large effects can clear t ≥ 2.5, acknowledged).

## 6. Gates — all must pass before any verdict is read

    apparatus  every scene's input.shape == "box" in all four arms; matched
               scene indices carry identical physics across arms; measured
               r(ctrl) within |r| < 0.10 and r monotone in g across arms.
    validity   base-arm learned coherence is NOT anchored to F13 (different
               world class — no historical single-shape number exists); the
               harness-level anchor is instead the untrained baseline, which
               must sit within ±0.02 of the multi-shape untrained values
               (~0.109) — F23 showed it is world-independent; if that breaks,
               the harness has drifted and the run is VOID.
    collapse   trained-mean PR >= 8 per arm.
    control    per-arm disjoint baseline |value| < 0.05. If a single control
               pair trips the bound, two additional pairs are trained and the
               gate is evaluated on the mean of three — the F26 refinement,
               adopted PROSPECTIVELY this time.

## 7. Predictions — frozen before any data

- **P1 (moderate-high):** H1 → SHAPE-CARRIED CONFIRMED: learned coherence in
  the single-shape zero-coupling world collapses below 0.05 (vs +0.187 with
  shapes). The F26 synthesis survives its audit.
- **P2 (high):** H3 → MATERIAL LEARNED — in a world where material is the
  only signal, gain rises strongly with r (the F22a result, unshadowed).
- **P3 (moderate, genuinely uncertain, my lean is yes):** H2 → RISES. With
  geometry silent and material demonstrably learned, both heads must consume
  the same channel and should co-respond. The counter-scenario I take
  seriously after F25: the heads can consume the same information through
  non-co-moving pathways, in which case NULL lands and the architecture-level
  negative is final. Stated odds: ~60/40 for RISES.
- **P4 (secondary, no verdict):** cross-seed learned alignment remains
  negative at low r (individuation persists without geometry) and is higher
  at loud than ctrl (the F24 pattern reproduced in the new world class).

## 8. Cost

Four 512-scene datasets (~10–15 min each, sequential), 20 training runs
(~30 min), four coherence-harness invocations plus gain/alignment
measurement (~20 min). Roughly 2 hours wall, all background.

## 9. Honest limits

- n = 3 seeds per arm: only large coherence effects can clear t ≥ 2.5. The
  historical effect sizes at stake (0.187 → <0.05; bonus +0.12) are large
  relative to seed sds (~0.03–0.12), so the design is powered for the
  question it asks, barely.
- One shape (box) — a result here is about geometry-quiet worlds, not about
  boxes; if H2 RISES, replicating on a second single-shape world (cylinder)
  would be the immediate robustness check.
- With one shape the render task is appearance-only; encoders could organize
  differently from every prior world (the PR gate and untrained-baseline
  validity gate are the guards).

---

## AMENDMENT 1 (2026-08-03) — first run VOID by its own gates; F27b recipe frozen

**The first run (lr 5e-4, the program-standard recipe) is VOID for its
verdict-bearing arms.** The ctrl and base encoders **fully collapsed** —
per-scene z constant to float32 noise (per-dim std ≈ 3e-5) while the renders
demonstrably vary — and the coherence harness's PR gate caught it (0.05/0.20
vs the ≥ 8 floor). Per §6, all verdicts are withheld. Three records from the
void run:

1. **The F12 cure does not transfer.** lr 5e-4 (which gave 20/20 healthy
   seeds in multi-shape worlds) collapses shape-degenerate worlds at weak r:
   with geometry gone and material unlearnable, the render task alone does
   not anchor the encoder. This is the steam-engine/representation-collapse
   reading measured in a new regime: remove the input diversity, get the
   collapse.
2. **The F10 law fired in a new costume.** The collapsed ctrl arm scored
   learned coherence **+0.393** — the highest raw number in the program's
   history, from an encoder representing nothing. Without the gate this
   would have read as "H1 REFUTED"; with it, it reads as the artifact F10
   documented. The gate, not the experimenter, caught it.
3. **A metric bug was found and fixed.** `alignment.participation_ratio`
   read the collapsed latents as PR ≈ 54–62 (uniform float32 noise ≈ full
   rank; no epsilon guard). Fixed to match the harness's guard; regression
   test added (`test_participation_ratio_flags_float_noise_collapse`).

**Contamination disclosure:** before the void was diagnosed, the
experimenter saw the first run's g2/loud numbers (gain 4.01/5.44; learned
coherence +0.304/+0.237 at lr 5e-4). The frozen rules and predictions
predate all data and stand unchanged, but P3's grading must carry this note:
partial (healthy-arm) first-run data was seen before F27b ran.

**F27b, frozen now:** identical design and rules; the single change is a
uniform recipe **lr 2e-4** (pilot check on ctrl seed 0: PR 41.7, de-collapsed;
only the PR/std apparatus values were read from the pilot), 50 epochs, all
four arms, 3 seeds + control pair each, under `runs/f27b/`. The measure
driver gains a `--root` argument; nothing else changes.

---

## AMENDMENT 2 (2026-08-03) — F27b results and grades

*Full entry: `docs/FINDINGS.md` F27.*

**Gates: ALL PASS** (PR 24–101; apparatus perfect; untrained anchor
0.1085–0.1086; every disjoint control within bounds).

**Verdicts (from `scripts/f27_verdicts.py --report runs/f27b/f27_report.json`,
unedited):** H1 → **REFUTED** (learned coherence at r = 0, single shape =
+0.434 vs the < 0.05 confirmation bar; disjoint control +0.0001). H3 →
**MATERIAL LEARNED** (+4.55, t 14.2). H2 → **INCONCLUSIVE** by the frozen
rule: the observed effect is a significant FALL (−0.229, t −3.29), a
direction the rule did not anticipate; no frozen interpretation applies and
none is claimed.

**Predictions graded:** P1 falsified (the second synthesis in a row killed
by its own audit). P2 correct. P3 falsified — and not toward its named
alternative: neither RISES nor NULL but a fall at extreme coupling; graded
with the Amendment-1 contamination note attached. P4 correct.

**The load-bearing observation** (stated in FINDINGS with the F13 dated
note): shared-latent +0.434 vs disjoint +0.0001 in a world where the two
tasks share nothing — joint training couples head sensitivities regardless
of content, so the magnitude-correlation coherence metric certifies
substrate entanglement, not content unity. The constructive successor
(directional, content-matched coherence) is recorded as candidate F28, not
preregistered here.
