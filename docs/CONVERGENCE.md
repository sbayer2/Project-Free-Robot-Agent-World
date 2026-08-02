# F24 preregistration — representational convergence: do independent learners of the same world build the same latent?

*Preregistered 2026-08-01, before any alignment number was computed on any
trained checkpoint. Decision rules are committed as executable scripts
(`scripts/f24_verdicts.py`) in the same commit as this document. The measurement
driver is `scripts/f24_measure.py`; the metric implementations live in
`src/pseudomarble/models/alignment.py` with unit tests on synthetic data only.*

---

## 1. Why this experiment exists

The Platonic Representation Hypothesis (Huh, Cheung, Wang & Isola 2024,
arXiv 2405.07987) claims that independently trained models converge toward the
same representation of the world as capability rises — measured by mutual
k-nearest-neighbor alignment between their embedding spaces. The paper's own
sharpest unrun falsification test is: *train models on data from genuinely
different simulated worlds and see whether alignment tracks the world or the
optimizer.*

This project already owns a close instance of that apparatus, built for F22:
a family of worlds sharing the **same 512 objects, same physics answer key,
same held-out corner**, differing only in how much the pixels reveal about the
physics (coupling r from 0 to 0.992; verified again for this preregistration:
matched scene indices carry identical shape/density/friction/restitution across
`pm_f22_ctrl/base/g15/loud` and `pm_big`, with only appearance moving). On top
of that sit multi-seed checkpoint sets nobody has compared to each other:
20+16+16+8+3 independently trained encoders of these worlds, plus 1-trit FSQ
models at two couplings.

Three questions, none touched by F1–F23:

1. **Universality vs world richness.** Do independent seeds of the same world
   build aligned latents at all (beyond what untrained encoders share for
   free), and does that convergence rise with coupling strength — the dial
   F22a proved controls how much learnable structure the world offers?
2. **World-specificity.** Is a latent of world A more aligned with another
   latent of world A than with a latent of world B? This is the static core of
   the paper's falsification test, run in a microcosm.
3. **Convergence under scarcity.** When two independent learners are each
   forced to keep only **1.6 bits** per scene (FSQ k=1), do they keep the
   **same** 1.6 bits? No published work asks this. It is where the Platonic
   hypothesis meets the small-state thesis: abundance-convergence says models
   agree when they can keep everything; scarcity-convergence would say the
   world itself ranks its own information, and every learner reads the ranking
   the same way.

Why this is not F23 again: coherence asks whether one model's two heads move
*together* (within-model, cross-head). Alignment asks whether two models'
latents have the *same shape* (cross-model, same representation). A world
could produce convergent latents whose heads never unify, or unified heads on
seed-idiosyncratic latents. F23 answered the first axis (NULL: coherence is
flat in r). This experiment answers the second. The four quadrants of
(coherent?, convergent?) are all interpretable, and F23 has already fixed the
row we are in.

## 2. The metrics, exactly

**Primary: mutual kNN alignment** (the 2024 paper's own primary). For
representations X, Y of the same n scenes: for each scene i, take its k
nearest neighbors (Euclidean, excluding self) in X and in Y; the score is the
mean over i of |N_X(i) ∩ N_Y(i)| / k. In [0,1]; chance for random pairing is
≈ k/(n−1).

- k = 10 (the article's choice), n = 512 → chance ≈ 0.0196.
- Scenes: **all 512** (train ∪ test) of each arm. Rationale: alignment is a
  claim about representation geometry, not generalization; all seeds within an
  arm saw the identical train set, so no seed is asymmetrically advantaged;
  512 scenes give kNN sets 5× the statistical support of the 103-scene corner.
  The test-only corner (n = 103, chance ≈ 0.098) is computed as a secondary
  robustness read, no verdict weight.
- z: the encoder's 256-dim latent, one vector per scene from all 16 views at
  the checkpoint's trained image size (128 px), chunked encode (the F21
  lesson). FSQ checkpoints are **excluded** from kNN alignment — a k=1 latent
  lives on ≤3 distinct points and neighbor sets are meaningless there; FSQ
  models are measured by H3's categorical statistic instead.

**Secondary: linear CKA** (Kornblith et al. 2019) on the same z matrices,
reported alongside every mutual-kNN number, no verdict weight. The two metrics
see different things (global linear shape vs local neighborhoods); divergence
between them is informative and will be reported, not adjudicated.

**H3 statistic: permutation-invariant trit agreement.** For two k=1 FSQ
models with per-scene codes a, b ∈ {−1,0,1}^512:
`agree(a,b) = max over the 6 relabelings σ of mean(σ(a) = b)`.
Chance is estimated by shuffling b across scenes (preserving marginals) and
recomputing the same max-over-relabelings statistic, B = 1000 draws; we use
the mean and p95 of that null distribution. The relabeling max is applied to
the null draws too, so the selection effect is in both numerator and baseline.

## 3. The baseline discipline — the JL floor

The source article's own toy is the cautionary tale: two **untrained** random
projections of shared structure score CKA 0.88. Random maps preserve relative
geometry (Johnson–Lindenstrauss — the same lemma behind F19's
untrained-encoder surprise), so two untrained encoders fed the same images
already share the images' neighborhood structure for free. Raw alignment
between trained models is therefore uninterpretable, exactly as raw coherence
was before F6.

Every reported number is

    learned_alignment(arm) = mean over trained pairs − mean over untrained pairs

with the untrained baseline computed **per arm on the same inputs**: 8 fresh
inits (`mx.random.seed(1000+i)`, the same `build_model` path training uses),
encoded on that arm's images, all 28 untrained pairs. For cross-arm
comparisons the baseline is untrained init i on arm A vs untrained init j on
arm B, i ≠ j, mirroring the trained cross-pairs.

The untrained floor is itself a registered readout (P3): if it is far above
chance, that is the empirical demonstration that PRH-style measurements need
baseline subtraction — a point absent from the source article.

## 4. H1 — does convergence rise with coupling strength?

**Arms** (continuous models only):

| arm | r | checkpoints | seeds |
|---|---|---|---|
| ctrl | 0.000 | `runs/f22/ctrl_cont_s*` | 3 (descriptive only — too few for the verdict) |
| base | 0.263 | `runs/f22/base_cont_s*` | 8 |
| g15 | 0.636 | `runs/f22/g15_cont_s*` | 16 |
| loud | 0.992 | `runs/f22/loud_cont_s*` | 16 |

All seeds are usable, including g15/loud s8–15 (F23's fresh seeds): no
hypothesis in this document was generated from any seed's latent geometry —
alignment has never been computed on any of these checkpoints.

**Unit of analysis:** per-seed mean alignment to the other trained seeds of
its arm (n = 8 or 16 per arm). Pairs within an arm are not independent; the
per-seed mean is the standard reduction and is what the Welch t is computed
over. The untrained subtraction is a per-arm constant, so it shifts every
seed's value equally and the variance of `learned` equals the variance of the
trained per-seed means (the F23 H1 repair, applied here from the start).

**PRIMARY_H1 = learned(r=0.636) − learned(r=0.263).**

Decision rule, fixed now:

    RISES         PRIMARY_H1 >= +0.05  AND  Welch t >= 2.5
    NULL          |PRIMARY_H1| < 0.03
    INCONCLUSIVE  otherwise

Secondary, no verdict weight: learned(0.992) (does convergence hold up where
extraction *fraction* falls, F23 H2), learned(0) at n=3, and the shape of the
four-point curve.

## 5. H2 — the two-worlds test: is alignment world-specific?

Cross-arm alignment on matched scene indices: trained seed i of arm A
(encoding A's renders) vs trained seed j of arm B (encoding B's renders),
i ≠ j (same-numbered seeds share an init via `mx.random.seed(seed)` in
train.py:143, so same-seed pairs are excluded from both trained and untrained
cross sets).

**PRIMARY pair: base (r=0.263) ↔ loud (r=0.992)** — well-seeded on both sides
(8 and 16). ctrl ↔ loud (the maximal world difference) is reported as
secondary; its n=3 side is too weak to carry a verdict.

**Statistic:** for each loud seed, the paired difference
`d_i = mean within-alignment (to other loud seeds) − mean cross-alignment (to
base seeds)`, one-sample t over n = 16. The base-side mirror (n = 8) is
reported as a consistency check. Both sides use trained-minus-untrained
learned values (cross baseline per §3).

    WORLD-SPECIFIC  mean d >= +0.03  AND  t >= 2.5  (loud side)
    NULL            |mean d| < 0.02
    INCONCLUSIVE    otherwise

**Honest scope note, written before the data:** the arms share *shape*
entirely — only the material-appearance statistics and the
appearance→physics relation differ. So H2 discriminates only the
material/coupling component of the geometry. A NULL here means the latent
geometry is dominated by what the worlds share (shape — consistent with F18),
not that world identity never matters. That reading is fixed now so it cannot
be invented post hoc.

## 6. H3 — convergence under scarcity: do 1-trit models keep the same 1.6 bits?

**Checkpoint sets:** `runs/fsq/k1_s0..2` (pm_big, r=0.263) and
`runs/f22/loud_fsq1_s0..2` (pm_f22_loud, r=0.992). 3 seeds → 3 pairs per
coupling. Codes are `out["code"]` (numerically `round(tanh(·))` ∈ {−1,0,1})
over each arm's 512 scenes.

**H3a — agreement.** Per coupling:

    CONVERGENT      all 3 pairs above their permutation-null p95
                    AND mean excess over null mean >= 0.15
    NOT CONVERGENT  mean excess < 0.05
    INCONCLUSIVE    otherwise

**H3b — content.** What the kept trit encodes: mutual information (natural
log, plug-in estimate over the 512 scenes) between the code and (a) the shape
label (5 classes), (b) each hidden factor (heaviness, grip, hardness) cut into
terciles. Classification, fixed now:

    shape-coded     MI(code; shape) > 1.2 × max over factors MI(code; factor)
    material-coded  max factor MI > 1.2 × MI(code; shape)
    mixed           neither exceeds the other by 20%

Reported per coupling. H3b is descriptive (no t-statistic); its role is to
say *what* the agreed-on bits are, and whether that changes when the world's
material channel becomes loud.

## 7. Independence of the hypotheses

The F23 lesson, applied in advance: **no verdict here may be cited as support
for any other**, and none of them supports or is supported by F23's verdicts.
In particular: if H1 returns NULL it must be reported beside F23's H1 NULL as
two *separate* negatives, not merged into one narrative of "nothing changes
with r" — H2 (extraction) already showed something does. And H3's agreement
statistic is categorical and unrelated to kNN alignment; a CONVERGENT H3
beside a NULL H1 is a coherent outcome (coarse partition shared, fine
geometry seed-idiosyncratic), not a contradiction.

## 8. Gates — all must pass before any verdict is read

    self       mutual_knn(z, z) = 1.0 exactly for one checkpoint per arm, and
               mutual_knn(z, z[random scene permutation]) < 3 × chance.
               (Implementation sanity: the metric and its indexing.)
    collapse   latent PR >= 8 (train.py:94 definition, the F12 healthy band)
               for every continuous checkpoint entering H1/H2. Checkpoints
               failing the floor are dropped and counted in the report; if an
               arm falls below 6 usable seeds (3 for ctrl) its numbers are
               reported but its verdicts are withheld.
    determinism  fixed RNG seeds for untrained inits (1000+i), permutation
               null (seed 2026), and kNN tie-handling by stable argsort;
               n, k, and per-arm seed lists are printed into the report.

## 9. Predictions — frozen before any measurement

- **P1 (confident):** learned within-arm alignment is positive at base, g15,
  and loud — trained pairs beat untrained pairs with Welch t ≥ 2.5 per arm.
  Training creates convergence beyond the architectural floor.
- **P2 (moderate):** H1 returns RISES. Stated despite F23's NULL, because
  convergence measures how much of z the *data* pins down, not whether heads
  unify; F22a proved the data pins down far more at r=0.636. If this instead
  returns NULL, the combined F23+F24 picture — richer worlds change *what* is
  extracted (F23 H2) but neither how unified (F23 H1) nor how convergent (this
  P2) the representation is — is itself a sharp result and will be reported as
  such, not as a failure of the instrument.
- **P3 (confident):** the untrained-pair floor is ≥ 0.10 at every arm — at
  least 5× chance (0.0196). The JL floor is real and large; raw alignment
  without baseline subtraction would overstate convergence severalfold.
- **P4 (moderate-high):** H2 returns WORLD-SPECIFIC: within-world alignment
  exceeds cross-world by ≥ 0.03 (paired t ≥ 2.5, loud side). If NULL, the
  fixed reading of §5 applies (shape-dominated geometry, F18 consistent).
- **P5 (confident at r=0.263, moderate at r=0.992):** H3a returns CONVERGENT
  at both couplings — independent 1-trit learners keep the same partition.
- **P6 (moderate):** H3b classifies the code as **shape-coded at BOTH
  couplings** — even at r=0.992 where the material channel is loud — because
  F22a showed the k=1 bottleneck converts the loud coupling into only
  Δ +0.170 of prediction, implying the trit's budget stays on shape. The
  alternative (the trit flips to material-coded under a loud world) would be
  the more surprising and more Platonic outcome: scarcity-forced agreement on
  *what matters most*, re-ranked by the world. P6 predicts it does NOT happen.

## 10. Cost

No training. ~81 chunked encodes (49 trained + 32 untrained model×arm
combinations) of 512 scenes × 16 views at 128 px on Metal, ~10–20 min total;
distance matrices are 512² numpy ops, seconds. The permutation null is
1000 × 6 relabelings on length-512 int vectors, < 1 s.

## 11. Honest limits

- **Capability is dialed by the world, not by scale.** The 2024 paper's claim
  is "alignment rises with model capability"; here every model is the same
  1M-param architecture and the *world's learnable structure* moves. This is
  a deliberate variant, not a replication, and will be reported as such.
- **Within-arm convergence is the universality hypothesis (Olah 2020), not
  the cross-modal PRH.** No result here speaks to language↔vision
  convergence.
- **Scenes were seen in training** (all seeds, identically). Alignment on
  held-out-only scenes is the secondary read; if the two disagree, that
  disagreement is reported, primary stays primary.
- **Trit agreement at n=3 seeds** is 3 pairs per coupling; the per-pair
  statistic is over 512 scenes (tight), but seed-level generality is weak and
  the verdict language will say "3 of 3 pairs", not "seeds in general".
- **The alignment thresholds (0.05/0.03/0.02) are conventions**, set a priori
  by analogy to F23's bands and the chance floor, not derived from pilot data
  (there is none — that is the point of freezing them here).

---

## AMENDMENT (2026-08-01) — after the run

*Everything above this line is unchanged from the freeze commit. This appendix
records outcomes and grades; the full entry is `docs/FINDINGS.md` F24.*

**Gates:** all passed. Self = 1.0 at every arm; shuffle 0.018–0.029 vs
3×chance 0.059; PR 9.8–70.3, zero checkpoints dropped.

**Verdicts (from `scripts/f24_verdicts.py`, unedited):**

- **H1 → NULL.** PRIMARY = −0.0172 (t = −1.62). Secondary: learned(0) =
  −0.154, learned(0.992) = +0.053.
- **H2 → WORLD-SPECIFIC** by the frozen loud-side rule (d = +0.098,
  t = 12.0, n = 16) — with the base-side mirror at −0.033 (t = −3.85), so the
  honest scope is "world-specific only from the loud side"; §5's frozen
  scope note applies.
- **H3 → CONVERGENT at both couplings** (agreements 0.87–0.955, nulls
  0.375–0.445, every pair above its p95). Content: shape-coded at both by the
  1.2× rule; at r = 0.992 hardness MI rose 0.001 → 0.231 while shape fell
  0.879 → 0.564.

**Predictions graded:**

- **P1 FALSIFIED at 3 of 4 arms** — the sign inverted: trained pairs align
  *below* untrained pairs at r ≤ 0.636 (t −4.3 to −5.8). Training
  individuates; only r = 0.992 shows positive learned alignment (+0.053,
  t = 3.08). The confident prediction was wrong in the most informative way
  available.
- **P2 FALSIFIED** — NULL, not RISES. The fallback reading frozen in §9
  applies verbatim: richer worlds change *what* is extracted (F23 H2), not
  how unified (F23 H1) nor how convergent (this) the representation is.
- **P3 CORRECT, under-called seven-fold** — the untrained floor is
  0.67–0.72, not the predicted ≥ 0.10. 35× chance.
- **P4 CORRECT by the frozen rule**, narrowed by the base-side mirror as
  above.
- **P5 CORRECT** at both couplings.
- **P6 CORRECT by rule**; the registered-visible shift inside it (hardness
  entering the trit at loud coupling) is recorded in F24 as an observation,
  not a verdict.

**Unregistered observation, labeled as such:** mutual kNN and linear CKA
anti-correlate across arms (CKA learned positive at ctrl/base/g15, negative
at loud — the mirror image of kNN), and the test-only secondary inverts the
full-set sign at every arm. Both were computed under the preregistered plan
(§2 required reporting them); the *interpretation* — individuation lives in
the training region and in local neighborhoods, while global axes converge —
is post-hoc and marked accordingly.
