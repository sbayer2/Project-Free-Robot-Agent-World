# F26 preregistration — the coherence-vs-coupling curve: is the baseline's co-movement "pre-filled" by the world?

*Preregistered 2026-08-02, before any new measurement. Decision rules are
committed as `scripts/f26_verdicts.py` in the same commit as this document.
No primary training: all 76 continuous checkpoints across nine worlds already
exist; the only new runs are twelve masked-loss control pairs.*

---

## 1. Why this experiment exists

F25's entry contains a claim that has never been tested as a curve: the k = 1
scarcity bonus vanished at r = 0.809 partly because "a continuous baseline
**already sits at +0.188**." The post-hoc mechanism offered alongside it (the
"pre-fill" reading): *the world's coupling itself correlates the two heads'
responses — the same news moves both incomes — so scarcity has nothing left
to add.* That story makes a concrete prediction: trained continuous learned
coherence should **rise with r**, with the zero-coupling control lowest.

The existing evidence is three points and they disagree with each other:

| r | learned coherence | source |
|---|---|---|
| 0.263 | +0.152 | F23 |
| 0.636 | +0.121 | F23 (the H1 NULL pair) |
| 0.809 | +0.188 | F25 cont |

Non-monotone, spread +0.067, each ± ~0.02–0.03 se. F23's frozen two-point
verdict was NULL; F25's baseline hints up. Only the full nine-point curve
settles whether the pre-fill reading survives — and it was offered post hoc,
so it must be either confirmed under preregistration or retracted.

## 2. Apparatus — the curve is already on disk

Nine continuous arms, all trained identically (lr 5e-4, 50 epochs, 128 px):

| r | tag | seeds |
|---|---|---|
| 0.000 | `runs/f22/ctrl_cont` | 3 |
| 0.263 | `runs/f22/base_cont` | 8 |
| 0.357 | `runs/f22/g11_cont` | 3 |
| 0.517 | `runs/f22/g13_cont` | 3 |
| 0.636 | `runs/f22/g15_cont` | 16 |
| 0.809 | `runs/f22/g2_cont` | 8 |
| 0.927 | `runs/f22/g3_cont` | 3 |
| 0.978 | `runs/f22/g5_cont` | 3 |
| 0.992 | `runs/f22/loud_cont` | 16 |

Harness: `scripts/run_coherence_experiment.py`, identical settings to
F23/F25 (behavior target headline, essence co-reported, 5 untrained inits,
test split). Existing reports are **reused unmodified** where the identical
harness already produced them (`runs/f23/coh_base_cont`,
`runs/f23/coh_g15_cont`, `runs/f25/coh_cont` for r = 0.809); the other six
arms are measured fresh. Each world gets its own masked-loss control pair
(render-only / behavior-only, seed 0, the F23 recipe) — twelve new ~87 s
runs; base/g15/g2 pairs already exist.

**H2 apparatus (descriptive):** the k = 1 FSQ level at three couplings —
`runs/fsq/k1` (r = 0.263), `runs/f25/g2_fsq1` (0.809), `runs/f22/loud_fsq1`
(0.992) — the third point measured fresh, the first re-measured with this
harness for comparability (F17's +0.27 was an earlier harness generation).
FSQ arms inherit their world's continuous control (the F25 disclosed repair).

## 3. Decision rule for H1, fixed now

Unit: per-arm learned coherence (trained_shared mean − untrained mean), with
Welch t from summary statistics (the F23 repair — `learned` shifts every seed
by a per-arm constant, so trained_shared's sd is the spread).

    RISES  learned(r=0.992) − learned(r=0) >= +0.05  AND  Welch t >= 2.5
           AND  Spearman rho(r, arm-mean learned) >= 0.6 over the 9 arms
    FLAT   max − min over the 9 arm means < 0.05
    MIXED  otherwise

Interpretations, frozen with the rule:

- **RISES** → the pre-fill reading survives; F23's NULL was a two-point
  sampling of a real slope (its arms happened to sit near each other).
- **FLAT** → the pre-fill reading is **retracted**: the +0.188 vs +0.146
  delta was noise, and the F25 bonus compression must be explained by the
  k = 1 *level* falling (H2), not by the baseline rising. The F25 discussion
  paragraph gets a dated correction.
- **MIXED** → the curve is noisy without a clean slope; the pre-fill reading
  is *unsupported* (not merely unproven) unless the ctrl arm is lowest, and
  the honest summary is "no interpretable r-dependence at n = 3–16."
- **Sharpest sub-check, graded separately:** is ctrl (r = 0, a world whose
  pixels carry *nothing* about physics) the lowest of the nine? If ctrl is
  NOT in the bottom two arms, the pre-fill story is dead regardless of the
  overall verdict.

## 4. Gates — all must pass before the verdict is read

    validity   base arm learned in [0.08, 0.22] (brackets F13's +0.146 — the
               same anchor F23 used; reused report must still satisfy it)
    collapse   trained-mean PR >= 8 on every arm
    control    per-world disjoint baseline |value| < 0.05 on every arm
               (fresh masked pairs for the six new worlds)

## 5. Predictions — frozen before any measurement

- **P1 (moderate):** ctrl (r = 0) shows the lowest learned coherence of the
  nine arms. This is the pre-fill story's sharpest and most survivable
  implication, and the one I actually believe.
- **P2 (low-moderate, stated with genuine uncertainty):** the overall
  verdict is MIXED — a noisy shallow rise that fails the Spearman/t bars.
  I do not predict RISES: F23's NULL between 0.263 and 0.636 is prior
  evidence the middle of the curve is flat, and n = 3 arms are noisy.
- **P3 (moderate):** the k = 1 level falls monotonically across
  r ∈ {0.263, 0.809, 0.992} — the bonus compression is at least partly the
  scarcity ceiling coming DOWN, not only the baseline going up. (F17's +0.27
  may also shrink on re-measurement with the current harness; if so, that is
  a harness-generation correction to F17's number, reported as such.)

## 6. Cost

Twelve ~87 s masked-control runs (~18 min), then seven coherence-harness
invocations (~3–5 min each). No primary training. Total under an hour.

## 7. Honest limits

- Six of nine arms have n = 3; the curve's error bars are wide in the middle.
  The frozen rule leans on the well-seeded endpoints (loud n = 16 vs
  ctrl n = 3) plus rank correlation, not on any single mid-curve arm.
- Reusing three existing reports assumes the harness is unchanged since
  F23/F25 — true by inspection (no commits touched
  `run_coherence_experiment.py` or `models/coherence.py` since), and the
  validity gate re-checks the base arm's number.
- H2 is three points; it is descriptive and will be reported without a fit.
