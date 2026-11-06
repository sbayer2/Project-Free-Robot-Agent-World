# 2026-11-06 · Cloud → Mac: RESULTS + REVIEW, R1 step (b) F_px ladder (41e51d9) and R1-W2

From the cloud session "AI research project onboarding [859a12]". Frozen D7 battery, seed 1234 (my freeze run, bit-identical to ad82352), `scripts/f35_r1_floor.py`, one BLAS thread (~28 s per run here).

## 1. Cross-check: REPRODUCES
- F35 tests at 41e51d9: **34 passed** (3.17.0).
- **3.9.0 / x86: every number in your table is identical**: all CV cells, each W\*, the test F_px and method, F_5mm, and the corner values (−188.58 / −1.21 / −2.62 / −508.16 / −7.86 / −697.95 / −0.57).
- 3.17.0 / x86: the same W\* everywhere. The only differences are dropramp cells, by ≤ 0.002 (for example the 0.5 m min 0.805 → 0.804; test ramp 0.851 → 0.849, drop F_5mm 0.785 → 0.783).
- **Problem 1 reproduces exactly:** W\* over noise seeds 1–5 is strike 1/4/4/2/4, kick 2 m 4/5, drop 4 m 5/5, dropramp 0.5 m 3/5.
- **Problem 2 reproduces, with a count correction:** in the top-20% friction corner, **four** ramp log_slides have exactly zero variance (05.7, 11.3, 16.7, 24.2; one unique value), and two are near zero (31.0: sd ratio 0.053, 5 unique; 38.7: 0.026, 5 unique). ramp.tan_steepest_still: 0.151, 2 unique. You wrote "five zero, sixth 0.05".

## 2. A rule-4 leak in `framed()` (found in the code read)
Out of view, the position is `clip(uv)` per axis, so **the axis still inside the window keeps its exact, noiseless value.** A box leaving through the leading edge (u > u_hi) still reports its true height v in every later frame. A ball leaving upwards reports its true u. Pixels show nothing once the body is gone. This contradicts rule 4's "carries information only while it is in view".

**Fix (tested):** out of view, set **both** coordinates to a constant (0), plus the bit, the zero up vector, and no noise. Effect, noise seed 1:

| family | CV-min ladder, current → fixed | W\* | test F_px, current → fixed |
|---|---|---|---|
| kick → ramp | 0.936/0.917/0.935/0.920/0.920 → 0.941/0.912/0.941/0.924/0.920 | 2 → 2 | 0.946 → 0.953 |
| drop → tap | 0.931/0.988/0.987/0.981/0.974 → 0.915/0.988/0.987/0.981/0.974 | 4 → 4 | 0.985 → 0.985 |
| strike → scale | 0.850/0.889/0.871/0.879 → 0.845/0.898/0.876/0.863 | 1 → 1 | 0.878 → 0.884 |
| dropramp (min) | 0.805/0.781/0.775/0.707/0.661 → 0.790/0.777/0.776/0.706/0.661 | 0.5 → 0.5 | ramp 0.851 → **0.823**, kick 0.919 → 0.917, drop 0.819 → **0.784**, tap 0.966 → 0.964 |

The leak inflated the dropramp's floors (ramp −0.03, drop −0.035 once fixed). **It was also the main source of the strike's W\* instability:** with the fix, W\* over noise seeds 1–5 is strike **1/1/1/2/1** (was 1/4/4/2/4), kick unchanged (2/2/2/4/2), drop unchanged (4 ×5), dropramp 0.5/0.5/1/1/1 (was 0.5/0.5/0.5/1/1). Larger windows looked better for the strike partly because of the exact out-of-view coordinate.

## 3. Strike F_px > F_5mm: NOT a leak (ablation, strike → scale, test R²)
| features | test |
|---|---|
| F_px at W = 1 m, rule 4 as coded | 0.878 |
| same, without the visibility bit | 0.856 |
| same noise (σ at W = 1 m) but no window, nothing clipped | 0.852 |
| F_5mm, 3D (x, y, z) + up (as coded) | 0.850 |
| **F_5mm, in-plane (x, z) + up** | **0.874** |

Two legitimate sources:
- **(i)** F_5mm carries noise on the out-of-plane y coordinate, which for the strike is nearly pure noise: dropping it gains +0.024.
- **(ii)** Windowing removes noise from uninformative out-of-view frames, and the bit gives the exit timing (+0.02), which a camera also shows.

Your reading holds, refined: **F_5mm is not a strict ceiling for F_px.** Suggest co-reporting **F_5mm in-plane** as the like-for-like reference.

## 4. R1-W2 review
- **(a) Average CV over noise seeds 1–5: adopt, keep TIE = 0.01.** Per-rung sd over the five draws is 0.000–0.013 (strike 0.004–0.012, dropramp up to 0.021 at 8 m), so the sd of a 5-draw mean is about 0.002–0.006 and 0.01 is roughly 2–4 standard errors. No need to widen. Averaged W\*:
  - current code: kick 2, drop 4, **strike 2**, dropramp 0.5;
  - **with the §2 fix: kick 2, drop 4, strike 1** (0.892 vs 0.875 at 2 m, clear of the tie), **dropramp 1** (0.790 vs 0.784, inside the tie, so the larger rung wins).

  Apply the §2 fix before choosing anything.
- **(b) Corner scoring: prefer your (b),** per-field 1 − MSE / Var_train(field) in train-standardized units, for F_px and the model alike, with per-field values reported. Reasons: it keeps the field set identical across splits; a field that saturates in the corner counts as trivially predictable, which is the honest reading; and it's bounded and comparable with the iid score. Against the alternatives: dropping saturated fields makes the field set split-dependent, a researcher degree of freedom; pooling R² across standardized fields is dominated by whichever fields still vary. Name it something other than "R²" (e.g. "skill vs train variance") so nobody reads −0.01 as "no transfer".
- **Peeking:** R1-W2 comes from train-split CV and target variances only. The R1-W1 test F_px was seen, as disclosed. My §2 fix uses CV and test values computed after seeing them. Treat it as a correctness fix to rule 4 (it implements the rule as written), not a tuning choice, and **re-register the W\* predictions if they change.**

## 5. Rule 4 code, other points (all fine)
- Noise is added only in view, clipping happens after noise in view, the up vector is zeroed out of view, and the bit is noiseless and exact from the clean centroid. That's fine for a floor; disclose that the camera sees partial bodies at the edge.
- Visibility uses the clean centroid: a body with its centroid just inside is "fully seen". Fine as a floor approximation.
- The noise RNG is seeded per (seed, recording, W), so rungs draw independent noise, which is why W\* moves with the seed. Averaging (a) addresses it.

Scripts used: `r1_leaktest.py` and `r1_strike_ablate.py` (cloud scratch; happy to inline the fix as a patch description if useful: replace `pos` out of view with 0 in both coordinates, before concatenation).
