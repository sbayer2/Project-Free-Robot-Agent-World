# Preregistered: does coupling strength buy COHERENCE, not just gain? (F23)

> Status: **PREREGISTERED 2026-07-30**, frozen before any run.
> Amendments are dated additions, not edits.
>
> Decision rules are executable and were committed **before** the runs:
> `scripts/f23_coherence_test.py` (H1) and `scripts/f23_extraction_test.py` (H2).
> Neither result may be cited as support for the other — see §6.

---

## 1. Why this experiment exists

F22a answered "world or architecture" using **prediction gain**. It did not touch
the metric the project was founded on.

**Coherence** — nudge `z`, and ask whether the render decoder and the behavior
decoder respond *together* — is the quantity that directly tests the Marble
claim: *one latent, two projections*. Gain only asks whether behavior is
predictable at all; a model could predict well through two internally separate
pathways and satisfy gain while refuting the thesis.

Coherence has **never been measured above r = 0.263.** F13's +0.146 (20 seeds,
t ≈ 8.3) was measured entirely in the weak-coupling world that F22a has now shown
was roughly half the strength needed for the essence to be learnable at all. So
the project's signature number was measured in the one condition we now know is
degenerate.

**The question:** in a world that demonstrably contains the signal (r ≈ 0.64,
where the model extracts 78.9 % of reachable gain), does the shared latent become
*more unified*, or does it predict better through pathways that stay separate?

Either answer is publishable, and they say opposite things about the thesis.

## 2. The metric, exactly

From `models/coherence.py`: for latent `z` and iid Gaussian directions `d`
(**not** unit-normalized — normalization creates a simplex artifact that makes
disjoint subspaces anti-correlate),

```
j_r = d/dε f_render(z + εd)|₀      j_p = d/dε f_behavior(z + εd)|₀
coherence = corr(‖j_r‖, ‖j_p‖)   over sampled directions
```

A correlation, so bounded [−1, 1]. Three reference levels:

| level | meaning | expected |
|---|---|---|
| independent (disjoint latents) | floor / control | ≈ 0 |
| untrained shared | what the architecture gives free | > 0 |
| trained shared | what training produces | > untrained if training couples |

**`learned_coherence = trained_shared − mean(untrained_shared)`** is the only
load-bearing number (F10: a *collapsed* encoder scores higher raw coherence, so
raw coherence rewards the degenerate solution).

## 3. The normalization decision — fixed now, not after

This is the fork that must be closed in advance, because it is exactly the error
made and corrected in F22a (`docs/ABO_COUPLING.md` A5): comparing a new world's
number against a reference computed on the old world.

**Rejected — "coherence exceeds +0.146 raw".** F13's +0.146 was measured at
r = 0.263. The untrained-shared baseline is computed from *images*, and images
change with coupling gain, so the free architectural baseline is itself
world-dependent. Comparing against a stored constant repeats A5 exactly.

**Rejected — "coherence as a fraction of a per-world ceiling".** Unlike gain,
coherence has no oracle. There is no established way to compute the maximum
coherence achievable in a given world, and inventing one now — unvalidated,
after seeing that the raw comparison is problematic — would be a worse error than
the one being avoided.

**ADOPTED — re-measure the reference world in the same run.** Both arms are
measured together, same code, same seed protocol, same harness:

```
PRIMARY = learned_coherence(r = 0.636) − learned_coherence(r = 0.263)
```

with `untrained_shared` recomputed **per arm** (≥ 5 untrained seeds each). This
sidesteps the normalization problem rather than solving it: no cross-world
constant is used, and no ceiling has to be invented. F13's +0.146 becomes a
**validity check on the r = 0.263 arm**, not the comparison target — if that arm
does not reproduce F13 approximately, the harness has drifted and nothing in the
run is comparable to the historical record.

## 4. H1 — the coherence hypothesis

**Arms.** r = 0.263 (`data/pm_f22_base`) and r = 0.636 (`data/pm_f22_g15`).
Both datasets already exist; scenes, physics answer key and held-out corner are
**identical** across them (only appearance differs), so the comparison is matched
by construction. 8 trained seeds per arm; 5 untrained seeds per arm.

**Decision rule** (`scripts/f23_coherence_test.py`, committed before the runs):

| verdict | condition |
|---|---|
| **COHERENCE RISES** | PRIMARY ≥ +0.10 **and** Welch t ≥ 2.5 **and** ≥ 6/8 seeds above the r=0.263 mean |
| **NULL** | \|PRIMARY\| < 0.05 |
| **INCONCLUSIVE** | anything else — report, do not claim |

**Gates, all of which must pass before the verdict is read:**

- **Collapse gate (F10).** Every included seed must have latent PR ≥ 8 (F12's
  healthy band). Collapsed seeds are excluded *and reported*, never silently
  dropped — a collapsed encoder inflates raw coherence.
- **Control gate.** The independent disjoint-latent baseline must satisfy
  \|indep\| < 0.05 on both arms. If the floor is not ≈ 0, the metric is
  misbehaving and nothing is interpretable.
- **Validity gate.** The r = 0.263 arm's `learned_coherence` must land in
  [0.08, 0.22], bracketing F13's +0.146. Outside that, the harness has drifted
  from the historical record and the run is void by its own terms.

**What each outcome means.**

- **RISES** — the thesis survives its sharpest test: a world with a real signal
  produces a *more unified* latent, not merely a better predictor. This closes
  the loop the project was founded on.
- **NULL** — the sharper negative. At r ≈ 0.64 the model extracts 78.9 % of
  reachable signal (F22a) while its two heads stay as separate as at r = 0.26:
  **prediction improves through pathways that do not unify.** That is a direct
  negative on "one latent, two projections", and it is stronger than anything
  F18–F21 established, because it holds in a world we have *proven* contains the
  signal rather than one that might have been too faint.

## 5. H2 — the extraction-decline hypothesis (registered separately)

F22a observed that *fraction of reachable signal extracted* falls above r ≈ 0.64:
78.9 % → 44.6 % → 40.4 % as the ceiling (2.494 → 2.789 → 3.107) outruns the model.
Welch t 2.93 vs the max-coupling arm.

**This is not yet a finding.** The pre-registered rule in F22 was written for Δ,
and this metric was adopted after Δ returned inconclusive — post-hoc metric
selection does not become sound because the number is larger.

**Fresh-seed requirement.** The hypothesis was generated by seeds 0–7. It will be
tested on **seeds 8–15 only**. Confirming a hypothesis on the data that generated
it is not a test.

**Metric.** `fraction = (model_gain − shape_oracle) / (fair_ceiling − shape_oracle)`,
with `fair_ceiling` = the `shape+appearance` oracle recomputed **per world**.

**Decision rule** (`scripts/f23_extraction_test.py`):

| verdict | condition |
|---|---|
| **DECLINE CONFIRMED** | fraction(r=0.636) − fraction(r=0.992) ≥ 0.20 **and** Welch t ≥ 2.5, on fresh seeds |
| **NULL** | \|difference\| < 0.08 |
| **INCONCLUSIVE** | anything else |

## 6. Independence of the two hypotheses

They share datasets and compute but **not** inference. Explicitly:

- A RISES verdict on H1 does not license any statement about H2, and vice versa.
- Neither may be described as "supporting", "consistent with", or "corroborating"
  the other in any write-up.
- If one passes and one fails, both are reported at equal prominence. The
  temptation this rule exists to block is reporting the passing one as the
  headline and the failing one as a caveat.

## 7. Cost — what actually has to be run

Counted against what exists on disk, not assumed. `g15_cont` and `loud_cont` have
8 seeds each; **`base_cont` has only 3.**

| need | runs | why |
|---|---|---|
| `base_cont` seeds 3–7 | 5 | the reference arm must match the strong arm at 8 seeds |
| render-only + behavior-only, per arm | 4 | the independent baseline. `--behavior-weight 0 --essence-weight 0` and `--render-weight 0` respectively (masked-loss training). **Without these the control gate is UNMET and H1 returns VOID** |
| H2 fresh seeds 8–15 at r = 0.636 and r = 0.992 | 16 | the fresh-seed requirement in §5 |
| **total** | **25** | ~75 min at ~3 min/run |

H1's coherence evaluation itself is cheap (48 directions, finite differences on a
frozen `z`) and needs no additional training beyond the above.

*(Recorded because the first draft of this section claimed H1 was "eval-only,
seeds already exist" — checking the disk showed the reference arm three seeds
short and the independent baseline entirely absent. A cost estimate that has not
been checked against the filesystem is a guess.)*

## 8. Honest limits

- Coherence is measured on the **behavior** target; the essence-target variant
  (F13 reported both) is secondary here and not gated.
- Both arms are synthetic and authored. Nothing here speaks to real objects; the
  ABO route remains blocked by F22b's outlier-dominated outcomes.
- 20 test scenes in the held-out corner. Every number carries that.
- H2's fresh-seed design tests the *hypothesis*, not the *effect size* — the
  point estimate from seeds 0–7 remains biased upward by selection and must not
  be quoted as the magnitude.
