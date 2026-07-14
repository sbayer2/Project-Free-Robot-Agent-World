# Preregistered: the oracle ceiling (F18)

> Status: **PREREGISTERED 2026-07-14**, frozen before the numbers are read.
> Amendments are dated additions, not edits.

## Why this runs before the benchmark upgrade

F17 closed with a prescription: "to see a bigger essence, the benchmark needs
more held-out scenes and harder prediction targets." The first half does not
survive scrutiny. **More held-out scenes shrink the error bar on the gain; they
cannot move the gain.** If the continuous model's true held-out gain is 1.36,
then k=1 trit matching it at 1.39 means three latent states genuinely capture
what this benchmark rewards. Scaling `pm_big` 512 → 4096 scenes costs Mac hours
and returns a tighter CI on the same 1.36. It cannot reveal a bigger essence.

The question F17 actually left open is: **is 1.6 bits a fact about the
instrument or about the world as authored?** That is answerable with zero
training runs, because `sample.json` ships `physics.raw` — the exact generator
inputs. Fit those directly to the behavior labels and you measure the
benchmark's information ceiling, independent of any encoder.

There is a specific reason to suspect the answer is uncomfortable. `pm_big` has
**5 shapes** — log2(5) = 2.3 bits. F17's saturation sits at 1.6 bits. Shape is
trivially recoverable from the renders and trivially drives the probes
(a sphere rolls, a box does not). So the model's entire 1.36 gain may be shape
recognition, with the hidden essence contributing nothing measurable to
held-out behavior. F13's coherence would still be real, but its interpretation
would shift from "the latent holds a physical essence" to "the latent holds
shape, which is visible in appearance and consequential in behavior" — a far
weaker claim, and one the F10 law does not catch (a shape-driven model is not
collapsed; its PR is healthy).

## Design

**Oracle** = a regressor from *known generator inputs* → the 21-dim normalized
`behavior_vector()`. No renders, no encoder, no training loop. Feature arms:

| arm | features | question it answers |
|---|---|---|
| `shape` | shape one-hot (5) | how much gain is available with **zero** essence? |
| `essence` | density, friction, restitution (normalized) | essence alone, no shape |
| `shape+essence` | both | the full ceiling: everything the generator knows |
| `shape+essence(b bits)` | shape + essence quantized to 2^b clusters | the oracle's **bit curve**, comparable to F17's k-sweep |

**Metric.** Identical to the project's: `gain = MSE(predict train-mean) /
MSE(oracle)`, on the same held-out set, same normalizers.

**Splits.** Two, deliberately:
- `corner` — train on the train split, test on the heavy+bouncy extrapolation
  corner. Apples-to-apples with the trained model (F13/F17).
- `iid` — a random 80/20 split of the training-region scenes only. Removes
  extrapolation as a confound.

**Regressor.** numpy ridge on {linear, quadratic-expanded} features, plus k-NN;
report the best per arm. A ceiling estimated this way is a **lower bound** —
some better regressor may exist. The honesty gate (P4) guards the inference.

## Registered predictions

- **P1 (the headline):** the `shape`-only oracle reaches **gain ≥ 1.25** on the
  corner — i.e. ≥ ~70% of the trained model's 1.36 comes from shape alone, with
  no essence whatsoever. Rationale: F17's flat-in-k curve, F16's
  probes-are-binding result, and 1.6 bits ≈ 3 buckets ≈ the shape vocabulary.
  *Falsifier:* `shape`-only gain ≤ 1.10 ⇒ shape is not the story; the essence
  genuinely contributes and F17's reframe stands as written.
- **P2:** `shape+essence` exceeds `shape`-only on the corner by **≥ +0.30
  gain**. Rationale: unlike ABO (F14, assumed contact params), `pm_big`
  *samples* friction and restitution, so tilt-slide and push-displacement must
  depend on them. *Falsifier:* the two arms tie.
- **P3:** the essence contribution (`shape+essence` − `shape`) is **larger on
  `iid` than on `corner`** ⇒ extrapolation, not probe design, is what eats it.
  *Falsifier:* the contribution is ~0 on `iid` too ⇒ the probes fail to expose
  essence to even a perfectly-informed predictor, corroborating F16 on the
  synthetic side.
- **P4 (honesty gate):** the oracle must fit TRAIN to **gain ≥ 2.0**. If it
  cannot fit the data it was given the true answers for, the regressor is too
  weak and this test is VOID — report that, not a ceiling.

## What each outcome decides

| P1 | P2 | reading | next step |
|---|---|---|---|
| true | false | probes expose no essence even to an oracle | **probe redesign is the only route**; dataset scaling is definitively wasted |
| true | true | the information is in the labels; the model doesn't extract it | model/eval binding — F17's upgrade is half-right, pursue it |
| false | — | essence genuinely contributes; 1.6 bits is real | F17's reframe stands; scale the benchmark |

P3 splits the first row further: if `iid` shows an essence contribution the
`corner` does not, the extrapolation split — not the probe battery — is the
binding constraint, which would be the first evidence against F16's reading.
