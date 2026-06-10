# Predictive coding — a 25-year-old idea this project quietly re-implements

This is a *connection note*, not a result. It records why two pieces of
pseudo-marble — the **render head's reconstruction loss** and the **coherence
control** (`learned = trained − untrained`) — are, on inspection, small
re-statements of ideas from computational neuroscience that predate the hardware
that now makes them runnable. Read [`ARCHITECTURE.md`](ARCHITECTURE.md) and
[`FINDINGS.md`](FINDINGS.md) first; this just draws the line back to the theory.

The point is not to dress the project in borrowed authority. It's the opposite:
the mechanism we arrived at by engineering — minimize prediction error, then
*subtract the prior to see what learning actually added* — is the same one Rao &
Ballard wrote down in **1999**, on machines that couldn't train it at any scale.
That a four-core laptop today runs the thing their paper could only describe is
itself part of the story.

## The source idea (Rao & Ballard 1999, and predictive coding generally)

Rao & Ballard's *"Predictive coding in the visual cortex"* (Nature Neuroscience,
1999) proposed that the brain is not a feed-forward classifier but a hierarchy of
**generative predictors**: each level sends a **top-down prediction** of the
activity below it, the lower level returns only the **prediction error** (what the
prediction missed), and learning is driven by minimizing that residual. The
representation a level holds is whatever best *predicts* its input — appearance is
explained by generating it, not by reading it off.

Two consequences matter here:

1. **Perception = generation checked against reality.** A model "understands" an
   image to the extent it can predict/reconstruct it; the error signal is the
   teacher.
2. **The interesting quantity is the residual, not the raw signal.** `error =
   observation − prediction`. The prediction (the prior, the expected) is
   subtracted off on purpose; what's left — the part the prior *didn't* already
   explain — is what carries information and drives change.

Both reappear in this codebase without us having set out to copy them.

## Mapping 1 — the render head is a literal prediction-error term

The shared-latent model emits three projections from one `z`; one is the
**render head**, a conv decoder `z → mean-view image`, trained with

```
L_render = mean( (decode(z) − observed_mean_view)² )
```

That is a top-down **prediction of appearance** (`decode(z)`) compared against the
**observation** (the pose-averaged view), with the squared **residual** as the
loss. In predictive-coding terms it is exactly a prediction-error term at the
sensory level: the latent must *generate* the look of the object, and is corrected
by where its generation differs from what was seen. We did not add the render head
to be photoreal (we explicitly chose a conv decoder over a Gaussian-splat one
precisely because we measure coherence, not photorealism — see
[`ARCHITECTURE.md`](ARCHITECTURE.md#render-head-the-appearance-projection)). We
added it so appearance is forced through the same `z` that has to explain
behavior. The Rao–Ballard reading is a free reinterpretation: the appearance
projection *is* the perceptual prediction error, and minimizing it is the same
move as "explain the input by generating it."

This is a tight, literal mapping. The caveat (below) is that ours is **offline**
error minimization by gradient descent over a dataset, not the **online**,
moment-to-moment error settling Rao & Ballard meant as a mechanism of cortex.

## Mapping 2 — the coherence control is "subtract the prior, keep the residual"

The sharper parallel is in the **measurement**, not the model. [`FINDINGS.md`
§F6](FINDINGS.md) records the project's most important sandbox finding: an
*untrained* shared model already scores high coherence (~0.45) purely from
architecture, because both heads read one `z`, so any latent nudge moves both. The
naive "shared beats independent" number is therefore mostly an **architectural
prior**, not evidence of learning. The honest signal is

```
learned_coherence = trained_shared − untrained_shared
```

baked into `coherence_bench.compare()` as `architectural_coherence` /
`learned_coherence`.

Structurally this is the **same operation as a prediction error**. The untrained
(architectural) coherence is the **prior** — what you'd expect *before* learning,
from the shape of the network alone. We subtract it off, exactly as predictive
coding subtracts the top-down prediction, and keep only the **residual**: the part
of the coupling that training actually added beyond the architectural expectation.

```
prediction error  =  observation        −  prediction (the prior)
learned coherence  =  trained coupling   −  untrained coupling (the architectural prior)
```

In both cases the prior is not noise to be ignored; it is a *known expectation*
that must be explained away so the informative remainder stands clear. Reporting
raw shared coherence would be the analogue of reporting raw cortical activity and
calling all of it "signal" — most of it is the prediction the system would have
made anyway. The control is what converts a vibe into a measurement, and it is the
predictive-coding instinct applied across *model states* (trained vs. untrained)
rather than across *time* (now vs. predicted).

## Mapping 3 — the renderer / simulator / planner triad

[`README.md`](../README.md) and [`TAXONOMY_NOTES.md`](TAXONOMY_NOTES.md) frame the
project on Li's taxonomy: **renderer** (outputs pixels), **simulator** (outputs
state), **planner** (outputs actions — *"the inverse of the renderer"*). Lined up
against the predictive-coding / active-inference picture, the triad is the same
three faces:

| Li's taxonomy | Predictive coding / active inference |
|---|---|
| Renderer — predict pixels | Perception — generate the sensory prediction |
| Simulator — predict state under dynamics | Forward model — predict consequences |
| Planner — choose actions (inverse render) | Active inference — act to minimize expected error |

pseudo-marble currently implements the first two (render + behavior/essence heads)
and *measures* whether they share one latent. The third — the planner, the action
arm — is the open socket: the place where "close the loop / let a planner write
back into `z`" would turn the instrument from a passive predictor into an
active-inference agent. The `probes.py` drop/tilt/push actions are the data-side
stub of that arm; the model side is unbuilt by design.

## The hardware footnote the user emphasized

Rao & Ballard described hierarchical error-minimizing generative models in 1999 —
on hardware that could not train them at any useful scale. The reason ideas like
this "circle back" now is not that they were wrong then; it's that
GPU-parallelized training (and, for this project, Apple-silicon MLX/Metal) is the
substrate that finally lets the idea *run*. pseudo-marble is a small worked example
of that loop: a 25-year-old hypothesis about perception-as-prediction, re-derived
from engineering pressure (force both outputs through one `z`; subtract the
architectural prior), and executed on commodity hardware that didn't exist when the
hypothesis was written. The academic value here is the synthesis — new framing
pulled out of the build, correlated to an older discipline, made runnable.

## Honest caveats (kept, not buried)

- **Offline control ≠ online mechanism.** Rao & Ballard propose a *dynamical
  process* in cortex — errors settling in real time. We do batch gradient descent
  over a dataset and a one-shot trained-minus-untrained subtraction. The *algebra*
  matches (`observation − prediction`); the *temporal mechanism* does not. This is
  analogy at the level of the objective, not a claim that the network is a cortex.
- **"Foundation of perception" overstates it.** Predictive coding is one
  influential theory among several (and remains debated empirically). Pointing out
  that our loss term has its shape does not validate the theory or the project.
- **Says nothing about consciousness or biological realism.** No spiking, no
  cortical microcircuit, no hierarchy of error units. One latent, three heads, a
  finite-difference probe. The mapping is conceptual.
- **The mapping doesn't change any result.** Nothing here moves a number. The
  headline result (`learned_coherence` on held-out essence regions, on the Mac) is
  still unmeasured ([`FINDINGS.md` §3](FINDINGS.md)). This note reframes; it does
  not report.

In short: the render head is a prediction-error term, and the coherence control is
prediction-error logic applied across training rather than across time. Neither is
a coincidence worth overselling — but the line back to Rao & Ballard is real, and
worth writing down so the connection lives in the repo rather than in a chat log.

---

*Connection note, not a result. Personal research; not affiliated with World Labs.
Reference: Rao, R. P. N. & Ballard, D. H. (1999), "Predictive coding in the visual
cortex: a functional interpretation of some extra-classical receptive-field
effects," Nature Neuroscience 2(1), 79–87.*
