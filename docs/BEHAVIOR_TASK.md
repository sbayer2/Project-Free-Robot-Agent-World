# The behavior-based task (v2)

This is the redesign that makes the project test something worth testing. It
replaces two weak choices from the first version with two stronger ones, after
the original task turned out to be too easy to be meaningful.

## What was wrong with v1

1. **Materials were a 10-row lookup table.** A model could "succeed" by
   recognizing which of 10 named materials it saw and looking up three numbers —
   memorization, not understanding. Worse, material identity was constant across
   shapes, so "glass looks like glass" transferred trivially and the held-out
   split was nearly free.
2. **The target was three static constants** (density, friction, restitution).
   But the project's whole premise (README) is that the *essence of a thing is
   what it does when you act on it*. Predicting constants isn't that.

## The two fixes

### 1. Continuous materials (`materials.MaterialSampler`)

Instead of 10 names, we sample a hidden **essence** from four latent factors —
*heaviness, grip, hardness, clarity* — and produce **both** appearance and
physics from it:

- physics: `density` (log-uniform 50–8000), `friction` (0.05–1.1),
  `restitution` (0.05–0.85);
- appearance: `metallic`, `roughness`, `transmission`, colour — each a function
  of the factors **plus Gaussian noise**.

The noise makes appearance *predictive but not invertible*: the model must infer
the essence from visual evidence, not read a label. A unit test confirms the
coupling is real but imperfect (correlation between "looks metallic" and density
sits in a band, not at 1.0). The 10 named materials survive as labelled
**anchors** for interpretable evaluation (`nearest_anchor`).

### 2. Behavior outcomes under probes (`probes.py`)

We *act* on each object with three probes and record the **outcome** (summary
statistics — the chosen granularity):

| Probe | Action | Reveals |
|---|---|---|
| **Drop** | release from a height | restitution + mass (bounces, settling, peak height) |
| **Tilt** | place on a ramp | friction (does it slide, how far) |
| **Push** | horizontal impulse at a height | friction + mass + **shape**: slide vs **topple** |

`PUSH` is the important one: a tall cylinder topples where a squat box slides
under the same impulse, so **shape and material interact**. "Glass is glass" can
no longer shortcut across shapes — the held-out test now has teeth.

Outcome fields: `toppled`, `settle_time`, `slid_distance`, `n_bounces`,
`max_height`, `path_length`, `final_tilt_deg`. Summarization is pure-Python over
a recorded trajectory, so it is unit-tested with synthetic toppling / sliding /
bouncing trajectories — no MuJoCo required.

## Generalization: hold out a region, not a list (`splits.RegionHoldout`)

With continuous materials there's no grid to hold combinations out of. Instead we
reserve a **box in normalized essence-space** for test. Two flavors, and the
distinction matters (it was an audit finding):

- **Interpolation** (`DEFAULT_REGION_HOLDOUT`) — an *interior* box (mid-grip,
  fairly-bouncy), surrounded by training data on all sides. A smooth model fills
  an interior hole almost trivially, so this is a **weak** test. ~8% of samples.
- **Extrapolation** (`EXTRAPOLATION_REGION_HOLDOUT`, now the **default** for
  generation) — the **heavy *and* bouncy corner** (`density≳0.55` **and**
  `restitution≳0.60`, normalized). Training has heavy objects and bouncy objects
  but never both-extreme together, so for held-out points no training object lies
  jointly beyond them — the model must extrapolate the coupling **outside the
  training manifold's convex hull**. ~3–4% of samples. This is the test *with
  teeth*.

Select with `--holdout-kind {extrapolation,interpolation}`; the choice + bounds
are recorded in the manifest's `holdout_region`. Report `learned_coherence` and
behavior generalization on the **extrapolation** split for the honest result.

## The schema (v2)

Each scene's `sample.json` now carries (additions in **bold**):

```jsonc
{
  "scene_id", "split", "generator",
  "schema_version": 2,
  "input": { "shape", "material": "<scene id for sampled materials>" },
  "appearance": { "resolution", "frames": [...] },   // what the model sees
  "physics":  { "raw", "normalized", "tags" },        // the essence labels
  "behavior": { "probes": [                            // ** the primary target **
     { "probe": "drop", "spec": {...}, "outcome": {...} },
     { "probe": "tilt", "spec": {...}, "outcome": {...} },
     { "probe": "push", "spec": {...}, "outcome": {...} }
  ]},
  "material_truth": {                                  // ** ground truth, not input **
     "factors": { "heaviness", "grip", "hardness", "clarity" },
     "nearest_anchor": "...",
     "appearance_params": { ... }
  }
}
```

`material_truth` is recorded for analysis only — the model never sees it; it lets
us measure how well the model *recovers* the hidden essence from images alone.

## How the model/experiment change (next, not built yet)

```
multi-view images ─┐
                    ├─► encoder ─► z (inferred essence)
probe spec (action)─┘             │
                         render head        behavior head
                    (reconstruct look)   (z + action → outcome)
```

- The behavior head is **action-conditioned** — from appearance, predict the
  consequences of acting (Li's "planner is the inverse of the renderer").
- Headline experiment, restated: does inferring **one shared essence** predict
  behavior on **held-out essence regions and shape×material combos** better than
  **two independent models** glued together? Yes → the shared-understanding idea
  earns its keep. No → an honest null we will report.

## Honest risks of this version

- **Chaos near boundaries.** An object balanced at its tipping point is sensitive
  to tiny differences → noisy `toppled` labels. Measured in `docs/FINDINGS.md` F8
  (sharp but localised; ~2–4% of scenes). **Mitigation implemented:** the MuJoCo
  generator's `--topple-jitter-reps K` runs the push K extra times with small action
  jitter and records `toppled` as a smooth P(topple) ∈ [0,1] instead of the hard
  bool (same 21-dim target; the sim is deterministic, so the jitter is on the
  *action*). Default off (binary). Prefer the continuous `final_tilt_deg` too.
- **Harder learning problem** than regressing three numbers — more compute, more
  careful design. That cost is the price of the task meaning something.
- **Appearance must really encode the essence.** With MuJoCo's simple renderer,
  the appearance map is a design choice we have to keep honest, or there's no
  coupling to find.
