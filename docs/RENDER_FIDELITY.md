# Preregistered: the render-fidelity ladder (F21)

> Status: **PREREGISTERED 2026-07-19**, frozen before the runs.
> Amendments are dated additions, not edits. Tiered by cost: the cheap arm runs
> first and gates whether the expensive arms run at all.

## The question, and why it is NOT one experiment

F20 closed the F18/F19 arc with "the gap is render-fidelity-bound": forcing the
essence-bearing channels (roughness/metallic/transmission) into `z` raised the
achievable behavior gain only 1.46 → 1.57, far short of the 2.26 clean-param
ceiling, because at 128px those channels sit in `z` at R² 0.2–0.5 — too noisy to
jointly localize the material. But "render-fidelity-bound" was never *isolated*
from a second authored degradation. There are two, on different links of the
generative chain:

```
essence --[appearance_noise 0.07 : LINK 1]--> appearance_params
        --[render: 128px, single top-down light : LINK 2]--> pixels --> z --> behavior
```

- **Link 1 — `appearance_noise` (`materials.py:251`).** Gaussian noise on the
  essence→appearance map, so `appearance_params = f(essence) + noise`. The F18
  oracle used the *recorded* (post-noise) params and still reached 2.26, so this
  noise is already baked into that ceiling. Lowering it makes appearance a
  cleaner essence proxy. **Cheap:** a `--appearance-noise` generator flag +
  regenerate + retrain at 128px.
- **Link 2 — render legibility (`generate_mujoco.py:192`).** A single
  straight-down light + flat ambient washes out the specular cues that roughness
  and metallic produce; 128px undersamples them. This sets the CNN-reachable
  ceiling (untrained-`z` R² roughness 0.22, metallic 0.53, transmission 0.23).
  **Moderate:** `--resolution 256` (flag exists) + an MJCF lighting change +
  retrain at 4× pixels.

F20's diagnosis pointed at Link 2, but the experiment that would isolate it —
holding the render fixed and relieving Link 1 — was never run. F21 runs it, and
only pays for the Link-2 arm if Link 1 does not already close the gap.

## Design — a cost-tiered ladder

**Arm 1 (cheap, runs first): `appearance_noise` sweep at fixed 128px render.**
Regenerate `pm_big` at `appearance_noise ∈ {0.07 (baseline), 0.03, 0.0}`,
holding resolution, views, lighting, and the split fixed. Retrain 3 seeds each
at `appearance_weight = 0.3` (the F20 winner), lr 5e-4, 50 epochs. Per noise
level, measure on the corner split: the F18 oracle ceiling (shape+appearance),
the **fresh-head-on-`z` ceiling** (the F20 disambiguator), the z→appearance
retention, and the trained behavior gain.

**Arm 2 (moderate, GATED on Arm 1): resolution 128 → 256 + oblique lighting.**
Runs only if Arm 1's P3 falsifier fires (noise=0 does not lift the ceiling).
Regenerate at 256px with a varied/oblique light (an MJCF change so roughness and
metallic cast legible specular highlights), retrain, re-measure.

**Arm 2 GATED IN and run parameters (added 2026-07-19, after Arm 1).** Arm 1's
P3 falsified backwards — removing `appearance_noise` *widened* the oracle−model
gap (0.80 → 1.01), so legibility (Link 2), not authored noise, binds. Arm 2 runs.
Concrete recipe, frozen here before the runs: regenerate at `--resolution 256
--lighting oblique --appearance-noise 0.07` (three fixed oblique specular lights
+ low ambient, `RenderConfig.lighting`; noise held at the pm_big value so only
the render changes), same 512 scenes / seed 1234 / extrapolation split; retrain 3
seeds at `appearance_weight = 0.3 --image-size 256`, lr 5e-4, 50 epochs; evaluate
with `render_fidelity_eval.py` against the **256px oracle ceiling** (recomputed
on the new render, not the 128px 2.28). The single-arm comparison is oblique-256
vs the Arm-1 flat-128 noise-0.07 baseline (oracle 2.28, trained gain 1.48,
reach 0.18/0.50/0.20).

## Registered predictions

- **P1 (manipulation check — the knob works):** the F18 oracle ceiling
  (shape+appearance) rises monotonically as `appearance_noise` falls; at
  noise = 0 it **exceeds 2.26** (appearance becomes a near-deterministic essence
  proxy). Near-mechanical; if it fails, the regeneration is wrong.
- **P2 (the two degradations are separable):** the CNN-reachable ceiling
  (untrained-`z` R² of roughness/metallic/transmission) stays ≈ 0.22 / 0.53 /
  0.23 across all three noise levels — because `appearance_noise` changes what
  value a channel takes, not how that value renders into pixels. *Falsifier:*
  reachability rises as noise falls ⇒ the links are not separable and the
  two-degradation model is wrong.
- **P3 (the crux + the gate):** lowering `appearance_noise` raises the
  fresh-head-on-`z` ceiling above the F20 value of 1.57 by **≥ +0.10** at
  noise = 0. **Decision rule:** if the noise = 0 fresh-head ceiling reaches
  **≥ 1.90**, Link 1 dominated — the barrier was essence↔appearance decoupling
  (cheap), F20's "render-bound" is reframed as partly authored-noise-bound, and
  **Arm 2 is NOT run.** If it stays **≤ 1.70**, Link 2 dominates — pixel
  legibility is the real floor, F20's diagnosis stands, and **Arm 2 runs.** The
  1.70–1.90 band is "both matter"; run Arm 2 to apportion.
- **P4 (Arm 2, only if gated in):** at 256px + oblique lighting the reachable
  ceiling (untrained-`z` R² of roughness/metallic) rises above 0.22 / 0.53, and
  the fresh-head ceiling climbs past 1.57 — isolating pixel legibility as the
  binding degradation. *Falsifier:* neither reachability nor the ceiling moves
  ⇒ the 128px→256px + lighting change is insufficient and the render limit is
  deeper than resolution/lighting (e.g. the color hue being an authored random
  cue starves the material signal regardless).

## What each outcome decides

| Arm 1 P3 | reading | next |
|---|---|---|
| ceiling ≥ 1.90 at noise 0 | the gap was authored decoupling, cheap | ship a low-noise `pm_big`; re-run F13 coherence + F18 oracle; **stop the ladder** |
| ceiling ≤ 1.70 | genuinely render-legibility-bound (F20 confirmed) | run Arm 2 (resolution + lighting) |
| 1.70–1.90 | both degradations bind | run Arm 2 to apportion the remainder |

This is the honest test of F20's "render-bound" claim: it puts a cheap
experiment between the claim and the expensive fix, and only spends the Mac
hours on resolution+lighting if the cheap knob provably cannot close the gap.

Artifacts to `runs/render_fidelity/` and `data/pm_noise*` (gitignored). Mac step.
