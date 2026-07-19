# Preregistered: the appearance-reconstruction auxiliary (F20)

> Status: **PREREGISTERED 2026-07-18**, frozen before the training runs.
> Amendments are dated additions, not edits. This supersedes the naive
> "reweight the behavior head" plan, which F19's follow-up killed (below).

## Why the head-reweight plan is dead (recorded before F20 runs)

F19's merged headline — "the appearance is in `z`; the behavior head fails to
USE it (loss/architecture, cheap fix)" — **overclaimed**, and three cheap
measurements taken after it show why. F20's entry will correct F19; the
correction is registered here first.

1. **The behavior-relevant appearance is not what `z` keeps.** Decomposing
   F18's shape+appearance ceiling (2.31) by channel group (corner split):
   - shape only → 1.33
   - shape + color (RGBA) → 1.60 (color adds only +0.27)
   - shape + {roughness, metallic, transmission, ior} → **2.26** (nearly all of it)
   And no *single* channel carries it — every one alone gives ~1.33; only the
   **joint** {roughness, metallic, transmission} reaches 2.245. The three
   individually-weak channels together localize the material identity.
2. **`z` keeps color and discards exactly those channels.** From the F19 probe,
   CNN-reachable (untrained-`z`) vs trained-retained R²: roughness 0.22→0.05,
   transmission 0.23→0.09, metallic 0.53→0.27, while color stays 0.91→0.80. The
   "65% appearance retention" F19 headlined was dominated by behaviorally-inert
   color.
3. **A perfect head on the frozen `z` caps at ~1.46.** Fresh ridge/kNN heads
   `z → behavior` on the corner split score 1.46 / 1.48 (trained model's own
   head: 1.41) — nowhere near 2.26. **The head is not the bottleneck; what `z`
   retains is.** Reweighting the head alone is therefore futile.

The binding constraint is the **encoder**: the render loss rewards color
fidelity (high pixel variance) over the subtle physics-material channels, so the
encoder drops them. F20 tests whether *forcing the encoder to retain them* — an
appearance-params reconstruction auxiliary — moves behavior off shape-only.

## Design

**Architecture.** A new auxiliary head `z → appearance_params` (8 normalized
channels: `base_color` RGBA, `roughness`, `metallic`, `transmission`, `ior`),
mirrored in all three backends, gated on `appearance_weight > 0` so the default
model is byte-identical (old checkpoints still load; the existing suite is
unchanged). Loss gains `+ appearance_weight · MSE(appearance_pred, target)`.
This is the *clean* appearance target, which the renderer used — recoverable
from the pixels up to the CNN-reachable ceilings above (0.22–0.53), not the
authored-noise-free essence.

**Sweep.** `appearance_weight ∈ {0.0 (control), 0.3, 1.0, 3.0}`, 3 seeds each,
`pm_big`, 50 epochs, lr 5e-4 — matching the F13/basin recipe so only the aux
term differs. 12 runs, ~20 min on the M5 Pro.

**Measurement (per weight, mean ± std over seeds, corner test split):**
- **behavior gain** = MSE(predict train-mean) / MSE(model's behavior head) —
  the same metric as F13/F18. Does it climb from 1.41 toward 2.26?
- **z-retention** of the 8 appearance channels (the F19 probe) — did the aux
  actually put roughness/metallic/transmission back into `z`?
- render MSE and essence gain — capacity-competition guards.

## Registered predictions

- **P1 (retention rises — the mechanism precondition):** at the best weight, the
  z→appearance probe shows **metallic R² ≥ 0.40, roughness ≥ 0.12,
  transmission ≥ 0.12** (up from 0.27 / 0.05 / 0.09 at `appearance_weight = 0`),
  moving toward the CNN-reachable 0.53 / 0.22 / 0.23. *Falsifier:* retention of
  these three does not rise ⇒ the aux head failed to reshape `z` and the test is
  inconclusive on behavior.
- **P2 (behavior gain moves — THE HEADLINE):** at the best weight, held-out
  behavior gain rises to **≥ 1.60** (from the 1.41 baseline). *Falsifier:* gain
  stays **≤ 1.45** at every weight ⇒ forcing the channels into `z` does not help
  behavior ⇒ the signal needs higher fidelity than 128px renders carry →
  **render-bound (branch 3)**, and this half of the F18 gap closes as "the
  encoder *can* be made to retain the channels, but at render fidelity they are
  too noisy to use."
- **P3 (mechanism link):** across the sweep, the behavior-gain increase tracks
  the physics-material retention increase (positive rank correlation over the 12
  runs). Retention up **and** gain up ⇒ F20's fix works; retention up but gain
  flat ⇒ render-fidelity-bound (corroborates the P2 falsifier).
- **P4 (honesty guard / capacity cost):** behavior gain is **non-monotone** in
  `appearance_weight` — a high weight spends latent capacity reconstructing
  color and should degrade behavior/essence. If gain instead rises monotonically
  to `appearance_weight = 3.0`, capacity was not the constraint and the aux is a
  free win (report as such).

## What each outcome decides

| P2 | reading | next step |
|---|---|---|
| gain ≥ 1.60 | forcing appearance into `z` recovers behavior signal | tune the weight; then re-run F13 coherence + F18 oracle gap at the winner |
| gain ≤ 1.45 (P3: retention rose) | `z` can hold the channels but they're too noisy at 128px | **render/lighting upgrade (branch 3)** is the only route; F18 gap is render-bound |
| retention did not rise (P1 fail) | the aux head could not reshape `z` | rethink the auxiliary (stronger weight, per-channel loss, or a bottleneck) |

Artifacts to `runs/appearance_aux/` (gitignored). This is a Mac/MLX step.
