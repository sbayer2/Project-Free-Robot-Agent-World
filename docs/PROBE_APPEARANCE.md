# Preregistered: the z → appearance_params probe (F19)

> Status: **PREREGISTERED 2026-07-16**, frozen before the numbers are read.
> Amendments are dated additions, not edits.

## The question F18 left, stated so it can be graded

F18 showed the trained model extracts ~8% of the essence signal the pixels
carry — the encoder is binding, not the benchmark. But "the encoder is binding"
has two mutually exclusive mechanisms, and they prescribe opposite fixes:

- **The encoder cannot SEE the essence.** The 128px renders under fixed
  lighting discard the visual channels (roughness, metallic, transmission, ior)
  before `z` is formed. Then `z` does not linearly encode `appearance_params`,
  and the fix is on the *render/capacity* side (higher resolution, varied
  lighting, refraction cues).
- **The encoder SEES it but the behavior head does not USE it.** `z` linearly
  encodes `appearance_params`, but the behavior head maps only the shape
  component to behavior. Then the fix is on the *loss/architecture* side (raise
  `essence_weight`, predict essence→behavior compositionally).

A linear probe from a **trained, frozen** `z` to the ground-truth
`appearance_params` (recorded in every `sample.json` under
`material_truth.appearance_params`) distinguishes them directly, on the
existing `runs/basin/lrlo_s*` checkpoints, with **no retraining and no data
regeneration**.

## Design

**Targets.** Two sets, both read from `material_truth`:
- **appearance** (8 ch): `base_color` RGBA (4), `roughness`, `metallic`,
  `transmission`, `ior` — the visual channels the renderer used.
- **essence** (3 ch): normalized density, friction, restitution — the hidden
  physical inputs. The trained model already has an essence *head*; F18
  measured its held-out gain at **1.067** (barely above predict-mean). A fresh
  linear probe on `z` asks whether the head is the weak link or the latent is.

**Probe.** Ridge regression `z → target`, standardized, 5-fold CV over all 512
scenes (appearance is per-scene; no extrapolation concern, so all data is used
for a tight per-channel estimate). Metric: **R²** per channel (1 = `z` fully
linearly determines it; 0 = no better than the mean).

**Arms (the F6/B4 discipline — subtract the free baseline):**
- `z_trained` — R² from the 20 trained checkpoints (mean ± std over seeds).
- `z_untrained` — R² from 10 fresh untrained inits. A random encoder's `z` is a
  random projection of the pixels and linearly encodes *some* appearance for
  free; this is the baseline that must be subtracted.
- `pixels` — R² from the mean-view pooled RGB + simple pixel statistics. This is
  the **ceiling**: how much of each channel is linearly available in the image
  at all. Fixed lighting may make `ior`/`transmission` unreadable even here.

**The headline metric** is the *preservation fraction* per channel:
`preserved = (R²_z_trained − R²_untrained) / (R²_pixels − R²_untrained)` — the
share of the pixel-available appearance signal that the trained latent keeps,
above the random-encoder floor. ~1 = `z` preserves it; ~0 = `z` discarded it.

## Registered predictions

- **P1 (color is kept):** for `base_color` RGB, `z_trained` preserves **≥ 70%**
  of the pixel-available signal and reaches **R² ≥ 0.5** absolute. Color is the
  most visible channel and drives the render loss. *Falsifier:* preservation
  < 40% ⇒ even color is being discarded, a stronger encoder-capacity failure
  than expected.
- **P2 (specular cues, partial):** `roughness` and `metallic` are preserved
  **≥ 40%** but strictly less than color — highlights are subtler than hue.
- **P3 (refraction, weak):** `transmission` and `ior` are preserved **< 40%**,
  and are the weakest even in the `pixels` ceiling — 128px fixed lighting barely
  renders refraction. If the `pixels` ceiling itself is near zero here, the
  channel is untestable and reported as such, not as a latent failure.
- **P4 (the diagnosis / headline):** averaged over the 8 appearance channels
  *that the pixels actually carry* (`pixels` R² > 0.1), `z_trained` preserves
  **≥ 50%** ⇒ **the information IS in the latent; the binding constraint is the
  behavior head's USE of it (loss/architecture), not encoder capacity.**
  *Falsifier:* preservation **< 20%** ⇒ the encoder discarded the appearance;
  the binding constraint is render/capacity, and raising `essence_weight` would
  be futile.
- **P5 (essence, the head-vs-latent split):** a fresh linear probe `z → essence`
  beats the trained essence head's 1.067 gain on **density** (heaviness is
  authored into color-dimming, so it should be visible), but not on
  `restitution`. If the probe beats the head on **all three**, the essence head
  is simply under-optimized (a loss-weight fix); if it beats on **none**, `z`
  never carried essence and only appearance is recoverable.

## What each outcome decides

| P4 | reading | next step |
|---|---|---|
| ≥ 50% preserved | `z` sees appearance; behavior head ignores it | raise `essence_weight`; compositional essence→behavior head; **no regeneration needed** |
| < 20% preserved | encoder discarded appearance at 128px | render upgrade (resolution, lighting) — the expensive path |
| 20–50% | partial; channel-by-channel (P1–P3) says which | mixed |

P5 refines the top row: if the probe recovers essence from `z` far better than
the trained head does, the very next experiment is just re-training with a
higher `essence_weight` — the cheapest possible fix for the F18 gap.

## Amendment 2026-07-16 (caught at smoke-test, before the full multi-seed run)

The registered **preservation-fraction** metric is mis-specified and is
replaced by **retention**. Reason, found on the first (s0 + 1 untrained) smoke
run: the untrained encoder decodes appearance *better* than the trained one
(e.g. color_r R² 0.920 untrained vs 0.874 trained). This is not noise — a
random linear projection is near-**lossless** for linear structure
(Johnson–Lindenstrauss), so a random encoder's `z` is close to the pixel
ceiling for any linearly-present channel, while training deliberately warps `z`
toward its heads and mildly *discards* linear appearance it does not need. The
F6/B4 "subtract the untrained baseline" logic therefore inverts here: training
can only reduce linear appearance-decodability, so `(trained − untrained)` is
≤ 0 and `(pixels − untrained)` ≈ 0, and the fraction explodes.

**Corrected metric (used for grading):**
- **Absolute `z_trained` R²** — is the channel present in the trained latent?
  P1's "R² ≥ 0.5 absolute" clause is graded directly on this and is unaffected.
- **Retention = `z_trained` R² / `z_untrained` R²**, only where `z_untrained`
  R² > 0.1 (else nothing linear is there to keep). This is the fraction of the
  random-encoder near-ceiling that survives training. **Retention ≈ 1 ⇒ the
  appearance is kept in `z` ⇒ the behavior head's failure to USE it is binding
  (loss/architecture). Retention ≪ 1 ⇒ training discarded it.** P4's diagnosis
  is re-expressed on retention: ≥ 50% retained ⇒ loss/architecture; < 20% ⇒ the
  discarding happens in the encoder itself.

The registered *directional* predictions (P1 color strong, P2 specular partial,
P3 refraction weak, P5 essence head-vs-latent) are graded as written on
absolute R²; only the aggregation metric changed, and only because it was
provably broken. Recorded here before the 20-seed run so the change is auditable.

## This is a Mac/MLX step

Encoding needs MLX (the checkpoints are `.safetensors` MLX weights); it runs on
the M5 Pro. The probe itself is numpy. Budget: ~30 seed-encodes of 512 scenes ×
16 views — minutes, not hours. Artifacts to `runs/probe_appearance/`
(gitignored).
