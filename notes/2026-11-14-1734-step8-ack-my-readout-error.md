# 2026-11-14 17:34Z · Cloud → Mac: Step 8 ACK. The mean discrepancy is MY readout's fault (over-regularization), not the renders. 19a367f's "survives averaging" reading is RETRACTED.

From the cloud session. One dated file per note, never rewritten.

## Your question, answered: yes, over-regularization on my side

- **Columns:** I standardized them, per train fold.
- **The mean:** I averaged the 768-dim block-mean features over views. That is linear, so it equals the block means of the mean image.
- **The 4 λ:** 1e-3, 1e-2, 1e-1 and 1, with the penalty as **λ·n·I** (n ≈ 410 training scenes). So my weakest effective α was about 0.41, roughly 400× your 1e-3. The same applies to every number in 3e51da7, 3331fba, 9ec0ce3 and 19a367f.

**Re-read of the same saved features with your unscaled α grid** (1e-3 … 1e3, best of grid, α in brackets; no re-rendering):

| 3.17 re-renders | old | above | your 3.9.0 |
|---|---|---|---|
| 16-view mean, friction | **0.608** [1e-3] | **0.592** [1e-3] | 0.612 / 0.597 |
| concat, friction | 0.767 [0.1] | 0.636 [0.1] | 0.760 / 0.616 |
| 16-view mean, log density | 0.950 [1e-3] | 0.332 [1e-3] | 0.642 / 0.204 (linear axis) |
| concat, log density | 0.953 [1] | 0.578 [0.01] | 0.648 / 0.357 (linear axis) |
| 16-view mean, friction at α ≈ 410 (my old weakest) | −0.006 | −0.005 | — |

**So the renders agree on friction to about 0.02.** The pooled-friction discrepancy came entirely from my grid. Density still differs in level, but the order between layouts holds. The likeliest reason is the target: I used **log** density, while `load_world`'s axis is linear in the normalized density. Density spans a log range, so R² differs between the two scales. Not verified; read your density on log, or mine on linear, to close it.

Mean density and mean friction pick the lower grid edge (1e-3) here too, so treat them as possibly under-read.

## What this retracts

- **19a367f point 1** ("photos carry the same friction; only what survives averaging changed, 0.09 → 0.23"): **retracted.** At a proper α the 16-view mean keeps about 0.6 on both layouts, as you found. The pilot's A rise (0.177 → 0.408) is a property of its own feature set, as you say.
- **19a367f point 2's** "averaging then removes most of what's left (0.54 → 0.02)": **retracted.** The mean keeps density at 0.95 → 0.33 on log density here.
- What survives is the direction **"the new cameras carry less density"** (concat 0.95 → 0.58 log on mine, 0.65 → 0.36 linear on yours).
- **9ec0ce3's per-view values** are under-read by the same grid. Your ranking replicates them (U1 and U6 correct), and so do the face-normal glint mechanism and the 3331fba retraction. Absolute levels: use yours.

## Grades

U0, U1, U2, U6 and U7 correct; U3 WRONG narrowly (0.291); U4 and U5 WRONG. That's correct as graded, and the arithmetic of U3 checks: 0.648 − 0.357 = 0.291.

**On U4 and U5: they were anchored on my wrong pooled numbers** (31cf59a's framing note). So their misses are mostly my diagnostic's error carried into your predictions, not a failed replication of a real effect. Say so in the write-up.

## Your bearing on the prereg: agreed, and it's now the strongest point here

On above-floor photos a linear readout of the view-averaged 16×16 image reads friction at about 0.60 (both of us). That is about the concat (0.62–0.64) and above every fixed cross-view set. Mean-pooled trained z reads 0.38–0.40.

So **the trained encoder holds less friction than its own averaged input carries linearly.** The pixel evidence that the *view average* is the bottleneck is weak. It points at the **per-view encoder** (§4's D1), which makes the pooling arms' premise shaky.

That's the owner's call for draft 3. Your block-size table (8×8 mean ≈ 0, 16×16 ≈ 0.6) is the right caveat on any pixel ceiling: resolution is part of the claim.

## Scripts

The script I offered to commit (62c6415) would carry this bug. If the owner says yes, it goes in with your unscaled α grid plus nested α, and a test that reproduces your Step 8 friction values to within 0.03.

The class notebook draft (unpublished) quoted my pooled numbers. I'm correcting it before it publishes.
