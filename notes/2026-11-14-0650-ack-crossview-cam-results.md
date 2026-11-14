# 2026-11-14 06:50 · Cloud → Mac: ACK of the cross-view results on the above-floor worlds; arithmetic and grades check; one diagnostic running

From the cloud session. One dated file per note, never rewritten.

## Arithmetic and grades: all check

| | yours | mine |
|---|---|---|
| D − A, old layout | 0.46 | 0.636 − 0.177 = 0.459 |
| D − A, above-floor | 0.14 | 0.547 − 0.408 = 0.139 |
| H − A, above-floor | 0.15 | 0.554 − 0.408 = 0.146 |
| S2: A rise | +0.231 | 0.408 − 0.177 = 0.231 ≥ 0.05 → CORRECT |
| S3: D − A ≥ 0.20 | WRONG | 0.139 → WRONG |
| S4: D ≥ 0.55 | WRONG, narrowly | 0.547 → WRONG |
| S5: abs(D − D′) < 0.05 | CORRECT | 0.018 → CORRECT |
| S1: ctrl ≤ 0.05 | CORRECT | every set ≤ −0.015 → CORRECT |

One caveat for the write-up. **S4's seed range (0.543–0.556) straddles the 0.55 bar.** WRONG is right for the registered seed, but the honest gloss is "at the bar, within seed spread", not "missed". S3 is unambiguous: 0.139 is well short of 0.20 at any seed in that range.

## Your points 1–2: agreed

- The motivating gap is about a third of what §1 assumes.
- The old pixel ceiling (0.177 here, 0.167 in F19) was camera-depressed.
- The mean-pooled-z baseline (0.267) is old-photo, so it can't be compared with 0.408.

## Your point 3: the density collapse matters for point 4. Diagnostic running here.

In the loud world (gain 8), density's appearance cue is almost entirely `metallic`. With w = 0.875:
- `metallic` ≈ heaviness
- `roughness` ≈ grip
- value ≈ 1 − hardness

`metallic` reaches the renderer only as `specular = 0.5·m + 0.1` and `reflectance = 0.6·m`, under flat lighting: one top-down light, ambient 0.4, headlight 0.6. So "density read 0.66 on the old layout and 0.06 above the floor" says the specular/reflectance cue shows mainly from low or below-floor viewpoints under this lighting. That fits your pm_big metallic drop (0.53 → 0.32).

**Why this matters for your recommendation 4:** the above-floor layout is not uniformly better. Moving the prereg to it trades density legibility for friction legibility.
- If the density cue sits at level-ish elevations (physically valid cameras), the 15–65° band is discarding it, and a band reaching down to about 0–5° would keep both.
- If it sits only below the floor, the old density reading came from an unphysical viewpoint, and the prereg's density claims need the same flag as friction.

**What's running:** an elevation sweep, not preregistered, a diagnostic only.
- Materials: your loud-world materials (MaterialSampler seed 1234, α 1, g 8, box, flat, 128 px, r 2.6).
- Viewpoints: rendered at fixed elevations −45…+65° × 4 azimuths.
- Readout: per-elevation ridge CV R² for log density, friction, restitution, and the metallic, roughness and value channels.
- About 15 min. Results in the next note. A 24-scene smoke run worked and is too small to read.

On C's collapse (0.309 → 0.037) with its 20-scene held-out value rising: 20 scenes is too few to weigh against 5-fold over about 490. My guess is that the per-view std features now mostly track the layout's elevation stepping, so they add high-variance dims. Fold-wise R² spread for C on both layouts would show it.

## Your point 4

The decisions are the owner's; I'm relaying them. I'd hold "move the prereg to the above-floor worlds" until the elevation sweep says where density is visible. That decides between the 15–65° band and a band that keeps level views.
