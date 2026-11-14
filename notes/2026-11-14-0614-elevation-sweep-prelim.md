# 2026-11-14 06:14Z · Cloud → Mac: elevation sweep (preliminary). Density lives in a narrow low band, about 5–10° above the floor, which the 15–65° layout skips. The old density came mostly from PHYSICAL low views.

From the cloud session. One dated file per note, never rewritten. This is a diagnostic, **not preregistered**. A robustness rerun at off-face azimuths is running, and its results come in the next note.

## Setup

- **Materials:** the same as `pm_f31fix_loud`: `MaterialSampler(seed=1234, coupling_alpha=1, coupling_gain=8)`, first 512 draws, box.
- **Rendering:** `build_mjcf(..., lighting="flat")`, 128 px, distance 2.6, lookat at half-height. MuJoCo 3.17.0 with OSMesa.
- **Views:** each scene is rendered at 9 fixed elevations × 4 azimuths (0/90/180/270°, i.e. **face-on to the axis-aligned box**, the caveat being checked).
- **Features:** per elevation, the mean over the 4 azimuths of 16×16×3 block-mean pixels.
- **Readout:** ridge, 5-fold CV R². λ is the best of 4, a mild optimism shared by every row.
- **Script:** `scratchpad f35w/elev_sweep.py` (the cloud scratchpad, not committed). I can commit it if you or the owner want it on a branch.

## Results (elevation stated physically: + above the floor, − below)

| camera height | log density | friction | restitution | metallic | roughness |
|---|---|---|---|---|---|
| 45° above | 0.01 | 0.26 | 0.95 | 0.01 | 0.27 |
| 25° above | 0.03 | 0.35 | 0.94 | 0.03 | 0.36 |
| **10° above** | **0.84** | 0.55 | 0.95 | 0.85 | 0.56 |
| 2° above | 0.33 | 0.47 | 0.92 | 0.32 | 0.48 |
| 5° below | **0.92** | **0.78** | 0.95 | 0.93 | 0.79 |
| 15° below | 0.36 | 0.61 | 0.95 | 0.37 | 0.63 |
| 30° below | 0.00 | 0.21 | 0.95 | 0.00 | 0.22 |
| 45° below | −0.02 | 0.02 | 0.95 | −0.02 | 0.03 |
| 65° below | 0.03 | 0.40 | 0.95 | 0.03 | 0.41 |

## Reading

1. **The density R² is the metallic R² (to ±0.01 in every row).** As expected at gain 8: metallic ≈ heaviness, and density reaches the pixels only through `specular = 0.5·m + 0.1` and `reflectance = 0.6·m`. Likewise friction tracks roughness (via shininess), and restitution tracks value (brightness), which is flat at about 0.95 at every angle.
2. **The specular cues show only near grazing.** Density is about 0 from 25° above and higher, and peaks within about 10° of the horizon. Friction is also best low (0.55 at 10° above, against 0.26–0.35 higher), but unlike density it keeps some signal at every angle.
3. **Where the old layout's density came from.** The old 16 views, in physical elevations (MuJoCo's negative = above):
   - two views at about 9° above (view 4: 8.9°, view 13: 8.4°);
   - two level (0, 15);
   - three higher above (23.6°, 30.4°, 42.6°, and two near 60°);
   - seven below (20–71°).
   The 9°-above pair sits right on the 10°-above density peak (0.84). Those are **physical cameras**. So the old worlds' density legibility did not depend on the bug's below-floor views. It came mainly from the two low physical views (plus the level pair). The `above` layout's lowest view is 15°, the next 18.3°, then 21.7° and up. By 25° density is gone, so **the new band cuts the density channel off.**
4. **Hypothesis for the mechanism (not measured):** under flat lighting the only real light points straight down. The box's vertical faces get no direct light and show the metallic difference (specular from the headlight, plus reflectance) only at grazing views. From above, the brightly lit top face dominates the frame and washes the difference out.
5. **Non-monotonic rows (2° above 0.33 between 0.84 and 0.92) are suspicious.** With face-on azimuths, the camera at about 0° sees almost only one vertical face edge-on to the floor line. Hence the off-face rerun.

## What this means for the band question (owner's call; relay as you see fit)

- The 15–65° band makes friction more legible on *averaged* views (A 0.18 → 0.41) at the cost of nearly all density legibility (0.66 → 0.06).
- A physically valid layout that keeps a few views at about 5–12° above the floor would plausibly keep both. That's a prediction, to be tested by rendering such a layout, not assumed.
- If the prereg freezes on the 15–65° worlds, its density readouts measure a channel the cameras barely see. That's fine for a friction-primary prereg, but say so in §4 (your point 9 there becomes "density is camera-excluded", not merely "changed").
- The F27b/F31a/F28 density numbers on old worlds stand as physical-view numbers. The camera bug does not void them, at least for density.

**Pending:** the off-face azimuth rerun (45/135/225/315°) at 7 elevations, all above the floor (35, 25, 20, 15, 10, 5 and 2°). About 15 min.
