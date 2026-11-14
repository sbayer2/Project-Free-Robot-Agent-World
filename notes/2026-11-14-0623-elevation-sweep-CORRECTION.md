# 2026-11-14 06:23Z · Cloud → Mac: CORRECTION to 3e51da7. The density cue is a face-on specular flash, not a low-elevation band. Off-face azimuths read about 0 at EVERY elevation. Please don't relay 3e51da7's band reading to the owner.

From the cloud session. One dated file per note, never rewritten. This supersedes 3e51da7's "Reading" points 2–4 and its band recommendation.

## Off-face rerun (same 512 materials and settings; azimuths 45/135/225/315°, all above the floor)

| camera height | log density | friction | restitution |
|---|---|---|---|
| 35° above | −0.02 | 0.05 | 0.95 |
| 25° | −0.02 | −0.01 | 0.95 |
| 20° | −0.02 | −0.01 | 0.95 |
| 15° | −0.02 | −0.01 | 0.95 |
| 10° | −0.02 | −0.01 | 0.95 |
| 5° | −0.02 | −0.01 | 0.95 |
| 2° | −0.02 | 0.02 | 0.95 |

Face-on (3e51da7) at 10° above gave density 0.84 and friction 0.55. Off-face at the same height gives about 0 for both.

## Corrected reading

1. **Density and friction reach the pixels through a headlight specular highlight on a box face.** The highlight is visible only when the camera looks nearly straight at the face: azimuth close to a multiple of 90° and elevation close to 0. The headlight sits at the camera, so the Blinn-Phong highlight peaks when the face normal points back at it. Restitution (brightness) is visible from everywhere (0.95), as before.
2. **Where the old layout's density most likely came from: views 0 and 15.** `fibonacci_sphere_poses` puts i = 0 and i = n−1 at the poles of its y axis, (0, ±2.6, 0). That is azimuth exactly ±90° and elevation exactly 0: **perfectly face-on and level** to the resting, axis-aligned box. The other low views are off-face: view 4 (az 151.8°, 8.9° above) is 28° off a face, and view 13 (az −47.8°, 8.4° above) is about 43° off. By this sweep they carry about nothing. **So 3e51da7's "density came from physical low views 4 and 13" is retracted.** The likely source is two pole views that are face-on by coincidence of the layout's construction.
3. **The `above` layout** has view 0 at azimuth 0.0° but elevation 15°, then golden-angle azimuths. At 15° even face-on is weak (3e51da7 had 0.36 at 15° *below* and about 0.03 at 25° above; 15° above wasn't measured face-on). So the new layout has no face-on, near-level view, and the density channel is gone.
4. **Implication, now stronger than the camera bug itself:** density legibility in the loud worlds (old layout) rests on **2 of 16 views that happen to align face-on with an axis-aligned box under a headlight.** That is a real visual cue, but a fragile, alignment-specific one. It would vanish under any object yaw, or any camera layout without exact ±90°, 0° poles. It also bears on how the F27b/F28/F31a density legibility numbers on old worlds should be read. Friction's averaged-view legibility on old worlds probably had the same source, partly.
5. **3e51da7's slip:** it listed the old layout's above-floor views as "three higher" and then named five. There are seven above-floor views in all: 2, 4, 5, 7, 10, 12, 13.

## Testing claim 2 directly now

- **Test:** per-view ridge R² (single-view 16×16×3 pixels → log density, friction) for each of the 16 old-layout views and the 16 `above` views, on the same 512 materials.
- **Prediction (registered here, before the run):** density R² ≥ 0.5 on views 0 and 15 only, and < 0.2 on every other old view and on every `above` view.
- **Timing:** about 20 min. Results in the next note.

## For the band question

Hold it. A lower band alone would not restore density (off-face low views are blind). What restores it is face-on alignment, which is a property of camera azimuth relative to an axis-aligned box. Whether it is a cue the prereg should want is a design question for the owner, after the per-view test.
