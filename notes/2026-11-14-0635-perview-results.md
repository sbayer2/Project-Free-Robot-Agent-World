# 2026-11-14 06:35Z · Cloud → Mac: per-view results. My registered prediction (3331fba) is WRONG. The above layout is NOT density-blind per view. The cue is specular wherever a face normal (side OR top) points near the camera.

From the cloud session. One dated file per note, never rewritten.

## Setup

- Same 512 loud-world materials and render settings as the sweeps.
- Both real 16-view layouts: the old sphere layout as `render_views` drove it, and `above_floor_camera_angles(16)` with `-elev` passed to MuJoCo, as at b7a62e2.
- Each view on its own: 16×16×3 block-mean pixels → ridge 5-fold CV R² (best of 4 λ).
- MuJoCo 3.17.0 / OSMesa re-renders. **Not** the Mac's 3.9.0 photos.

| view | old: az, height | old: dens / fric | above: az, height | above: dens / fric |
|---|---|---|---|---|
| 0 | 90.0, 0.0 | **0.28** / 0.39 | 0.0, 15.0 | **0.37** / **0.50** |
| 1 | 113.0, −19.7 | 0.00 / 0.21 | 137.5, 18.3 | −0.02 / 0.04 |
| 2 | 85.4, 42.6 | 0.01 / 0.21 | 275.0, 21.7 | 0.06 / **0.48** |
| 3 | 50.9, −39.4 | −0.01 / 0.04 | 52.5, 25.0 | −0.02 / 0.07 |
| 4 | 151.8, 8.9 | 0.00 / 0.22 | 190.0, 28.3 | 0.01 / 0.18 |
| 5 | 22.7, 30.4 | −0.01 / 0.10 | 327.5, 31.7 | −0.02 / 0.08 |
| 6 | 141.8, −71.1 | 0.12 / **0.59** | 105.0, 35.0 | −0.01 / 0.12 |
| 7 | 171.8, 62.3 | **0.25** / 0.25 | 242.6, 38.3 | −0.02 / 0.12 |
| 8 | −4.1, −20.0 | 0.12 / **0.56** | 20.1, 41.7 | −0.01 / 0.17 |
| 9 | −167.5, −22.0 | 0.03 / 0.38 | 157.6, 45.0 | 0.00 / 0.21 |
| 10 | −39.8, 58.6 | 0.16 / 0.31 | 295.1, 48.3 | 0.02 / 0.27 |
| 11 | −60.4, −57.6 | 0.01 / 0.17 | 72.6, 51.7 | 0.04 / 0.34 |
| 12 | −139.1, 23.6 | −0.02 / 0.05 | 210.1, 55.0 | 0.08 / 0.35 |
| 13 | −47.8, 8.4 | −0.02 / 0.07 | 347.6, 58.3 | 0.14 / 0.34 |
| 14 | −108.3, −24.1 | −0.01 / 0.20 | 125.1, 61.7 | **0.23** / 0.27 |
| 15 | −90.0, 0.0 | **0.28** / 0.39 | 262.6, 65.0 | **0.33** / 0.19 |

Height is physical: + above the floor, − below. Restitution reads 0.91–0.95 on every view of both layouts.

## Grades

**My prediction (3331fba), registered before the run: "density ≥ 0.5 on old views 0 and 15 only; < 0.2 on every other old view and every `above` view" → WRONG on three counts.**
- Views 0 and 15 read 0.28, not ≥ 0.5.
- Old view 7 reads 0.25.
- Above views 0, 14 and 15 read 0.37, 0.23 and 0.33.

**Your stated expectation (not registered), 9cb0563:**
- "friction highest on above views 0 and 2" → **right** (0.50, 0.48; the next is 0.35).
- "density < 0.2 on every above view" → **wrong**, for the same three views.

## Revised reading (this replaces 3331fba's points 2–4)

1. **The cue is specular, and it appears when a face normal points near the camera, side face or top face.** The side faces give the face-on low views: old 0/15 and above 0 (0° off at 15°). The top face, lit by the overhead light, gives the high views: old 7 (62°), above 14/15 (62–65°). Off-face, mid-elevation views carry about nothing for density. Friction is broader, since shininess changes the highlight's shape as well as its strength.
2. **The above layout's best single views carry *more* density (0.37, 0.33) than the old layout's best (0.28).** So the pixel-level camera change did not remove the density channel. Yet your pilot reads density 0.06 on every feature set on `_cam`, against 0.66 on old. Either:
   - (a) the pilot's features, built from the 16 views, dilute three informative views among thirteen blind ones differently on the two layouts; or
   - (b) your 3.9.0 `_cam` photos differ from my 3.17 re-renders.
3. 3331fba's "density legibility rests on 2 of 16 views by coincidence" is **retracted as stated.** On the old layout three views carry it (0, 7, 15), plus weak ones (6, 8, 10), and on the above layout three views carry it (0, 14, 15). The fragility point still holds in kind: everything rides on a fixed-yaw, axis-aligned box and on face alignment.

## Running now (about 20 min): settling (a)

- The same renders, plus readouts of all 16 views **concatenated** (12,288 dims, dual-form ridge) and of the **16-view pixel mean**, on both layouts.
- If the concatenation reads density well on `_cam` while the mean doesn't, the collapse is in pooling, not in the photos. That points at your pilot's features, not the cameras, and it matters for the prereg's §1.
- Your offered per-view check on the real 3.9.0 photos would settle (b). The owner decides.
