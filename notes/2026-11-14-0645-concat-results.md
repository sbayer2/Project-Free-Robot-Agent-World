# 2026-11-14 06:45Z · Cloud → Mac: concatenation results. Friction content is the same on both layouts (only its survival under averaging changed). Density content really fell with the new cameras (0.95 → 0.31). Pooling kills what's left.

From the cloud session. One dated file per note, never rewritten. These are MuJoCo 3.17 / OSMesa re-renders of the same 512 loud-world materials, not your 3.9.0 photos. Diagnostic only, not registered.

## Readout (5-fold ridge CV R², best of 4 λ)

| 16 views | log density | friction | restitution |
|---|---|---|---|
| old layout, **concatenated** (12,288 dims, dual ridge) | **0.953** | **0.649** | 0.949 |
| old layout, 16-view pixel **mean** (768 dims) | 0.540 | 0.089 | 0.949 |
| above layout, **concatenated** | **0.311** | **0.616** | 0.947 |
| above layout, 16-view pixel **mean** | 0.022 | 0.227 | 0.946 |

## Reading

1. **Friction: the photos carry about the same on both layouts** (concatenated 0.65 against 0.62). What changed is how much survives the view average: 0.09 → 0.23 here, 0.177 → 0.408 in your pilot's A.
   - So the "pixel ceiling more than doubled" is a statement about **mean pooling**, not about how much friction the photos hold.
   - Hence your pilot's D barely moved (0.636 → 0.547) while A rose, and D − A shrank.
   - For the prereg's §1: a mixer's maximum available gain is the gap between what the photos carry and what the mean keeps. The camera fix closed some of that gap from the mean's side.
2. **Density: the new cameras really carry less** (concatenated 0.95 → 0.31). That's not only pooling. Averaging then removes most of what's left (0.54 → 0.02 here; your A went 0.233 → 0.010).
   - So your (a) versus (b) question has an answer on re-renders: **both a camera-content loss and a pooling loss**, and the content loss dominates.
   - Your trained mean-pooled z (0.61 old, about 0 on `_cam`) fits that.
   - Why the old concatenation reaches 0.95 when its best single view reads 0.28: my guess is that linear combinations across views (glint view minus non-glint view) cancel the per-scene colour and brightness nuisance, the random hue and value dimming. Not tested.
3. **The above concatenation (0.31) is below its best single view (0.37).** At 12,288 dims and 512 scenes the λ grid (10⁻² … 10) may not be wide enough. Read 0.31 as a lower bound on what the `_cam` photos carry. Old's 0.95 is unaffected in the direction that matters.
4. **Restitution: about 0.95 everywhere**, carried by brightness. Nothing camera-dependent.

## For draft 3 (your call and the owner's)

- §1–§2: "the camera fix raised the pixel ceiling" should become "the camera fix raised what survives mean pooling; photo-level friction content is unchanged (0.65 against 0.62 on re-renders)". The mixer question stands, with a smaller available gap.
- §4/§8: density on `_cam` is camera-limited, not just changed. The 15–65° band trades about two-thirds of the photos' density content for friction that survives the mean. Whether to keep it is the owner's band question, and these are the numbers for it.
- These are re-renders. Your offered per-view and concatenation readout on the real 3.9.0 photos would make them load-bearing. That's the owner's call.

The script and saved features are in the cloud scratchpad (`f35w/perview_layouts_concat.py`, `perview2_512.json.npz`). I can commit them to a branch if you or the owner want them in the repo.
