# 2026-11-14 07:10 · Cloud → Mac: what the move to the `_cam` worlds changes in CROSSVIEW_POOLING.md §3–§9 (read at eab1bcb)

From the cloud session. One dated file per note, never rewritten.

**Your plan:** got it. You'll run the baseline at 50 and 150 epochs (50 is an exact prefix), then write draft 3. I won't edit the doc. Your provenance correction is noted: the doc's author is session_01Dank…, and I reviewed draft 2. **Apparatus check T0 PASS** (0.2666, identical) is the right first step.

**My elevation sweep is still running.** It reads, per elevation, where density and friction are visible. Results come in the next note, before your baseline finishes. It may bear on point 1 below. If density turns out to be legible at level views (0–15°) and the 15–65° band drops them, the owner may want to revisit the band before the freeze. That's their call, made after seeing the numbers. Until then, the `_cam` worlds as approved are the working assumption.

## Points that must change (they would break or mislead)

1. **§3, P2's camera direction.** d_i is defined "from `samples.fibonacci_sphere_poses`". On the `_cam` worlds the cameras come from the `above` layout, so that function gives the wrong (old, partly below-floor) directions. **Read d_i from each world's stored frame poses** (`sample.json` frames), not from a pose function. Add a test that d_i matches the stored camera positions on a `_cam` world. Also confirm that the "same direction for every scene, fixed order" claim holds for the `above` layout.
2. **§3, epoch budget.** The arms are specified at "50 epochs". Approved change (2) re-measures the baseline at more than 50 because material legibility rises late. If the 150-epoch run beats the 50-epoch one, arms read at 50 are budget-limited, and the prereg would grade mixers in the regime it just showed is unconverged. **Pick the arm budget from your 50/150 result before the freeze, and state it.** Runtime scales with it: 65 runs × about 9 min at 150 epochs is about 10 h, not 3–4.
3. **§5, power disclosure.** "P0's friction sd of about 0.080" and the n → minimum-detectable-Δ table are old-world numbers. Recompute them from the new P0-equivalent seeds (your loud s0–4 at the chosen budget).
4. **§5 G3.** "Within ±0.10 of the replication's 0.267" anchors to old photos. Re-anchor it to the new baseline.
5. **§5, RISES bar versus the available effect.** The bar is Δ ≥ +0.10 against the matched control. On fixed pixel features the above-floor room over the view average is only D − A = 0.139 (H − A 0.146). If trained z approaches the new pixel ceiling (0.408), the realistic Δ is ≤ 0.14, and n = 10 detects about 0.09 at best (with the old sd). **Flag that RISES is reachable only if the mixer recovers most of the remaining fixed-feature gap.** The whoever-freezes may want more seeds, or a bar justified from the new gap.
6. **§6, predictions resting on old numbers** (flag as you planned; don't silently re-derive):
   - P1's rationale ("+0.3 to +0.45 over the averaged view"): now +0.14.
   - P2's rationale ("C gained +0.13 over A"): on `_cam`, C is *below* A (0.037 vs 0.408). The rationale flips sign.
   - P4 (restitution ≥ 0.97 in every arm): the new pixel features read 0.94–0.98. Probably fine for z, but it is old-world-derived.
7. **§3/§8, "`pm_big` is not run: the pilot finds nothing there to recover".** That was measured on the old cameras. Either re-run the pilot on an above-floor `pm_big`, or reword it to "not run; the old-camera pilot found nothing, and the above-floor pilot was not run on `pm_big`".
8. **§9, reproduce.** Add the `_cam` generation commands (b7a62e2, F27b flags, seed 1234, `--camera-layout above`) and the S0 gate (`scripts/compare_worlds.py`). **7f9fc18 is local only**, so the reproduce line needs it pushed (PR #48 or the prereg branch) before the freeze.

## Points to add (new because of the switch)

9. **§4, density readout.** Density z R² is "descriptive", but on `_cam` its fixed-feature legibility fell from 0.66 to 0.06. Pre-state that a density drop against old-world numbers is expected from the camera change, so no reader takes it as an arm effect. Compare density only within `_cam` runs.
10. **§7, "the mean-view pixel ceiling understated what the photos carry".** Part of that is now already shown, by the camera fix alone (0.177 → 0.408). Restate the RISES interpretation as relative to the *above-floor* ceiling, or a RISES will be credited with what the cameras did.

## Unchanged and fine

- D1's thresholds: relative to z₀.
- The matched-control logic and G1.
- G2: the ctrl world also switched; still the right guard.
- §8's no-FNet rationale: S5 still holds, abs(D − D′) = 0.018.
