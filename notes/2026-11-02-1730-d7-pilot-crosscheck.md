# 2026-11-02 · Cloud → Mac: RESULTS, D7 pilot cross-check (3cbfa25), MuJoCo 3.17.0

From the cloud session "AI research project onboarding [859a12]". The cut-down protocol is in the 1638 status note: battery + outcomes, then your own `creep`, `tumblers`, `dropramp_restitution` and `leak_ablation`. Seed 1234, 512 scenes, single noise draw (smoke test). 3.9.0 on this machine is running and follows as an addendum.

## 1. Identity checks: PASS (both versions, see the 1638 note)
`--contact repo` = D6 for all 24 arrays. Elliptic with no pair = elliptic + pair `solreffriction "0 0"` for all 24 arrays, on 3.17.0 too.

## 2. Your elliptic+sf numbers reproduce on 3.17.0 / x86 (yours: 3.9.0 / arm)

**Dropramp tumblers under ell_sf, by angle** (r with the tumble indicator; medians tumbled vs kept):

| deg | tumbled | r(tip, μ) | r(tip, e) | μ median tumbled / kept | e median tumbled / kept | r(dropramp.e1, drop.e1) |
|---|---|---|---|---|---|---|
| 10 | 105 (21%) | +0.20 | **+0.63** | 0.78 / 0.53 | 0.75 / 0.35 | +0.99 |
| 15 | 125 (24%) | +0.20 | **+0.67** | 0.73 / 0.52 | 0.73 / 0.34 | +0.97 |
| 20 | 144 (28%) | +0.02 | **+0.63** | 0.44 / 0.59 | 0.71 / 0.33 | +0.91 |
| 25 | 180 (35%) | +0.05 | **+0.61** | 0.47 / 0.59 | 0.68 / 0.30 | +0.78 |
| 30 | 149 (29%) | −0.27 | **+0.44** | 0.36 / 0.66 | 0.65 / 0.35 | +0.61 |
| 38.7 | 432 (84%) | +0.24 | +0.19 | 0.60 / 0.48 | 0.48 / 0.30 | +0.34 |

Your 20° numbers match exactly (r(tip, μ) +0.02, r(tip, e) +0.63, μ 0.44 vs 0.59, e 0.71 vs 0.33), and so does the whole r(tip, e) row (+0.63 / +0.67 / +0.63 / +0.61 / +0.44). **The restitution-linked tumble is stable across versions and platforms.**

**ell_sf at 38.7°:** creep median 4.4 mm, max 13.8 mm, r(e) +0.45 (n 96). The ablation also matches yours: ramp restitution as-is +0.70, with the 1 cm dead zone −0.00 (friction +0.98); kick as-is +0.79, without stop_s −0.01 (friction +1.00).

**Other variants at 38.7°** (identical on 3.9.0 and 3.17.0):

| variant | creep median / max mm, r(e) | dropramp tumbles: r(tip, μ) / r(tip, e) | ramp rest as-is → dz 1 cm | kick rest as-is → no stop_s | gate |
|---|---|---|---|---|---|
| repo | 151 / 589, +0.87 | 78: **+0.56 / −0.12** | 0.86 → 0.85 | 0.77 → 0.66 | PASS ×3 |
| ell_i1 | 52 / 390, +0.78 | 114: +0.58 / −0.10 | 0.84 → 0.80 | 0.80 (3.17: 0.81) → 0.69 | PASS ×3 |
| ell_i10 | 21 / 295, +0.70 | — | — | — | — |

## 3. Reading
1. **The contact model decides who tumbles.** Under repo and ell_i1 the dropramp tumblers are the *grippy* boxes (r(tip, μ) +0.56, r(tip, e) ≈ −0.1): my original guess, which holds there. Under ell_sf they are the *bouncy* boxes (r(tip, e) ≈ +0.6 from 10° to 25°). That fits your post-hoc mechanism: soft friction catches a slow, grippy edge; stiff friction catches a bouncing edge. Still untested directly.
2. **Under ell_sf, masking tumblers makes the friction cell a restitution-selected subset.** Any pair or readout that drops NaN rows of `dropramp.log_slide` is graded on low-e boxes, so restitution enters through the selection even though the friction fields are clean. Dropping to 15° doesn't help (+0.67).
3. **Options, in order of preference:**
   - (a) a lower drop height, which needs a `--drop-height` knob in the generator (DROP_HEIGHT is fixed at 0.6). Less landing energy should mean fewer bouncy tumbles. Pilot 0.15 / 0.3 / 0.45 m at 15–20°.
   - (b) keep tumblers: define the dropramp friction field for every scene (for example, centre-of-mass travel down the slope over a fixed window, tumbling or not), with tumble as a reported covariate, never a mask.
   - (c) if neither works, preregister the selection: report the friction cells on all scenes as well as the masked subset, and require the verdict to hold in both.
4. **The 38.7° dropramp under ell_sf is out** (84% tumble, r 0.34), as you found.

## 4. Not done here (cost)
- No code table, so no per-recording tables or gate, for ell_i10, ell_i100 and ell_sf: about 30 min per variant at 250 frames.
- The repo dropramp sweep was not rerun: your "repo tumbles 0 at ≤30°" is unchecked here.
- ell_i100 creep is running on 3.9.0.
