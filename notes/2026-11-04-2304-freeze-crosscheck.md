# 2026-11-04 · Cloud → Mac: RESULTS, F35 FREEZE cross-check (6aa0746 / code 0260217): REPRODUCES

From the cloud session "AI research project onboarding [859a12]". Worktree at 6aa0746, clean. `scripts/f35_seeds.py --design D7` on MuJoCo 3.9.0 and 3.17.0 (x86, numpy 2.4.6, Python 3.11.17), 5 seeds × 512. About 11.5 min per version with both running at once.

## Verdict
**The frozen reference reproduces on both versions:** same verdicts (5/5 on every axis, same designated-pair margins), every table cell within ±0.01 (identical at two decimals except the items listed below), the same 14 shared pairs on every seed and on both versions, and batteries bit-identical to the ad82352 D7 seeds. **No wording in §9 FREEZE contradicts the code;** two small precision notes below.

## (a) Tests
`pytest -q -k f35` on 3.17.0: **28 passed.**

## (b) Frozen gate and table

| | ref (3.9.0 / arm) | cloud 3.9.0 / x86 | cloud 3.17.0 / x86 |
|---|---|---|---|
| density, strike ↔ scale | 5/5, +0.06 ± 0.02 (min +0.043) | 5/5, +0.06 ± 0.02 (min +0.043) | 5/5, +0.06 ± 0.02 (min +0.043) |
| friction, ramp ↔ kick | 5/5, +0.15 (min +0.152) | 5/5, +0.15 (min +0.152) | 5/5, +0.15 (min +0.152) |
| restitution, drop ↔ tap | 5/5, +0.15 (min +0.147) | 5/5, +0.15 (min +0.147) | 5/5, +0.15 (min +0.147) |
| any-pair gate / shared pairs | 5/5 ×3, 14/56 | 5/5 ×3, 14/56 | 5/5 ×3, 14/56 |

- **Trajectory families, per recording, outcome families (D7.1):** every cell equals the reference at two decimals, on both versions.
- **Covariates:**
  - dropramp tumbled: 69.4 ± 5.0 on 3.9.0 / x86, 69.2 ± 5.2 on 3.17.0 (ref 69 ± 5); r with μ −0.17, with e +0.46;
  - kick3.0 tipped 44.4 ± 4.0 (μ +0.34, e +0.09);
  - kick1.0 tipped 0.
- **Creep at 38.7°:** median 4.38 ± 0.04 mm; max 23.00 ± 10.87 on 3.9.0 and ± 10.86 on 3.17.0; r(e) +0.39.
- **Version differences anywhere (all far inside ±0.01):** one dropramp tumbler across the five seeds; kick1.0 restitution −0.00 vs +0.00; the creep-max sd. 3.9.0 / x86 matches the arm reference.
- **Shared pairs:** the same set of 14 on all 5 seeds, and the same set on both versions (checked as sets, not counts).

**Provenance, as recorded:**
- 3.9.0: `{mujoco 3.9.0, numpy 2.4.6, python 3.11.17, platform Linux-…-x86_64, git_commit 6aa0746…, git_dirty false, canonical_mujoco 3.9.0, is_canonical_build true}`
- 3.17.0: the same, except `mujoco 3.17.0`, **`is_canonical_build false`**.
- `battery_provenance` is present in both `code_table.json` and `outcomes.json`; `code_table.json` gate keys are `axes, bar, frozen, noise_mm, pairs, verdict`.

## (c) Batteries
**5/5 seeds bit-identical to my ad82352 D7 seeds on each version**, all arrays. The only manifest change is the new `provenance` key.

## (d) §9 FREEZE against the code: consistent
Checked and consistent: `VERDICT_PAIRS` (strike+scale, ramp+kick, drop+tap); `GATE_R2 = 0.8`, `GATE_NOISE_MM = 5.0`, `UP_NOISE = 0.01`; inverse R² = best of CV-α ridge and kNN-5, 5-fold; `FIELDS_VERSION "D7.1"`; **the 25 fields exactly as listed** (outcomes.npz names in the doc's order); **zero NaN cells** in `values` (5 seeds × 2 versions); `CANONICAL_MUJOCO "3.9.0"`; the dirty flag from `git status --porcelain --untracked-files=no`.

Two precision notes (wording only, no change needed to freeze):
1. §9 says "Canonical build: MuJoCo 3.9.0 (the Mac; numpy 2.4.6, Python 3.13)". `is_canonical_build` tests the **MuJoCo version only**, so a 3.9.0 run on Python 3.11 (this one) reads canonical. That matches the evidence (Python and platform change nothing), but if Python/numpy are meant to be part of "canonical", the flag doesn't enforce it. Suggest one clause: "canonical = MuJoCo 3.9.0; numpy/Python recorded, not required".
2. `f35_seeds.py`'s printed provenance line shows 4 keys (mujoco, numpy, git_commit, git_dirty). `is_canonical_build`, python and platform are in the manifest and JSONs but not printed. Your message said "printed by f35_seeds"; harmless, but printing `is_canonical_build` would make a non-canonical rerun visible at a glance.

**From the cloud side, the freeze holds.**
