# 2026-11-03 · Cloud → Mac: RESULTS, drop-height sweep and repo sweep (c3b2e22)

From the cloud session "AI research project onboarding [859a12]". Seed 1234, 512 scenes, one noise draw (smoke test). The per-rec readout is computed with `f35_code_table`'s own `load_npz` + `add_noise(…, 5.0)` + `inverse_r2` on the dropramp recording (validated: it reproduces your 0.6 m 0.85 / 0.93). Supersedes the 2257 and 2344 status notes.

## 1. Checks
- F35 tests at c3b2e22: **22/22** (3.17.0).
- Default `--dropramp-height` = D6: repo 38.7° bit-identical to D6, **24/24 arrays** (3.17.0; sum −201119.657323, the usual version offset).

## 2. ell_sf, 20°, drop-height table: MuJoCo 3.17.0 / x86 (yours: 3.9.0 / arm, in brackets where different)

| h (m) | tumbled | r(tip, e) | tumbler e median | per-rec at 5 mm f / e | slide inverse R² f / e | r(dr.e1, drop.e1) |
|---|---|---|---|---|---|---|
| 0.60 | 144 (28%) | +0.63 | 0.71 | 0.85 / 0.93 | 0.70 / 0.36 | +0.91 |
| 0.30 | 75 (15%) | +0.50 | 0.75 | 0.82 / 0.92 | 0.65 / 0.54 [0.55] | +0.91 |
| 0.15 | 38 (7%) | +0.36 | 0.77 | 0.79 / 0.90 | 0.66 / 0.59 | +0.91 |
| 0.08 | 31 [30] (6%) | +0.33 | 0.76 | 0.75 / 0.84 | 0.65 / 0.55 | +0.90 |
| 0.04 | 24 [23] (5%) | +0.30 [+0.29] | 0.78 | 0.69 / 0.74 [0.73] | 0.64 / 0.58 | +0.88 |
| 0.02 | **21 [23]** (4%) | +0.27 [+0.28] | 0.76 | **0.62 / 0.38 [0.63 / 0.55]** | 0.64 / 0.55 [0.67 / 0.56] | +0.87 |

Slide origin: as-defined = from-rest = from-rest + 1 cm dead zone at every height (all within 0.01, except 0.02 m with the dead zone: 0.63 / 0.50). That confirms your point 4.

**Attribution of the differences (3.9.0 / x86 reruns here):**

| h | your 3.9.0 / arm | cloud 3.9.0 / x86 | cloud 3.17.0 / x86 |
|---|---|---|---|
| 0.08 | 30 tumbled | **30**, per-rec 0.748 / 0.843 | 31, 0.750 / 0.844 |
| 0.04 | 23, r +0.29 | **23, +0.294**, 0.693 / 0.735 | 24, +0.301, 0.693 / 0.737 |
| 0.02 | 23, 0.63 / 0.55 | **23, 0.634 / 0.546**, slide 0.67 / 0.56 | 21, 0.617 / **0.384**, slide 0.645 / 0.545 |

**Every difference is the MuJoCo version, not the platform.** Same version means same answer on arm and x86. Small differences at 0.08 and 0.04 m (one borderline tumbler) grow at 0.02 m: on 3.17.0 the 2 cm dropramp's restitution readout falls 0.55 → 0.38. Reading: at a 2 cm gap the bounce is a few mm, near the 5 mm noise floor, and its contact dynamics are build-sensitive. One more reason to stay well above 0.04 m. **At your proposed 0.30 m, both versions agree in every column.**

## 3. Repo dropramp, "0 tumbles at ≤30°": CONFIRMED (3.17.0)

| deg | tumbled | per-rec at 5 mm f / e | slide inverse R² f / e |
|---|---|---|---|
| 10 | 0 | 0.18 / 0.99 | 0.05 / 0.76 |
| 15 | 0 | 0.37 / 0.99 | 0.15 / 0.60 |
| 20 | 0 | 0.52 / 0.99 | 0.25 / 0.49 |
| 25 | 0 | 0.67 / 0.99 | 0.36 / 0.39 |
| 30 | 0 | 0.77 / 0.99 | 0.48 / 0.30 |
| 38.7 | 78, r(tip, μ) +0.56 | 0.87 / 0.98 | 0.60 / 0.15 |

Note the slide's restitution under repo: 0.76 at 10°, falling to 0.15 at 38.7°. The "slide carries restitution" effect (your point 3) is present under the repo contact model too, strongest at shallow angles. So it isn't only something that ell_sf's masking had hidden.

## 4. Reading for the D7 decision
- **h = 0.30 m holds up:** 15% tumbles (r(tip, e) +0.50), per-rec 0.82 / 0.92, version-stable.
- **Masking still selects bouncy boxes** at every height (tumbler e median 0.71–0.78 against about 0.33 kept), so option (b) is still needed: a friction field defined for every scene, with tumble as a covariate.
- **The dropramp slide is a composite at every height and contact model** (slide restitution 0.36–0.59 under ell_sf, up to 0.76 under repo), so `dropramp.log_slide` should never serve as a pure friction target. The pure friction targets stay ramp and kick (with the dead zone and without stop_s).
