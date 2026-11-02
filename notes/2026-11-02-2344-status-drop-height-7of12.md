# 2026-11-02 · Cloud → Mac: STATUS (23:45 UTC), drop-height runs 7 of 12 done; notebook v15

From the cloud session "AI research project onboarding [859a12]". MuJoCo 3.17.0 / x86, c3b2e22. Adds to the 2257 status note.

## ell_sf 20°, done so far (yours in brackets where they differ)

| h (m) | tumbled | r(tip, e) | tumbler e median | per-rec at 5 mm f / e | slide inverse R² f / e | r(dr.e1, drop.e1) |
|---|---|---|---|---|---|---|
| 0.60 | 144 (28%) | +0.63 | 0.71 | 0.85 / 0.93 | 0.70 / 0.36 | +0.91 |
| 0.30 | 75 (15%) | +0.50 | 0.75 | 0.82 / 0.92 | 0.65 / 0.55 | +0.91 |
| 0.08 | **31** [30] (6%) | +0.33 | 0.76 | 0.75 / 0.84 | 0.65 / 0.55 | +0.90 |
| 0.04 | **24** [23] (5%) | +0.30 [+0.29] | 0.78 | 0.69 / 0.73 | 0.64 / 0.58 | +0.88 |

Pending: 0.15 and 0.02 m.

- **0.60 and 0.30 match yours exactly in every column.**
- **At 0.08 and 0.04 m the tumble count is one box higher here,** and r(tip, e) moves by 0.01 at 0.04 m. Every other column matches. Both differences are at low drops, where the near-threshold tumbles sit. I'll check them on 3.9.0 / x86 here after the main runs, to separate version from platform.

## Repo, "0 tumbles at ≤30°": confirmed so far
| deg | tumbled | per-rec at 5 mm f / e | slide inverse R² f / e |
|---|---|---|---|
| 10 | **0** | 0.18 / 0.99 | 0.05 / 0.76 |
| 30 | **0** | 0.77 / 0.99 | 0.48 / 0.30 |
| 38.7 | 78 (15%), r(tip, μ) +0.56 | 0.87 / 0.98 | 0.60 / 0.15 |

Pending: 15, 20, 25°. The per-rec friction values at 10° and 30° match your earlier "repo friction only 0.18–0.77 at ≤30°".

## Class notebook: v15 published (owner's request)
Entry 19 gained a dated note: the cloud's first re-check of the drop-height table matches to two decimals, except the one-box tumble difference at 8 cm, flagged as the first count our computers have disagreed on, with the version-vs-platform check next. **Start any class-page edit from v15.**

ETA for the rest: about 00:15–00:30 UTC. A full results note follows.
