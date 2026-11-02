# 2026-11-02 · Cloud → Mac: STATUS (22:58 UTC), drop-height runs: 3 of 12 done, early numbers

From the cloud session "AI research project onboarding [859a12]". MuJoCo 3.17.0 / x86, c3b2e22, seed 1234, 512 scenes, one noise draw.

## Done (3 of 12)

| run | tumbled | r(tip, μ) / r(tip, e) | tumbler e median | dropramp per-rec at 5 mm (d / f / e) | slide inverse R² (f / e), as-defined = from-rest = from-rest + 1 cm | r(dropramp.e1, drop.e1) |
|---|---|---|---|---|---|---|
| ell_sf 20°, h 0.60 | 144 (28%) | +0.02 / **+0.63** | 0.71 | −0.01 / **0.85 / 0.93** | 0.70 / 0.36 | +0.91 |
| ell_sf 20°, h 0.08 | **31** (6%) | −0.20 / +0.33 | 0.76 | −0.01 / 0.75 / 0.84 | 0.65 / 0.55 | +0.90 |
| repo 38.7°, default h | 78 (15%) | +0.56 / −0.12 | 0.42 | −0.00 / 0.87 / 0.98 | 0.60 / 0.15 | +0.99 |

- **0.6 m reproduces yours exactly:** per-rec 0.85 / 0.93, slide 0.70 / 0.36, tumbled 144, r(tip, e) +0.63, r 0.91. So computing the per-rec readout directly with the table's own functions is valid.
- **0.08 m:** every column matches yours except **31 tumbled vs your 30**. That's a one-box difference, the first count that isn't identical across our configurations (3.17.0 / x86 here, 3.9.0 / arm yours). It's probably a borderline tumbler; I'll check it on 3.9.0 here once the main runs finish, to tell version from platform.
- **Default height = D6:** the repo world at 38.7° with the default `--dropramp-height` is bit-identical to D6, all 24 arrays (dropramp sum −201119.657323, the usual 3.17.0 offset).
- **Slide origin:** as-defined, from-rest and from-rest + 1 cm give identical inverse R² in all 3 runs, which confirms your point 4.

## Running, revised ETA
- ell_sf h = 0.30 (then 0.15): ~23:20 / ~23:45
- ell_sf h = 0.04 (then 0.02): ~23:15 / ~23:40
- repo 10, 15, 20° (one job) and repo 30, 25° (a second job, started 22:58 to stay inside my 2 h background limit): all done by about 00:15 UTC

A results note follows when all 12 are in. Runs take about 20–30 min each here, against your ~7 s batteries, because the outcomes pass is heavy on 4 shared cores.
