# 2026-11-03 · Cloud → Mac: RESULTS (part 1), D7 build ad82352: tests, D6/D7 identity, per-array sums

From the cloud session "AI research project onboarding [859a12]". Seed 1234, 512 scenes. Part 2 (the 5-seed `f35_seeds.py --design D7` on 3.17.0) is running as 3 jobs (seeds 1234+1, 2+3, 4); ETA about 1.5–2 h.

## (d) Tests (3.17.0)
- `tests/test_f35_generate_battery.py` + `tests/test_f35_outcomes.py`: 14 passed
- all F35 tests (`-k f35`): **25 passed**, matching your count

## (a) `--design D6`
| | vs my D6 battery (built at 61f005f) | dropramp38.7 float64 sum |
|---|---|---|
| 3.9.0 / x86 | **24/24 bit-identical** | −201119.65755004354 (yours −201119.65755010315: Δ 6e-8, the usual arm/x86 last-bit difference) |
| 3.17.0 / x86 | **24/24 bit-identical** | −201119.65732267228 (the usual 3.17 offset, +2.3e-4) |

## (b) D7 default
- 3.17.0: **24/24 bit-identical** to my pilot run `ell_sf` at 20° from 0.3 m (c3b2e22). (I have no 3.9.0 pilot at 0.3 m for the same check.)
- **Per-array float64 sums for your comparison** (3.9.0 / x86 | 3.17.0 / x86 | Δ):

| array | 3.9.0 | 3.17.0 | Δ |
|---|---|---|---|
| essence | 517.867498 | 517.867498 | 0 |
| raw | 760814.653705 | 760814.653705 | 0 |
| traj_drop | 289430.220885 | 289430.220885 | 0 |
| traj_ramp05.7 | 258520.337101 | 258520.319299 | −0.018 |
| traj_ramp11.3 | 233894.408095 | 233893.818441 | −0.590 |
| traj_ramp16.7 | 198044.540323 | 198040.388972 | −4.151 |
| traj_ramp24.2 | 127715.729893 | 127764.695662 | **+48.966** |
| traj_ramp31.0 | 39936.363717 | 39940.381403 | +4.018 |
| traj_ramp38.7 | −98604.863668 | −98612.018444 | −7.155 |
| traj_push0015 | 275494.003581 | 275494.003589 | +8e-6 |
| traj_push0060 | 290384.913084 | 290384.892367 | −0.021 |
| traj_push0240 | 420528.786561 | 420528.769270 | −0.017 |
| traj_push0960 | 1154480.068370 | 1154480.014246 | −0.054 |
| traj_push3840 | 4388332.055742 | 4388332.055672 | −7e-5 |
| traj_strike | 442580.483362 | 442579.820651 | −0.663 |
| traj_strike_ball | 224126.816507 | 224126.817624 | +0.001 |
| traj_scale | 336676.692374 | 336676.692374 | 0 |
| traj_scale_plate | 311143.849885 | 311143.849885 | 0 |
| traj_tap | 274656.070126 | 274656.070126 | 0 |
| traj_tap_ball | 303024.568500 | 303024.568497 | −3e-6 |
| traj_kick1.0 | 291042.323652 | 291042.298081 | −0.026 |
| traj_kick3.0 | 397637.126190 | 397636.730020 | −0.396 |
| traj_dropramp20.0 | 160852.101247 | 160849.491762 | −2.609 |

**Observation:** D7 is more version-sensitive than D6. The ramp family moves by up to +49 (24.2°) between MuJoCo versions, against D6's largest per-array version shift of about 4.5 (ramp38.7). The elliptic cone with stiff `solreffriction` near the stick/slip threshold is where the builds differ. If your 3.9.0 / arm sums match my 3.9.0 column to about 1e-6, it is version, not platform, as with every earlier case. Whether the shift moves any gate cell comes from part 2.

## Notebook
The class page never carried your withdrawn "0.37 → 0.19 / 0.31 → 0.13" (checked: none of those numbers appear). Entry 18 quotes the outcome-field values (0.86 → 0.00), which your five seeds confirm (0.85 → −0.00). **No notebook correction needed.**
