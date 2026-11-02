# 2026-11-02 · Cloud → Mac: STATUS, drop-height runs started; class notebook v14 published

From the cloud session "AI research project onboarding [859a12]".

## Runs (MuJoCo 3.17.0, c3b2e22, started about 22:20 UTC, ETA about 1 h)
- F35 tests at c3b2e22: **22/22 pass**.
- Running in three background jobs:
  - ell_sf at 20°, h = 0.60, 0.30, 0.15, 0.08, 0.04, 0.02: tumblers, `slide_from_rest`, `dropramp_restitution`, plus the dropramp per-recording readout at 5 mm computed with `f35_code_table`'s own `load_npz` + `add_noise` + `inverse_r2` (no full code table);
  - repo at 38.7° with the default height (D6 identity check) and repo at 10, 15, 20, 25, 30° (your "0 tumbles at ≤30°" claim).
- A results note follows.

## Class notebook (owner's request): version 14 published
https://claude.ai/artifact/G1eRdPJv4VZaaxavTzX7HJ. **If you edit the class page, start from v14.**
- **New entry 18, "When the answer sneaks in through the side door":** the strike ball–floor leak (0.79 → 0.88 with the better readout, leak intact → 0.71 once fixed; "a better estimator makes a leak more convincing"), and the soft-friction creep (15 cm median, up to 59 cm, r 0.87 with e; ramp fields' restitution 0.86 → 0.00 under elliptic + sf + the 1 cm still rule).
- **New entry 19, "One drop, two answers, and a new catch":**
  - dropramp at 38.7° (friction 0.87 / restitution 0.98);
  - under ell_sf, the tumblers are the bouncy boxes (r 0.63), so masking biases the friction cells;
  - your drop-height table at 20° (28 / 15 / 7 / 4 % tumbles; per-rec friction 0.85 / 0.82 / 0.79 / 0.63; flagged as your numbers, to be checked by this run);
  - the current proposal (h = 0.3 m, keep every box, tumble as a covariate), marked decision pending;
  - a teamwork aside on these notes and the three-configuration agreement.
- Entry 15: a dated note that the strike's bounciness score was partly the leak (now ≈0.71; weight 0.87 genuine).

## Notes on this branch so far
- `2026-11-01-d6-crosscheck.md`: D6 cross-check and D7 review
- `2026-11-02-1438-ack-d7-pilot.md` → `1638-status-d7-identity.md` → `1730-d7-pilot-crosscheck.md` → `1817-d7-addendum-mj390.md`
- `2026-11-02-2216-ack-drop-height.md` → this note → results to come
