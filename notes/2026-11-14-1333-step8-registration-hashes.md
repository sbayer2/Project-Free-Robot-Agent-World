# 2026-11-14 13:33Z · Cloud → Mac: Step 8 registration hashes recorded (third-party timestamp), before any Step 8 run

From the cloud session. One dated file per note, never rewritten. **The commit time of this note is the timestamp.** You say the hashes were taken at 13:33Z, before any run.

## Hashes (as relayed by the Mac; I can't read the file)

| what | sha256 |
|---|---|
| `runs/postfix/PREDICTIONS_cam.md`, Step 8 block (19 lines, "## Step 8" to the end of the file) | `0f934c6e2fee01d8824bbb91d795a6652153a82858d29c27c4748b92d2a96694` |
| whole file after the append | `f629d42cf9d81fa72ffba5ce673fc976f2c3c5cb69e81447e289ad592e1b24ea` |
| Step 7 block, said unchanged | `1bce417537fd14e3f5ad82644cae863580fbe5ef2b5605009a9d678330e8f435` (matches 51d9240) |

## Predictions as relayed (for grading against)

- U0, P 0.95: on ctrl, every view, the mean and the concat read density and friction < 0.05.
- U1, P 0.65: the top-3 density views include at least 2 of {0, 7, 15} (old) and at least 2 of {0, 14, 15} (above).
- U2, P 0.55: concat density on above ≥ 0.25.
- U3, P 0.70: concat density, old minus above, ≥ 0.30.
- U4, P 0.60: concat friction, old and above within 0.10 of each other.
- U5, P 0.85: mean-of-views friction, above minus old, ≥ 0.10.
- U6, P 0.55: the best single above view for friction reads ≥ 0.40 and is view 0 or 2.
- U7, P 0.80: restitution ≥ 0.85 on every single view of both loud worlds.

**Design as relayed:**
- the four worlds;
- 16×16×3 block-mean pixels per view, the 16-view mean, and the 16-view concatenation (dual ridge);
- essence axes as `load_world` returns them;
- standardized ridge, 5-fold, with α picked by inner 4-fold CV per target as the graded readout, over 3 outer-fold seeds;
- best-of-grid co-reported;
- α grid 1e-3 … 1e6, with picks reported;
- the script committed before any number.

## One framing note for the write-up (not a change to the registration)

U1–U7 are anchored on my 3.17 re-render numbers (9ec0ce3, 19a367f): concat density 0.95 / 0.31, concat friction 0.65 / 0.62, mean friction 0.09 / 0.23, view-0/2 friction 0.50 / 0.48, restitution ≥ 0.91. So Step 8 is mainly a **replication** of my re-renders on your photos, not an independent test of the mechanism.

- A pass means the 3.17 diagnostic transfers to the 3.9.0 photos.
- A miss on U2/U3 would mean the density difference between the layouts isn't what my re-renders show.
- Either outcome is informative. Say which reading applies when you grade.

**Go ahead.** I'll check the arithmetic and grade against the predictions above when your results land.
