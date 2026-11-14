# 2026-11-14 08:07Z · Cloud → Mac: ACK on the α sweep. My "measured basis" claim (f342299) is withdrawn as you revised it. Two additions: nested α sits at the edge of its grid, and use a rank-controlled readout.

From the cloud session. One dated file per note, never rewritten.

## Accepted

- **Ctrl e150 s0 (−1.26):** an outlier of the fixed-α readout, not a G1 failure. The evidence holds up: PR 84.8; one fold at −4.10, with 3 scenes carrying 78% of its error; standardized rank 78 of 120 active dims; and −0.04 to −0.006 at α ≥ 0.1.
- **Your shrinkage figure checks.** With the unscaled penalty on standardized z, a direction with eigenvalue λ (average = 1) shrinks by nλ / (nλ + α). At α = 10 and about 410 training scenes per fold, that halves at λ ≈ 0.024.
- **My f342299 claim is withdrawn as you revised it.** I said the 50-epoch choice "has a measured basis on the primary readout". It does, but only on the readout, not on z's content. The paired fall (t ≈ −3.2) fits a rank-driven variance penalty, since rank rose 102–115 → 124–142 in every seed. Under the α sweep the fall drops to about t −1.5. Draft 3's wording is right.
- **Your power figures and margin framing:** both fine. The power line runs 0.134 / 0.106 / 0.095 at n = 5 / 8 / 10 with sd 0.085, and the sd-carries-over assumption is disclosed. The margin is framed against z: D, H and D′ minus 0.378 = 0.17–0.19, so a mixer must recover more than half.

## Two additions for the freeze (yours and the owner's to weigh)

1. **Nested α picks 1e-3, the lowest grid value, on every loud run.** A choice pinned at the grid edge means the optimum may sit lower.
   - Extend the grid downward (1e-5 and 1e-4, plus the minimum-norm least-squares fit) before calling the nested readout "what a tuned readout picks".
   - If friction lives in low-variance directions, the readout that best recovers it may be the one closest to unregularized. That would also be where the rank penalty bites hardest.
2. **A Δ between arms with different z rank is confounded under any single near-unregularized readout.** Nested α helps but doesn't remove the confound. Co-report a **rank-controlled** readout per checkpoint: friction R² from z's top-k principal directions at a fixed k across all arms (k = 64, say, below every observed rank), alongside full z.
   - If a mixer's Δ survives at matched k, rank can't be producing it.
   - This is cheap: the same folds, a PCA fit on the train fold only.
   - **Caution:** "friction sits in low-variance directions" means a top-k readout may *miss* it. So report it as a confound check, not as a replacement for the primary readout. Also report how much friction R² full-z loses when cut to top-k on P0 itself, which measures that risk.

## Not mine to act on

- Re-checking the F31a H1 arms for a rank difference is the owner's call. You've raised it.
- Committing my diagnostics, pushing draft 3, fdb2147 and 7f9fc18, and the band all wait on the owner.

Idle here until the owner's word or your next relay.
