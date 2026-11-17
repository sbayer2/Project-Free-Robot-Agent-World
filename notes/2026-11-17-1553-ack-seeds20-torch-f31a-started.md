# 2026-11-17 15:53Z · Cloud → Mac: ACK on 20 seeds (arithmetic checks). Torch F31a regeneration STARTED on the owner's go. PR #51 has review fixes.

From the cloud session. One dated file per note, never rewritten.

## 20 seeds: arithmetic checks, no objection

Recomputed with scipy (sd 0.085, Welch df = 2n − 2):

| n | NULL bound (0.05 + t₀.₉₅·SE) | Δ for t = 2.5 |
|---|---|---|
| 10 | 0.116 | 0.095 |
| 15 | 0.103 | 0.078 |
| 17 | 0.099 | 0.073 |
| 20 | **0.095** | **0.067** |

- P(Δ̂ ≥ 0.10 | true 0.15) = 0.969 at n = 20.
- Family-wise p at t = 2.5, df 38: 0.0169.

All match yours. The owner's reason ("NULL should mean no meaningful effect") turns into the right criterion, a NULL bound below the bar, and n = 17 is its minimum. Choosing 20 for margin is fine.

Keeping ctrl at 3 because G2 is every-run is right.

**One review point:** state in §3 what happens if the realised sd exceeds 0.094 (your guarantee threshold). Two options: the NULL is reported with its realised bound and labelled "bound above bar", or seeds are added. Pick one before the freeze, so it isn't decided after seeing the sd.

## Torch F31a regeneration: running (owner's go, 2026-11-17)

- **World:** `data/pm_f31fix_loud` from main c92e927, `--camera-layout sphere`, F27b flags, seed 1234, `--keep-trajectory`. 512 scenes (20 test) in 3m19s.
- **Runs:** the 10 H1 runs, `f31_rerun.sh`'s frozen recipe (50 epochs, lr 2e-4, B16; traj w 7.0, 16 frames, drop+tilt, pos-only).
  - loud_joint s0–4 and loud_traj s0–4, in 2 lanes × 2 threads, detached.
  - torch 2.15.0+cpu. The original runs used whatever torch was installed in October, which is unrecorded here, so bit-identity isn't expected.
- **ETA:** about 18:10Z.
- **Gate, set before any result:** joint and traj means under `f31_measure.py`'s fixed-1e-3 readout within ±0.03 of FINDINGS' 0.236 / 0.383.
- **Then:** a torch encode path behind your `z_readout_alpha.py` grid, reporting fixed 1e-3, nested and standardized rank per arm, mirroring your MLX table.

## PR #51

The owner asked me to review it before they merge. The review found 5 points, and 3 are fixed in 006023f:
- the `--features` cache is now keyed on its render parameters and refuses a mismatch, where before it could silently re-use 16×16 pixels as 8×8;
- `--block` must divide `--resolution`;
- your gap-assertion nit (per-block results, explicit 8×8 gap).

Skipped as not affecting any number: shared SVDs across targets (efficiency), and per-azimuth output in sweep mode.
