# F31a on the repaired ramp: MLX cross-backend replication

**Status: complete (2026-10-24), recorded 2026-10-26.** A replication check of
the torch F31a repaired-ramp result (`docs/FINDINGS.md`, "F31a on the
repaired ramp", commit `2639edf`) on the canonical MLX trainer. It is not a
new preregistered experiment and does not change F31a's frozen verdicts. It
changes how they read: **the trajectory head never learned the trajectory, on
either backend**, so H1's sign is set by optimizer dynamics, not by trajectory
learning.

## 1. Why

F31a's torch run read H1 RISES (friction z R² 0.236 → 0.383, t 3.31). The
torch↔MLX parity gate (`5ecc9df`) had been run only on the joint arm, on a
pre-fix world. The F31a trajectory arm had never run on MLX.

## 2. Setup

- **Worlds.** `data/pm_f31fix_loud` and `data/pm_f31fix_ctrl` regenerated on
  the Mac from `main` at `01643e1`: `--num-scenes 512 --views 16 --resolution
  128 --seed 1234 --shapes box --keep-trajectory`, loud `--coupling-alpha 1
  --coupling-gain 8`, ctrl `--coupling-alpha 0 --coupling-gain 1`. Every one
  of each world's 8192 renders is byte-identical (sha) to the local
  `pm_f27_{loud,ctrl}`, and the behavior targets match the cloud's to five
  decimals (predict-mean MSE 0.13741; `tilt.slid_distance` 78.8 % of target
  variance).
- **Recipe.** The frozen F27b recipe (lr 2e-4, 50 epochs, batch 16) and the
  frozen F31 head (`--trajectory-weight 7.0 --trajectory-frames 16
  --trajectory-probes drop,tilt --trajectory-pos-only`); loud joint/traj
  seeds 0–4, ctrl joint/traj seeds 0–2. `pseudomarble.models.train` (MLX
  0.31.2, M5 Pro), about 3 min per run.
- **Readouts.** `scripts/f31_measure_mlx.py` (f31_measure's probes and report
  schema), graded by the frozen `scripts/f31_verdicts.py` and Amendment 2's
  co-reports from `scripts/f31_coreport.py`. Runner: `scripts/f31_mlx_rerun.sh`.

## 3. Plan and prediction, as written before the data

Written 2026-10-24T12:40:35Z, after loud seed 0 of both arms had finished
(12:36Z, 12:39Z) and before any seed 1–4 run (first finished 12:44Z). Times
are the session transcript's and the runs' `metrics.json` mtimes; the
original scratch file was lost with the session scratchpad and is restored
here from the transcript verbatim.

> Seen before writing: MLX loud seed 0 only (joint fric 0.319 / held-out
> 0.624; traj 0.298 / 0.730; gains 1.065/1.044), plus all cloud torch F31a
> numbers.
>
> Reading rule: apply the frozen F31a bars unchanged, read as a REPLICATION
> CHECK, not a new verdict: H1 (loud-traj − loud-joint friction 5-fold z R²):
> RISES d ≥ +0.10 & t ≥ 2.5; NULL |d| < 0.05; else INCONCLUSIVE. G2: every
> ctrl run friction z R² < 0.10. Also report D_L and held-out (b) as in
> Amendment 2. "Replicates" = MLX H1 RISES with G2 pass. "Does not
> replicate" = NULL. INCONCLUSIVE = say so.
>
> Prediction (mine, before data): replicates RISES 45 / INCONCLUSIVE 40 /
> NULL 15. Reason: torch CI [+0.04, +0.26] is wide and MLX seed 0 shows
> traj ≈ joint on the 5-fold readout.

## 4. Result: does not replicate

| repaired world, MLX | joint | + trajectory head |
|---|---|---|
| loud friction z R², 5-fold (5 seeds) | 0.319 / 0.361 / 0.286 / 0.181 / 0.187 (0.267) | 0.298 / 0.194 / 0.210 / 0.212 / 0.296 (**0.242**) |
| loud friction R², ridge on train z → 20 held-out scenes | 0.624 / 0.646 / 0.510 / 0.659 / 0.553 (0.598) | 0.730 / 0.623 / 0.568 / 0.583 / 0.478 (0.596) |
| ctrl friction z R² (3 seeds) | −0.204 / −0.171 / −0.216 | −0.239 / −0.186 / −0.205 |
| held-out behavior gain | 1.052–1.065 (ctrl 1.033–1.036) | 1.043–1.049 (ctrl 1.042–1.045) |

- **Gates:** G1 PASS (PR 31.4–91.7); G2 PASS (every ctrl friction z R² < 0).
- **H1 → NULL:** Δ −0.025, t −0.59 (≈95 % CI [−0.13, +0.08]).
- **Amendment 2 (a), leakage-adjusted:** Δ_L −0.012, SE 0.047, t −0.25.
- **Amendment 2 (b), held out:** 0.596 vs 0.598.
- **H2 → NULL by the frozen rule** (Δ −0.013, t −4.88), the same metric
  saturation as on torch: every arm sits at gain 1.03–1.07, and no run escapes.

The joint arm agrees with torch (0.267 vs 0.236, Welch t +0.59). The
trajectory arm does not (0.242 vs 0.383, t −4.29). The prediction was wrong:
it gave NULL 15 %.

## 5. The trajectory head collapsed

| loud-traj heads (5 seeds each) | MLX default | MLX + bias correction |
|---|---|---|
| train trajectory gain | 0.983–0.991 | 0.978–0.981 |
| held-out trajectory gain | 1.007–1.009 | 1.007–1.009 |
| prediction std across scenes (train) vs target std | 0.002–0.009 vs 0.249 | 0.002–0.004 vs 0.249 |
| drop block gain, train / held out | 0.52–0.57 / 0.40–0.47 | 0.52–0.56 / 0.46–0.47 |
| tilt block gain, train / held out | 0.99 / 1.02 | 0.98 / 1.02 |
| dead hidden ReLUs: behavior head / trajectory head | 70–76 % / 62–65 % | 67–71 % / 59–63 % |

Every loud-traj head predicts nearly the same path for every scene. On this
world the tilt block is 99.4 % of the trajectory target's variance and the
drop block 0.6 %; the head does worse than the train mean even on the
legible drop block. Most hidden ReLUs in both heads of the trajectory arm are
dead on every scene (the joint arm was not measured this way).

**Torch collapsed too.** The cloud session's diagnostic
(`scripts/f31_head_diag.py`, PR #46) found the same in all 8 torch traj seeds
(5 loud, 3 ctrl): train gain about 0.98, prediction std about 0.002 against
a target std of 0.249, held-out trajectory gain never above 1.014 in any
epoch. Its numbers were measured before that container was reclaimed; the
torch checkpoints no longer exist.

## 6. Post-hoc diagnostic: Adam bias correction (H-BC), falsified

MLX's `optim.AdamW` defaults to `bias_correction=False` (mlx 0.31.2); torch's
AdamW always corrects. Hypothesis, written 2026-10-24T13:24:35Z before the
runs: uncorrected early steps are 2–3× larger, and with the trajectory loss
starting at 19× the behavior loss that collapses the head; correcting should
un-collapse it (train gain > 1.05) and move H1 toward torch. Falsified if
heads still collapse with correction on, or heads fit but friction does not
rise.

Same 10 loud runs with `optim.AdamW(learning_rate=args.lr,
bias_correction=True)` at `models/train.py:196` (a local patch, never
committed; `ROOT=runs/f31fix_mlx_bc`):

| loud, MLX + bias correction | joint | + trajectory head |
|---|---|---|
| friction z R², 5-fold (5 seeds) | 0.264 / 0.198 / 0.308 / 0.377 / 0.272 (0.284) | 0.176 / 0.090 / −0.015 / 0.112 / −0.025 (**0.068**) |
| friction R², held out | 0.577 | 0.320 |
| density z R² | 0.641 | 0.551 |

H1 by the frozen bars: Δ −0.216, t −4.47 (FALLS). No ctrl arms were run for
this diagnostic, so G2 and Δ_L are not available.

**Falsified.** The heads still collapse, and H1 flips to FALLS.

## 7. Reading

Same worlds, recipe and head; three trainers; three H1 verdicts:

| trainer | joint | traj | Δ | t | H1 (frozen bars) | head |
|---|---|---|---|---|---|---|
| torch (cloud, `2639edf`) | 0.236 | 0.383 | +0.147 | +3.31 | RISES | collapsed (8/8) |
| MLX default | 0.267 | 0.242 | −0.025 | −0.59 | NULL | collapsed (5/5) |
| MLX + bias correction | 0.284 | 0.068 | −0.216 | −4.47 | FALLS | collapsed (5/5) |

Neither backend learned the trajectory. The friction content of z in the
trajectory arm therefore moves with how each optimizer handles a loss term
that starts 19× louder than the behavior loss and that the head cannot fit,
not with the trajectory being learned. F31a's H1 RISES stands as graded by
its frozen rule on torch, but it is **not evidence that trajectory
supervision puts friction into z**. The open question F31a's Amendment 2 (c)
named (denser target or louder one) is superseded: under raw MSE on this
target, the head does not learn at all.

One F31a conclusion does not depend on the mechanism. Torch's trajectory arm
ended with friction z R² 0.38 (0.66 held out) while its ctrl arm stayed below
zero, so an encoder trained on these photos can hold far more friction than
the joint arm's 0.24. Amendment 1's reading "C1 NULL → the pixel limit is
binding" stays refuted.

Consequence for F31b: rescale the target before any rerun (per-probe-block
standardization, or log slide distance). Lowering the loss weight alone
(the proposed w ≈ 0.36 arm) leaves a 99 % tilt-path target in metres that
both backends' heads settled on the mean of.

## 8. Reproduce

```bash
bash scripts/f31_mlx_rerun.sh                    # 16 runs (~50 min), then the report
bash scripts/f31_mlx_rerun.sh report             # readouts only
python scripts/f31_head_diag_mlx.py --data data/pm_f31fix_loud \
    --runs runs/f31fix_mlx/loud_traj_s{0,1,2,3,4} --out runs/f31fix_mlx/head_diag_loud_traj.json
```

Related, from the same session: `scripts/tilt_bug_oracle.py` measures what the
0.349° ramp did to the behavior target on `pm_big` (F18 oracle and variance
shares before vs after the fix, zero training).
