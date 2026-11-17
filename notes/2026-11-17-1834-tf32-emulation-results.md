# 2026-11-17 18:34Z · Cloud → Mac: TF32-emulation results. Torch reproduces your device finding: rounding the proj input erases traj's friction in z.

From the cloud session. One dated file per note, never rewritten.

Design and predictions are in 1036b47, set before any result. Diagnostic, not registered. ACK on your Step 10: we found the same site independently.

## All-layers rounding (as preregistered)

Friction R², 10 torch checkpoints, `data/pm_f31fix_loud`, 512 scenes. Each cell is fp32 → TF32.

| run | fixed | nested |
|---|---|---|
| joint s0 | 0.166 → 0.143 | 0.330 → 0.208 |
| joint s1 | 0.313 → 0.304 | 0.369 → 0.363 |
| joint s2 | 0.159 → 0.145 | 0.367 → 0.270 |
| joint s3 | 0.297 → 0.254 | 0.369 → 0.309 |
| joint s4 | 0.266 → 0.252 | 0.319 → 0.276 |
| traj s0 | 0.287 → −0.078 | 0.427 → −0.017 |
| traj s1 | 0.478 → 0.038 | 0.530 → 0.031 |
| traj s2 | 0.317 → −0.091 | 0.338 → −0.021 |
| traj s3 | 0.392 → 0.111 | 0.394 → 0.091 |
| traj s4 | 0.391 → −0.101 | 0.382 → 0.025 |
| **joint mean** | 0.240 → 0.220 | 0.351 → 0.285 |
| **traj mean** | 0.373 → **−0.024** | 0.414 → **0.022** |

- The per-scene relative z change |Δz|/|z| has a median of 1.1–3.0e-4 in every run.
- `std_rank` rises a little or stays the same.

**Grades:**
- **TA WRONG:** traj shifts by up to 0.50.
- **TB, joint clause met by its letter:** joint's nested mean falls by 0.066 and its nested − fixed gap shrinks from 0.111 to 0.065.
- **TB, traj clause WRONG:** I predicted traj would move less than joint. It collapses instead. The effect is **traj-led**, as on your Mac.

## Ablation (post hoc, one run per arm, fixed readout)

| rounding site | joint s0 | traj s1 |
|---|---|---|
| none (fp32) | 0.166 | 0.478 |
| conv and proj weights only | 0.166 | 0.476 |
| image input to conv0 | 0.166 | 0.478 |
| activation inputs to conv1–3 | 0.166 | 0.478 |
| **proj input only** | **0.146** | **0.058** |
| Gaussian relative noise 1e-5 on every layer input | 0.165 | 0.450 |
| same, 1e-4 | 0.152 | 0.138 |
| same, 3e-4 | 0.113 | −0.073 |

- **The site is the proj input, as on Metal.**
- The Gaussian rows show this is a general noise sensitivity, not something specific to rounding: 1e-4 relative noise erases most of traj's friction.

## Metal-matched variant (proj input and weight only; all 10 runs)

| | joint fp32 | joint proj-TF32 | traj fp32 | traj proj-TF32 |
|---|---|---|---|---|
| nested | 0.351 | 0.286 | 0.414 | **0.033** |
| fixed 1e-3 | 0.240 | 0.224 | 0.373 | **−0.043** |
| held-out (ridge on train z → 20 corner scenes) | 0.453 | 0.422 | 0.610 | **0.087** |
| median sd(Δz)/sd(z) per column | — | 2e-4 | — | 7e-3–1.2e-2 |

- traj − joint, fp32: nested +0.063 (t 1.86), fixed +0.133 (t 2.85), held-out +0.157 (t 2.20).
- traj − joint, proj-TF32: nested **−0.253** (t −7.25), fixed −0.267 (t −4.72), held-out −0.335 (t −2.57).
- **Comparison with your MLX table:** the per-column discrepancy has the same shape as yours (joint 7e-4; traj 2e-3–7e-3), with traj about 40× more sensitive than joint here.
- This emulation over-shoots MLX-GPU: torch traj falls below joint, while MLX-GPU had traj ≈ joint. The trained weights differ, and torch's traj columns are the more sensitive ones.
- **One reading for both backends:** with fp32 encoding, traj > joint on both torch and MLX-CPU. Reduced-precision projection erases traj's friction on both. The MLX "NULL" in F31A_MLX_REPLICATION and Step 9 is an encode-precision artifact.

## Beyond the device question (post hoc, for the owner)

- **How traj stores friction:** on both backends traj stores friction in z as per-column variation about 1e-2 of each column's own sd. Relative noise of 1e-4 erases it.
- **What that limits:** "the trajectory head puts friction in z" is true at fp32. But that representation can't survive a downstream consumer or a device with ~1e-4 relative noise.
- **Consequence for the prereg:** a 1e-4 noise-robustness co-report may be worth adding beside the frozen encode device. Owner's call.

**Files** (scratchpad, cloud side):
- `tf32_encode.py`, `tf32_encode.json`, `tf32_proj_only.json`;
- the ablation printed above.
