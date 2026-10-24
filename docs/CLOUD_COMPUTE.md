# Cloud compute: what the sandbox can do, and the torch↔MLX parity gate

*(Written 2026-09-26, before any torch-trained number on an f27b world existed.
The parity rule in §3 is frozen as of the commit that adds this file.)*

## 1. What a cloud session is

Measured in this session, not assumed:

| resource | value |
|---|---|
| CPU | 4 vCPU (Xeon @ 2.1 GHz) |
| RAM | 15 GB |
| GPU | **none** |
| MuJoCo render, 128 px, software GL (`MUJOCO_GL=osmesa`) | 26 ms/frame |
| 512-scene world, 16 views, trajectories kept | ~3 min (ctrl: 173 s) |
| torch training step, canonical model (1.0M params, B8, N16, 128 px) | 0.74 s |
| one canonical training run (492 train scenes, 50 epochs, B16) | ~15 min uncontended (see §5) |

So the cloud's advantage over the Mac is **width, not speed**: independent
sessions run in parallel, each at roughly the numbers above. Datasets are
deterministic from their seeds and the container is ephemeral, so the cloud
is for work whose output is small (reports, verdicts, code), not for shipping
data to the Mac.

Setup that a fresh session needs (not in the repo's deps): `apt-get install
libosmesa6`, `pip install -e ".[dev,mujoco]" torch numpy scipy pillow`
(torch from the CPU index), and `MUJOCO_GL=osmesa` for generation.

## 2. Why a parity gate

`models/train_torch.py` mirrors `models/train.py` (same parser, config and
warmup; `torch_net` mirrors `mlx_net`), but it is not the canonical trainer:
initializers differ between frameworks, and F29 found that **the init, not
the objective, sets encoder quality in this regime**. The cloud also
re-renders the world with a different GL stack than the Mac. Either could
move gains. Until a gate passes, torch-trained numbers may be used only in
experiments whose **every arm** trains on torch; they may not be pooled with
or compared against MLX numbers.

## 3. The gate (frozen)

- **World:** `pm_f31_loud` — the F27b loud recipe (`--shapes box
  --coupling-alpha 1 --coupling-gain 8`, 512 scenes, 16 views, 128 px, seed
  1234), regenerated in the cloud with `--keep-trajectory` (trajectories do
  not touch images or labels).
- **Recipe:** F27b's frozen recipe — lr 2e-4, 50 epochs, batch 16, default
  `ModelConfig`; seeds 0, 1, 2.
- **Reference:** F27b loud joint gains (MLX, Mac): per-seed
  [5.040, 5.910, 6.067], mean **5.672**.
- **Apparatus check (must hold before the gains are read):** measured
  roughness↔friction r on the regenerated world within ±0.02 of F27b's
  0.992; train/test split sizes equal to F27b's.
- **PASS** iff all of:
  1. torch 3-seed mean held-out gain in **[4.82, 6.52]** (±15 % of 5.672);
  2. every torch seed's gain ≥ 3.0 (no seed fails outright);
  3. every torch seed's final latent PR ≥ 8 (F10/F12 healthy band — no
     collapse).
- **FAIL** otherwise. A FAIL does not stop torch experiments; it confines
  them to all-torch designs and is reported as a finding about the
  backend, with the per-seed numbers.
- The ±15 % band is wide on purpose: it asks "same regime", not "same
  number". MLX's own seed spread (5.04–6.07) is ±9 % around its mean.

## 4. Consequence if PASS

Cloud sessions become a certified second training site for torch-mirrored
experiments at the f27b scale, and multi-seed arms can be fanned out across
parallel sessions. Every cloud-trained result is still labeled with its
backend in FINDINGS.

## 5. Result (2026-09-26) — PASS

Run exactly as frozen in §3 (`scripts/parity_verdict.py`, report
`runs/parity/verdict.json`).

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| torch, cloud | 4.893 | 4.619 | 5.551 | **5.021** |
| MLX, Mac (F27b) | 5.040 | 5.910 | 6.067 | 5.672 |

- Apparatus: r = 0.9922, split 492/20 → PASS. Stronger than the frozen
  check required: the cloud-regenerated worlds reproduce F27b's published
  appearance-oracle ceilings **to three decimals on all four arms**
  (0.957 / 1.841 / 3.226 / 4.706), so physics outcomes and material labels
  are identical to the Mac's; only rendered pixels could differ.
- Gate: mean 5.021 ∈ [4.82, 6.52]; min seed 4.619 ≥ 3.0; PR 74.1–89.2 ≥ 8
  → **PASS**.
- Disclosed beyond the rule: torch sits **11.5 % below** MLX (−0.651,
  Welch t −1.54 at n = 3, seed sd 0.48 vs 0.55). Not significant, but the
  direction is recorded, and torch-trained numbers keep their backend label.
- Observed (post hoc, labeled): every seed sits at predict-the-mean
  (gain ≈ 1.0) for ~20 epochs, then escapes — seeds 0 and 2 around epoch
  25, seed 1 around 35. The late escaper finished lowest and was still
  rising at epoch 49, so **escape timing is one concrete mechanism of the
  seed spread**, and the 50-epoch recipe truncates late escapers (cf. F10,
  F12, F29's init thread).
- Timing, corrected: an uncontended run takes **~15 min** (875–988 s), not
  the ~30 min estimated in §1.

**Consequence:** the cloud is a certified second training site for
torch-mirrored experiments at the f27b scale. Rules that still hold:
every cloud-trained number carries its backend; cross-backend
comparisons disclose the −11.5 % offset; all-torch designs remain the
cleanest.
