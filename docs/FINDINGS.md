# pseudo-marble — model & initial sandbox findings

Status: the apparatus is complete **and the headline experiment has been run on the
Mac.** F1–F7 record the model and what the **in-sandbox** (Linux CPU, no Mac) tests
established; **F8** is the first real-MuJoCo result (probe-label stability); **F9**
is the 5-seed coherence measurement (a weak, seed-unstable positive); **F10** is the
20-seed resolution — the mean gain is significant, the "instability" is two training
basins (escaped vs encoder-collapsed), and raw coherence is inflated by the collapsed
basin. **Quote F10, not F9.** **F11** turns the instrument outward: an external LLM
world model (Qwen-AgentWorld) scored against our MuJoCo ground truth — reasoning
format transfers, contact physics doesn't, topple calibration beats our own model.

---

## 1. What the model is

A single shared-latent network that, from multi-view images of an object, emits
three projections of one understanding:

```
images (B, N, H, W, 3)
   └─ encoder: per-view CNN → global avg-pool → Linear → mean over N views → z
        z  ──► behavior head : 21-dim drop/tilt/push outcomes   (the real target)
        z  ──► essence  head : (density, friction, restitution) (aux supervision)
        z  ──► render   head : conv decoder → mean-view image    (appearance)

loss = behavior MSE + 0.3·essence MSE + 1.0·render MSE
```

- **Render head** is a lightweight conv decoder (Linear → seed map → [nearest
  upsample ×2 + Conv] × k → sigmoid), *not* a Gaussian-splat decoder — we measure
  coherence, not photorealism. `image_size` must be `render_seed·2^k`.
- **One architecture, three mirrored backends** from one `ModelConfig`:
  `mlx_net` (canonical trainer, Apple Metal), `numpy_net` (forward-only),
  `torch_net` (CPU, trainable in-sandbox).

### The task it learns
- **Continuous materials** (`MaterialSampler`): a hidden 4-factor essence
  (heaviness/grip/hardness/clarity) generates *both* physics and appearance, with
  noise so appearance is **predictive but not invertible** — the model must infer
  the essence, not look it up.
- **Behavior under action** (`probes.py`): drop (bounce/settle), tilt (slide),
  push (slide vs **topple** — shape × material interact). Outcomes are summary
  stats: toppled, settle_time, slid_distance, n_bounces, max_height, path_length,
  final_tilt_deg.
- **Generalization split** (`RegionHoldout`): a region of essence-space is held
  out for test, so the metric is interpolation/extrapolation, not memorization.

### The measurement (`coherence_bench.py`)
Nudge `z`; if the directions that change the *render* also change the *behavior*,
the heads share structure (coherent) vs. live in private subspaces (~0). Compared
against an **independent baseline** (render-only + behavior-only models with
separate latents, joined as a disjoint latent → ~0 by construction).

---

## 2. Initial findings from sandbox tests

All numbers below are from a 4-core / 15 GB Linux CPU container, using the numpy
and torch backends. They establish that the **apparatus works**; they are **not**
the scientific result.

![initial sandbox findings](figures/sandbox_findings.png)

*(Figure regenerable from live data with `python scripts/make_figures.py`.)*

### F1 — No usable Linux MLX; numpy/torch fill the gap
The pip `mlx` wheel on Linux x86 is non-functional (missing `libmlx.so`), and
there's no GPU (so cudamat/Theano/CUDA paths are out). NumPy and CPU PyTorch
install and run, which is why the model has three mirrored backends: MLX stays
the canonical Mac trainer; numpy validates forward shapes anywhere; torch CPU
verifies the training loop converges.

### F2 — The authored coupling is real but noisy (as intended)
Across 500 sampled materials, "looks metallic" vs. log-density correlates inside
the band `0.3 < r < 0.99` — predictive enough to learn, noisy enough that it
can't be a lookup table. (Guarded by a test.)

### F3 — The full model trains (gradients flow end to end)
The torch overfit smoke (encoder + all three heads, incl. the render decoder)
drops loss ~4.8–4.9× in 200 steps on a fixed batch. This proves the architecture
is differentiable and wired correctly — **not** that it learns generalizable
physics.

### F4 — In-sandbox scale ceiling (full model, incl. render head)

| case | img | views | params | ms/step | peak RSS | loss (overfit) |
|---|---|---|---|---|---|---|
| tiny | 32 | 4 | 0.07M | 15.8 | 328 MB | 0.449→0.304 |
| small | 64 | 8 | 0.19M | 49.9 | 379 MB | 0.430→0.198 |
| medium | 128 | 12 | 0.41M | 268 | 668 MB | 0.416→0.180 |
| **large** | **128** | **16** | **1.01M** | **847** | **913 MB** | **0.420→0.114** |
| xl | 256 | 8 | 1.02M | 2357 | 1309 MB | 0.430→0.126 |

Comfortable to **~1M params / 128px / 16 views** (sub-second–~0.85 s/step) for
correctness + convergence checks. Past 256px, CPU step time makes real training
impractical — confirming MLX/Metal on the Mac as the canonical trainer. **Memory
is never the bound** (peak 1.3 of 15 GB); compute time is.

### F5 — The coherence metric had a subtle bug, now fixed
Unit-sphere perturbation directions make two *disjoint* latent subspaces
anti-correlated (a simplex artifact), so an independent baseline would score
*negative* instead of ~0. Fixed to **iid Gaussian directions**; a test locks it.

### F6 — ⭐ The shared-vs-independent gap is mostly architectural, not learned
Running the harness on **untrained** numpy models:

```
shared_coherence        0.4565
independent_coherence    0.0076   ← "glued together" control, ~0 as expected
architectural_coherence  0.3591   ← a second untrained shared init
```

An **untrained** shared model already scores ~0.45 coherence — purely because
both heads read the same `z`. So a naive "shared beats independent" result would
be an **architectural artifact, not evidence of a learned eigenvector**. Had we
run that comparison on the Mac and seen a big gap, we'd have falsely "confirmed"
the hypothesis.

The honest signal is therefore:

> **`learned_coherence = trained_shared − untrained_shared`**, averaged over
> several untrained seeds (the baseline itself varies ~0.36–0.46), evaluated on
> **held-out essence regions**, and paired with **behavior generalization**.

This control is baked into `compare()`. The apparatus caught this *before* any
Mac time was spent — arguably the most valuable sandbox finding. Its logic
(subtract the architectural prior, keep the residual) is the same move predictive
coding makes for a prediction error — see
[`PREDICTIVE_CODING.md`](PREDICTIVE_CODING.md).

---

### F7 — Per-probe difficulty (methodology illustration)

Which behavior outputs are hard to predict from appearance? Real per-field
difficulty needs the MuJoCo run, but training the *actual* torch model on a
synthetic toy-physics stand-in shows the apparatus and the expected ordering:

![per-probe difficulty](figures/probe_difficulty.png)

Smooth outcomes (settle time, max height) are easiest; **toppling is hardest** —
it is a bimodal threshold (tips or it doesn't), so a regression head cannot land
cleanly near the tipping point (the "chaos near tipping points" risk from
`BEHAVIOR_TASK.md`, made visible). The model beats a predict-the-mean baseline on
every field. The *numbers* are from a synthetic stand-in
(`scripts/figure_probe_difficulty.py`); the real ordering comes from the Mac run.

---

### F8 — ⭐ First real-MuJoCo result: the topple label is locally chaotic but the contamination is bounded — and degenerate for half the shapes

This is the **first empirical result from real MuJoCo on the Mac** (not a synthetic
stand-in), produced by `tests/batch_probe_stability.py` (pure-sim, no rendering,
~5 s). It de-risks the behavior labels before training, directly testing the
"chaos near tipping points" risk in `docs/BEHAVIOR_TASK.md`. The PUSH probe records
`toppled = final_tilt_deg > 50°` — a hard threshold.

**The sim is deterministic.** Same `(shape, material)` run 3× ⇒ max field diff
`0`. So the docs' "average over seeds" mitigation cannot work by re-running a
scene; it requires injecting initial-condition / action jitter.

**Chaos is real, sharp, and localised to the crossover** (sweeping heaviness→density
at grip=0.5, hardness=0.3). The cylinder — the docs' canonical "tall topples where
squat slides" case — flips from upright to over across a near-step transition
(slope ≈ 2160°/heaviness-unit at crossover h≈0.23). Under a **3% push-impulse
jitter** (24 reps), the binary label's flip-rate is:

| location (cylinder) | p_topple | flip-rate | final_tilt |
|---|---|---|---|
| far_light (h=0.00) | 1.00 | 0.00 | 90.0 ± 0.0° |
| **near boundary (h=0.23)** | **0.33** | **0.33** | **31.5 ± 41.3°** |
| far_heavy (h=0.53) | 0.00 | 0.00 | 0.0 ± 0.0° |

A tiny action difference flips the label ~⅓ of the time **only** at the crossover;
30% away on either side it is rock-stable (σ=0°). Chaos is a thin shell, not a fog.

**But `toppled` is degenerate for half the shape set** (multi-seed population,
5 seeds × 200 scenes, the generator's own sampler):

| shape | topple_rate (mean ± std) | label class |
|---|---|---|
| box | 0.000 ± 0.000 | **never** (always slides) |
| capsule | 1.000 ± 0.000 | **always** (rolls onto its side) |
| sphere | 0.447 ± 0.101 | **ill-posed** (no upright pose; tilt just measures rolling) |
| cylinder | 0.244 ± 0.047 | genuine boundary |
| ellipsoid | 0.062 ± 0.033 | genuine boundary |

So the topple bit carries **zero learnable signal for box/capsule** (a constant)
and is **meaningless for the sphere** (rotational symmetry → arbitrary final
orientation). It is a genuine, non-degenerate target only for the cylinder and
ellipsoid — and that is exactly where it is chaotic.

**The contamination is bounded** (fraction of all scenes within a deadband of the
50° threshold, multi-seed): `±5°: 0.017 ± 0.015`, `±10°: 0.030 ± 0.018`,
`±15°: 0.043 ± 0.021`. Only ~2–4% of sampled scenes sit in the ambiguous band;
overall topple_rate `0.357 ± 0.033`.

**Implications for the experiment** (to decide before the coherence run, not yet
applied):
- Prefer the **continuous `final_tilt_deg`** over the binary `toppled` (no
  threshold discontinuity), or replace `toppled` with a **jitter-averaged soft
  topple probability** — a well-defined, smooth target.
- `toppled`'s degeneracy means a behavior head can score "well" on it for free
  (box/capsule) or be supervised by noise (sphere). When reading per-field
  behavior MSE and `learned_coherence`, treat `push.toppled` separately — consider
  reporting coherence with and without it, and excluding the sphere from
  topple-based analysis.
- The smooth fields (settle_time, max_height, slid_distance, …) are unaffected.

Reproduce: `python tests/batch_probe_stability.py` (writes
`runs/stability/probe_stability.json` with the raw sweep arrays for plotting).

---

### F9 — ⭐ Coherence experiment: a weak, seed-UNSTABLE positive — the multi-seed run CORRECTS the single-seed pilot

This is the project's headline measurement, run properly. **The honest verdict:
training raises render↔physics coherence modestly above both baselines *on
average*, but the effect is small and swings wildly with the training seed — it is
NOT (yet) evidence of a reliable shared eigenvector.** A single-seed pilot badly
overstated it; this entry is the correction.

Setup: dataset `pm_big` — 512 scenes, 8 views, 128px, **extrapolation** holdout
(heavy+bouncy corner), 492 train / **20 test**. **5 shared seeds** (50 epochs each)
+ a **render-only** and a **physics-only** model (per-head loss weights) for the
independent control. Coherence on the held-out region (48 dirs); architectural
baseline over 5 fresh untrained inits. `scripts/run_coherence_experiment.py`.

| target | trained (5 seeds) | architectural (5 seeds) | independent (disjoint-latent) | learned = trained − arch |
|---|---|---|---|---|
| behavior | 0.276 **± 0.166** | 0.109 ± 0.029 | 0.030 | +0.168 (within seed noise) |
| essence  | 0.261 **± 0.157** | 0.096 ± 0.021 | 0.046 | +0.165 (within seed noise) |

**What holds:** the ordering **independent (~0.04) < architectural (~0.10) <
trained (~0.27 avg)** is real — the disjoint-latent ~0 control works, and trained
models sit above both baselines on average.

**What breaks the strong claim:** trained coherence is **enormously seed-unstable**
— essence ranges **0.10 → 0.49**, behavior **0.17 → 0.60** across the 5 seeds. The
trained std (~0.16) is as large as the gain. Crucially the architectural baseline's
std on the *same* 20 scenes and directions is only ~0.03, so the spread is **genuine
training-seed variance, not measurement noise**: some inits learn the coupling, some
don't. By mean ± std the learned gain is *within* the cross-seed band (t≈2.2, n=5 —
marginal, not significant).

**The correction.** The earlier provisional pilot (`pm_mid`, single seed 0) reported
essence learned **+0.369 at ~7.7σ**. That was a lucky draw: on `pm_big` the **seed-0**
model gives essence coherence **0.110** — right at the untrained baseline. Multi-seed
collapses the headline to +0.165 and reclassifies it as marginal. The pilot's other
"tell" was also atypical: its participation ratio was 61.7 (vs untrained 7.6); across
the 5 `pm_big` seeds it is **10.8 vs 8.3** — the trained models barely spread the
representation here.

**A precise sub-finding:** prediction quality is **stable** across seeds (held-out
behavior MSE 0.032–0.051, all modestly beating predict-mean) while representational
**coupling is not**. The heads *predict* consistently; whether they *share structure*
depends on initialization. Coupling of direction (when it appears) without reliable
accuracy on the extrapolation corner.

**Open:** more seeds (10–20) to settle significance; longer training / larger
held-out set; investigate the init-sensitivity (loss-landscape basins, or the F8
behavior-label noise). Reproduce: train the 7 models (`runs/big/*`), then
`python scripts/run_coherence_experiment.py --data data/pm_big --checkpoints
runs/big/shared_s0/model.safetensors,…,shared_s4/… --render-only
runs/big/render_only/model.safetensors --physics-only
runs/big/physics_only/model.safetensors` (writes `runs/big_coherence/coherence_report.json`).

**→ Settled by F10:** the 20-seed sweep resolved the significance question and
explained the instability — read F10 before quoting any F9 number.

---

### F10 — ⭐ 20-seed sweep: the mean gain is real — but "seed instability" is two basins, and the coherence metric prefers the DEGENERATE one

Setup: same data + protocol as F9 (`pm_big`, extrapolation holdout, 50 epochs,
48 dirs), extended to **20 shared seeds** (F9's s0–s4 reused, s5–s19 new), the
architectural baseline widened to **10 fresh untrained inits**, same independent
(disjoint-latent) controls. `scripts/run_coherence_experiment.py` →
`runs/big_coherence_20seed/coherence_report.json`.

| target | trained (20 seeds) | architectural (10 inits) | independent | learned = trained − arch | t (diff of means) |
|---|---|---|---|---|---|
| behavior | 0.346 ± 0.184 | 0.125 ± 0.031 | 0.030 | **+0.221** | **5.2** |
| essence  | 0.237 ± 0.150 | 0.072 ± 0.040 | 0.046 | **+0.165** | **4.6** |

**At the mean level, F9's "marginal" is corrected to "significant":** with n=20
the learned gain is well-resolved for both targets, and the F6 ordering
independent < architectural < trained is intact.

**But the per-seed spread is not noise — it is bimodal.** Held-out behavior MSE
splits the 20 seeds into two exact populations:

| basin | seeds | held-out gain over predict-mean | latent participation ratio (PR) | coherence (behavior) |
|---|---|---|---|---|
| **escaped** | 13/20 | 1.36–1.64× | 8–84 | 0.279 ± 0.167 |
| **collapsed** | 7/20 (s2,5,6,11,12,15,19) | 1.00–1.01× (exactly predict-mean) | **0.0 — all of them** | **0.469 ± 0.173** |

The collapsed basin is an **encoder collapse**: PR ≈ 0 means `z` is (numerically)
the *same vector for every scene* — the encoder ignores its input (untrained
encoders score PR ≈ 8, so training actively destroys input-dependence in this
basin). Train loss never leaves the 0.0746 plateau; the behavior head can only
predict the mean. The separation is binary — every stuck seed has PR 0.0, every
escaped seed PR ≥ 8 — so PR is a clean post-hoc basin classifier.

**The falsification test (run, negative, informative).** Escaped seeds leave the
plateau at epochs 3–48, which *looked* like a stochastic waiting time — implying
"train longer and everyone escapes." Training is deterministic per seed (init
seeded by `--seed`, shuffles seeded per-epoch), so we retrained all 7 collapsed
seeds to **150 epochs** (first 50 reproduce exactly): **0/7 escaped** (loss
0.0746 → 0.0742, flat). The basin is **selected at initialization**; longer
training does not rescue it. And at 150 epochs the collapsed models' pooled
"learned coherence" rises to **+0.288 — it "clears the band"** — while two of
them score held-out behavior MSE *worse* than predict-mean (0.0537–0.0539 vs
0.0516 baseline).

**Methodological caution (the load-bearing lesson).** Across the 20 seeds,
coherence *positively* correlates with held-out error — corr(behavior coherence,
behavior MSE) = **+0.54**. Mechanism: a constant-`z` encoder gives every scene
the same latent operating point, and whatever local render↔behavior alignment
exists at that one point is credited for the whole test set. **Raw coherence is
therefore inflated by degeneracy and must never be reported alone — pair it with
(a) held-out prediction gain-over-mean and (b) latent participation ratio. A
high-coherence model with gain ≈ 1 or PR ≈ 0 is collapsed, not coupled.**

**The honest headline, conditional on non-collapse:** among the 13 escaped seeds,
learned coherence = **+0.154 behavior (t≈3.3)** and **+0.155 essence (t≈4.2)** —
real, modest, and now statistically resolved. This replaces both F9's +0.168
"within noise" and the raw 20-seed +0.221 (which is partly collapse-inflated) as
the number to quote.

**Open:** what property of an init selects the basin (13:7 escape:collapse at
these hyperparameters); whether a behavior-weight warmup or LR schedule changes
the collapse rate; the soft-topple clean-label re-run (unchanged from F8/F9).

Reproduce: train seeds 5–19 (as F9, `--seed N --out runs/big/shared_sN`), then

```bash
python scripts/run_coherence_experiment.py --data data/pm_big \
    --checkpoints runs/big/shared_s0/model.safetensors,…,shared_s19/model.safetensors \
    --render-only runs/big/render_only/model.safetensors \
    --physics-only runs/big/physics_only/model.safetensors \
    --untrained-seeds 10 --out runs/big_coherence_20seed
# escape test: retrain s2,5,6,11,12,15,19 with --epochs 150 --out runs/big/shared_sN_e150,
# then the same runner over those checkpoints -> runs/big_coherence_e150
```

---

### F11 — ⭐ LLM world-model transfer test: the reasoning *format* transfers, the contact physics doesn't — and its topple call is better-calibrated than our own trained model's

An external "language world model" — **Qwen-AgentWorld-35B-A3B** (Alibaba,
arXiv:2606.24597; community Q8 MLX quant, text-only — Qwen shipped no vision
tensors) — was scored against this project's exact MuJoCo ground truth with
`scripts/eval_llm_transfer.py`, served locally (oMLX, M5 Pro). Design point:
AgentWorld's CPT→SFT→RL tuning covers seven **digital** domains (MCP, Search,
Terminal, SWE, Android, Web, OS); rigid-body physics is **out-of-domain** for
that tuning, so this measures whether next-state-prediction *skill* transfers
to a physical substrate — its physics content can only come from base-model
pretraining.

Setup: all 20 held-out `pm_big` scenes (the heavy+bouncy **extrapolation
corner**) × 3 probes × 2 information conditions; temperature 0; scored with
the behavior-head normalizers against the predict-train-mean baseline. The
model reasons long (typically ~17k-token chains, ~4.5 min/request at ~64
tok/s; 4k and 16k budgets truncate mid-derivation — 32k needed). **Zero parse
failures in 120/120 responses.**

| condition | normalized MSE | baseline | gain | push.toppled Brier | fields beating baseline |
|---|---|---|---|---|---|
| essence (true density/friction/restitution given) | 1.277 | 0.052 | 0.04× | **0.100** | **8/21** |
| appearance (rendering params only) | 0.789 | 0.052 | 0.07× | 0.150 | 4/21 |

**Aggregate: ~15–25× worse than predict-the-mean** — but the aggregate is
outlier-dominated and the decomposition is the finding:

1. **The failures are one coherent mistake, not noise.** Nearly all of the MSE
   is ramp-contact: `tilt.slid_distance` 18.70 vs baseline 0.013 (~1400×),
   tilt path_length/n_bounces. The model *derives*, at length, objects
   bouncing and rolling down the 20° incline; MuJoCo's high-friction objects
   grip and stay put. Its bounce counts are ideal restitution-series counts,
   where the simulator's velocity-threshold counter reports few — partly a
   measurement-convention mismatch, partly a different contact world.
2. **The wins are real and structured (essence, 8/21 fields):** `max_height`
   on all three probes (2–4× better than baseline), drop settle_time and
   path_length, push final_tilt_deg — ballistics and geometry. And
   **push.toppled Brier 0.100 vs 0.199 base-rate**: a *calibrated* topple
   judgment on exactly the label F8 showed is chaotic and our regression head
   handles worst.
3. **The essence→appearance ablation shows it actually uses the numbers.**
   Hiding the physics parameters shrinks the contact blow-ups (tilt MSE 3.54
   → 2.08 — it hedges toward typical materials) but collapses the ballistic
   wins (8 → 4 fields; drop settle/path/max_height and push final_tilt all
   lost). Quantitative inputs measurably flow through its derivations.
4. **Against our trained model:** the escaped-basin trained networks score
   held-out behavior MSE 0.033–0.051 (gain 1.36–1.64×) on the same scenes —
   ~15–40× better than the LLM in aggregate. 492 in-world scenes beat a 35B
   out-of-world prior. But the LLM holds two things the small model lacks:
   calibrated topple probability, and per-field wins with **zero** training
   scenes.

**Honest verdict:** "world modeling" transfers as *form* — hold a state, apply
an action, derive the next state, with usable uncertainty on discrete events —
while the *content* (MuJoCo's contact regime) does not. A language world model
is a prior over worlds, and it reasons confidently in a nearby-but-different
one.

Caveats: one model, one community quant, one prompt format; normalized MSE is
outlier-dominated (hence per-field reporting); some "errors" are convention
mismatches with our summarizers (n_bounces counter, 50° threshold); the
appearance condition *describes* rendering parameters in text — the true
picture→physics condition needs a vision-capable AgentWorld artifact, and the
only one that exists is an unvalidated community graft (see next steps).

Reproduce: serve the model (oMLX, OpenAI-compatible), then per condition::

    python scripts/eval_llm_transfer.py --data data/pm_big --split test \
        --base-url http://127.0.0.1:8000/v1 --model <served-name> \
        --max-tokens 32768 --condition essence --out runs/llm_transfer_essence
    # and --condition appearance -> runs/llm_transfer_appearance
    # responses cached per (scene, probe); reports: transfer_report.json

## 3. What is NOT yet known (honest gaps)

- **The learned coupling is now resolved but modest.** F10 (20 seeds): among
  non-collapsed seeds, `learned_coherence` ≈ **+0.15** on both targets (t≈3–4).
  Real, but small — and still coupling of the *generator's* authored structure
  (see below), measured on one dataset scale.
- **Why initialization selects the training basin is unexplained.** 7/20 inits
  fall into an encoder-collapse basin that longer training does not rescue
  (F10's falsification test). What distinguishes those inits — and whether a
  behavior-weight warmup, LR schedule, or the F8 label noise changes the 13:7
  rate — is open. Prediction quality *within* the escaped basin is stable;
  basin membership is the lottery.
- **The coupling is authored.** MuJoCo/Blender decouple appearance and physics, so
  we are (at best) learning the *generator's* eigenvector, not reality's. The GSO
  experiment (`docs/GSO_EXPERIMENT.md`) is the parked route to real measured data.
- **The clean-label coherence run hasn't been done.** `push.toppled` is degenerate
  for box/capsule and ill-posed for the sphere (F8); the jitter-averaged soft-topple
  label exists (`--topple-jitter-reps`) but hasn't been pushed through a full
  coherence experiment. The smooth fields (settle_time, max_height, …) are
  unaffected.

---

## 4. Next steps — the result is in; now harden it

F8 (label stability), F9 (5-seed coherence), F10 (20-seed resolution + basin
falsification test), and F11 (LLM world-model transfer, both text conditions)
have all been run on the Mac (MLX/Metal). What remains:

1. **Explain basin selection** — what distinguishes the 7/20 collapsing inits;
   try a behavior-weight warmup or LR schedule and measure whether the collapse
   rate moves (F10).
2. **Re-run with cleaner labels** — the soft-topple probability
   (`--topple-jitter-reps`) and/or `push.toppled` excluded and the sphere dropped
   (F8) — to see whether less label noise widens the escaped basin or tightens
   its coherence spread.
3. **GSO** — real measured objects, to test reality's coupling rather than the
   generator's (`docs/GSO_EXPERIMENT.md`).
4. **F11 vision condition (optional follow-up)** — show the model our actual
   renders instead of text-described appearance params (the picture→physics
   inverse proper). Blocked on a vision-capable AgentWorld artifact: the only
   one is `havok2/Qwen-AgentWorld-35B-A3B-VL36`, an unvalidated cross-generation
   weight graft (Qwen3.6 vision tower on the Qwen3.5 backbone, no co-training;
   70 GB BF16 → needs local 4–6-bit mlx-vlm conversion to fit 64 GB). Gate on
   sanity-checking its vision on trivial images first — the graft is a built-in
   confound.

Reproduce F10: see the F10 entry. Reproduce F11: see the F11 entry.
Reproduce F8: `python tests/batch_probe_stability.py`.

---

*Tests: 156 across 22 suites, all passing; core imports with no
mujoco/bpy/trimesh/numpy/mlx/torch. Personal research; not affiliated with World
Labs.*
