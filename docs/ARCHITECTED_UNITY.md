# F25 preregistration — architected unity in a strong world: can constraint buy the coherence the signal never did?

*Preregistered 2026-08-02, before any F25 arm was trained or measured. Decision
rules are committed as `scripts/f25_verdicts.py` in the same commit as this
document. Implementation: a gated coherence objective in the training loss
(`coherence_weight`, default 0.0 — architecture and defaults byte-identical,
the F20 pattern) plus the existing FSQ bottleneck; no new parameters anywhere.*

---

## 1. Why this experiment exists — the fork, resolved into a design

Three experiments now agree that **signal alone does not organize the latent**:

- **F22a**: strengthen the coupling and the essence is *predicted* (Δ +0.15 →
  +0.92, ~86 % of reachable at r ≈ 0.64).
- **F23 (H1 NULL)**: the same strengthening leaves render↔behavior *coherence*
  flat — prediction improves through pathways that do not converge.
- **F24**: it leaves cross-seed *convergence* flat too, and below the untrained
  floor at r ≤ 0.64 — while a 1-trit budget forces 90–95 % cross-model
  agreement at every coupling. Scarcity manufactures coherence (F17) and
  consensus (F24); abundance individuates.

Option B was "build a stronger-coupled world so a learnable essence exists."
F22 built those worlds; F23/F24 showed the missing Marble-property is not in
the world at any coupling. So Option B, sharpened (FINDINGS §4 item 5): **use
the strong world for signal, and architect the unity on top** — then measure
whether the architected model is *actually* more Marble-like, on all three
axes at once: prediction (Δ), unity (coherence), consensus (alignment).

**Vault review (2026-08-02, `ai-research` wiki), folded into the design:**

- *sigreg*: SIGReg's isotropic-Gaussian prior **fails on low-diversity data**
  — a 512-scene world is that regime, so the organizing pressure here is
  task-anchored (a coherence objective over the model's own two heads), not
  distribution-shaped. What we do keep from SIGReg is the design standard:
  one regularizer, one swept hyperparameter.
- *representation-collapse* (+ steam-engine Phases 3–4): any new organizing
  loss invites **partial collapse toward the easier structure**; the
  observation-anchored render/behavior heads are the anti-collapse anchor
  (steam-engine's "parallel raw-prediction head" mechanism), and the PR gate
  (F10/F12) sits on every arm.
- *small-state-thesis*: the FSQ k-ladder is the thesis's "axes compose" cell
  (bit-width × latent dimensionality), and its open question — *does the
  structure survive the cut, or only the benchmark score* — is why every arm
  gets all three readouts, not just Δ.

## 2. The world and the arms

**World: `data/pm_f22_g2` (r = 0.809).** Exceeds the r ≈ 0.6 target per
FINDINGS §4 item 1; avoids the r ≈ 0.64 point that F22a's peak test showed is
the least stable in the study. Its continuous baseline **already exists**:
`runs/f22/g2_cont_s0..7` (8 seeds; F22a measured Δ = +0.650 ± 0.232, n = 8).

New arms, all `--data data/pm_f22_g2 --lr 5e-4 --epochs 50 --image-size 128`:

| arm | mechanism | seeds | runs out |
|---|---|---|---|
| cont | none (baseline) | 8 (exist) | `runs/f22/g2_cont_s*` |
| fsq k ∈ {1, 2, 4, 8} | FSQ bottleneck (`--latent-trits k`) | 3 each | `runs/f25/g2_fsq{k}_s*` |
| cohw w ∈ {0.3, 1.0, 3.0} | coherence objective (`--coherence-weight w`) | 3 each | `runs/f25/g2_cohw{tag}_s*` |
| controls | render-only / behavior-only (masked loss, F23 pattern) | 1 each | `runs/f25/g2_{render,behav}only` |

FSQ × coherence-objective composition is **deferred** — this study isolates
each mechanism against the same baseline.

## 3. The coherence objective, exactly

Gated on `cfg.coherence_weight > 0` (default 0.0: the default model, its
checkpoints, and the suite are byte-identical — old checkpoints that stored
the reserved value 1.0 are unaffected because the term adds no parameters and
exists only inside training). Per training step:

1. Sample D = 4 directions u_d ~ N(0, I_256), normalized, fresh each step.
2. Perturb every sample's latent: z + ε·u_d with ε = 0.1.
3. Response magnitudes per (sample, direction): mean |Δrender| over pixels,
   mean |Δbehavior| over the 21 outcomes.
4. Loss term: `coherence_weight × (1 − Pearson(r_mag, b_mag))` over the
   B×D = 128 (sample, direction) points.

This is a differentiable, training-time analogue of the F6/F13 measurement
(`models/coherence.py`: correlation of response magnitudes across latent
directions). Mirrors: MLX (trains), torch (CPU smoke), and a pure-Python
reference of the correlation math in `models/losses.py`, unit-tested without
any framework.

**Goodhart disclosure, fixed before any run:** the cohw arms are trained on
(a stochastic analogue of) the metric they will be measured with. Their
coherence readout is therefore **engineering evidence, not discovery** — "the
objective moved its own metric" is expected. What makes a cohw arm
interesting is everything it must NOT break while doing so: Δ, PR, the
essence-coherence co-readout, and (H2) whether consensus rises. The FSQ arms
carry no such circularity: nothing in their objective mentions coherence.

## 4. Measurements — the full triad on every arm

1. **Δ** (prediction): held-out gain − shape-only oracle on the same corner
   split, exactly the F22 helpers (`scripts/f22_pilot_eval.py` conventions).
2. **Learned coherence** (unity): `scripts/run_coherence_experiment.py` per
   arm — behavior target headline, essence target co-reported, 5 untrained
   inits, test split, with the F23 control gate (disjoint render-only /
   behavior-only baseline must sit within |coh| < 0.05) and the F12 collapse
   gate (PR ≥ 8).
3. **Consensus**:
   - cont and cohw arms: within-arm learned kNN alignment (F24 harness — k=10,
     512 scenes, minus an 8-init untrained baseline on `pm_f22_g2`).
   - FSQ arms: **adjusted Rand index** between seeds' code partitions (each
     distinct code vector = one cluster). ARI is permutation-invariant by
     construction and chance-adjusted to ≈ 0, so it generalizes F24's
     max-over-relabelings statistic (which is factorially infeasible past
     k = 1) to every k. For k = 1 the F24 trit-agreement statistic is also
     reported for direct comparability.

## 5. Hypotheses and frozen decision rules

Reference values measured in-run from the cont arm (n = 8):
`coh_cont` (learned behavior coherence), `D_cont` (Δ), `align_cont`
(learned kNN).

**H1a — the FSQ frontier (non-circular).** Per k:

    UNIFIED-WITHOUT-COST(k)  iff  learned_coh(k) − coh_cont >= +0.10
                                  with Welch t >= 2.5 (per-seed units)
                             AND  Δ(k) >= D_cont − 0.15

H1a = SUCCESS if any k qualifies; FAILURE if all four are measured cleanly
and none does; INCONCLUSIVE otherwise (gate failures, missing runs).

**H1b — the coherence objective (circular; engineering).** Per w:

    ENGINEERING-SUCCESS(w)  iff  learned_coh(w) − coh_cont >= +0.10 (t >= 2.5)
                            AND  Δ(w) >= D_cont − 0.15
                            AND  min seed PR >= 8

**H2 — does architected unity buy consensus?** The F24 pattern predicts
constraint → consensus. Two independent readouts:

    H2-OBJECTIVE  cohw arms' learned kNN − align_cont >= +0.03, Welch t >= 2.5
                  (per-seed units, n = 3 vs 8; low power acknowledged)
    H2-BUDGET     ARI(k) > 0.5 for k = 1 (replicating F24 H3 at a new r,
                  in generalized form)

**H3 — dose-response of the budget.** Across k ∈ {1, 2, 4, 8}: coherence
falls and Δ rises monotonically (Spearman sign over the 4 points; descriptive,
no t). ARI falls monotonically with k — consensus decays as the budget
loosens. This is the direct test of "constraint manufactures consensus" as a
*graded law* rather than a two-point contrast.

**Independence:** per the F23/F24 rule, no verdict supports another; H1b's
circularity note travels with any citation of its result.

## 6. Predictions — frozen before any training

- **P1 (confident):** fsq k=1 clears the coherence bar (+0.10, replicating
  F17's doubling at a new coupling) but **fails the Δ condition** — F22a P4
  showed the 1-trit budget costs prediction badly at strong coupling.
- **P2 (moderate — the engineering bet):** at least one of k ∈ {2, 4, 8}
  reaches UNIFIED-WITHOUT-COST. Best guess k = 2 or 4: F17 found behavior
  saturates at ~1 trit in the weak world, but the strong world's material
  channel needs more bits; 3–6 trits (≈ 4.8–9.5 bits) should carry it while
  still being scarce enough to force sharing.
- **P3 (moderate):** cohw 0.3 and/or 1.0 reach ENGINEERING-SUCCESS; w = 3.0
  fails it by dragging Δ or PR (over-regularization — the steam-engine
  partial-collapse warning made quantitative).
- **P4 (moderate — the scientifically novel one):** H2-OBJECTIVE passes —
  the coherence objective raises cross-seed consensus even though nothing in
  it references other models. If it does, "scarcity manufactures consensus"
  generalizes to "constraint manufactures consensus," which is the claim F24
  could only make for budgets.
- **P5 (confident):** H2-BUDGET passes (ARI(k=1) > 0.5).
- **P6 (moderate):** H3's monotone pattern holds in all three series
  (coherence ↓, Δ ↑, ARI ↓ in k), with any single inversion at adjacent k
  treated as noise if within 1 sd.

## 7. Cost

23 training runs on the M5 Pro: 12 FSQ (~87 s each), 9 cohw (~3× slower for
the D = 4 extra decoder passes, ~4.5 min each), 2 masked controls — about an
hour of training. Measurement reuses the F22/F23/F24 harnesses (~20 min).

## 8. Honest limits

- n = 3 seeds per new arm: enough for the frozen t-tests only when effects
  are large; every marginal result will say "n = 3" out loud. The cont
  reference has n = 8.
- The cohw coherence readout is circular by construction (§3); only its
  *costs* and its consensus effect are discovery-grade.
- ε and D in the objective are fixed (0.1, 4), not swept — a null for cohw
  is a null for *this* instantiation, not for coherence objectives generally.
- ARI across FSQ arms compares partitions of different granularity (3^k
  cells); ARI is chance-adjusted for that, but the k-series comparison
  (H3) is descriptive, not inferential.
- The strong world is synthetic with an authored coupling; nothing here
  speaks to real objects (ABO stage 2 remains blocked by F22b).
