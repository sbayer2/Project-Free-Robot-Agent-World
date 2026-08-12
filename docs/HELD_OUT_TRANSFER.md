# F29 preregistration — held-out-transfer coherence: does joint training carry operational physical essence that disjoint training doesn't, measured as prediction advantage on materials neither student saw?

*Preregistered 2026-08-11, before any held-out prediction has been computed
with any student on any held-out material. Decision rules are committed as
`scripts/f29_verdicts.py` in the same commit as this document. The
extrapolation region is set by a **pilot** whose procedure is frozen here
and whose result is pinned in Amendment 1 (committed after the pilot runs,
before any main-experiment measurement is taken). F30 and F31 remain
registered-but-undesigned (F28 Amendment 2); F32 is parked
(executive-stack pilot went DESIGN VOID at first contact; branch
`claude/f32-executive-stack-prereg` kept as work-in-progress record); F29
takes the next slot.*

---

## 1. Why this experiment exists

F27b showed the magnitude-coherence metric certifies substrate entanglement,
not content unity. F28's directional metric, verified free of that
entanglement (H1 CLEAN), read **zero** at every coupling — no internal
geometric signature of "one latent, two projections" survives at this scale
even where content is near-definitional and demonstrably extracted (loud:
metallic ≈ density at r 0.992, probe R² ≥ 0.94, essence gain 5.67). The
registered next question (F28 Amendment 2, motivated by [[dyna-2]]): **if
unity has no internal signature, does it have an *operational* signature?**
Does joint training on a shared latent produce transferable structure that
two disjoint (independently trained) single-head students don't?

Dyna-2 is the external existence proof this experiment inherits: joint
training on a shared trunk *can* carry unified essence across a task
boundary — but the payoff *opens with scale* and is invisible at low data
(their controlled ablation: joint-alone doesn't scale; only joint with
extra content-bearing video does, and the gap opens above the 10k–100k
hour threshold). That framing is the honest lens for F29's outcome
readings: **a null at our scale bounds unity below the apparatus, per
Dyna-2's own low-scale ablation; it does not refute unity in the
universe.** This design pre-commits to that reading.

## 2. The metric

For each world W in the coupling ladder, each student S ∈ {joint, disjoint},
on a held-out material extrapolation region H (defined by the pilot; §3):

    held_out_gain(S, W) = MSE(H, train_mean(W)) / MSE(H, predictions(S, W))

where `train_mean(W)` is the per-behavior-field mean over W's training
scenes (F27b's default predict-the-mean baseline), and `predictions` come
from student S applied to H's scenes. Gain is self-normalizing so no
untrained-baseline subtraction is needed (both students are trained; the
comparison is fair).

    advantage(W) = mean_seeds(held_out_gain(joint, W))
                 − mean_pairs(held_out_gain(disjoint, W))

Positive = joint transfers better = unity has an operational signature.
Null = joint and disjoint transfer equally = no operational advantage at
this scale (scale-bound reading). Negative = disjoint transfers better =
joint OVERFITS to training substrate. Per-seed and per-pair values are
reported alongside the means so Welch t is computable at the sample level.

## 3. The extrapolation region — set by a PILOT whose procedure is frozen here

The held-out region H is a subset of material space no training scene
occupies. Its **aggression** is set by a pilot before the main measurement;
Amendment 1 pins the choice.

**Aggression α (frozen definition):** materials in the top-α tail of joint
(density × restitution) as observed in the corresponding f27b arm's
training set, rejection-sampled to fall strictly outside the training
convex hull in normalized (density, friction, restitution) space.

**Pilot procedure (frozen; may not be altered by Amendment 1):**

1. Generate three small held-out datasets `data/pm_f29_pilot_a{5,20,40}`,
   each 128 scenes per arm, at α ∈ {0.05, 0.20, 0.40}.
2. On each pilot dataset, measure held_out_gain(joint, loud) using existing
   `runs/f27b/loud_s{0,1,2}`, per-seed then mean.
3. On each pilot dataset, measure held_out_gain(disjoint, loud) using
   newly-retrained `runs/f29_disjoint/loud_pair{0,1,2}` (§4), per-pair
   then mean.
4. Per student, per α: **drop_ratio(S, α)** = 1 − held_out_gain(S, loud, α)
   / train_gain(S, loud), where train_gain is that student's gain measured
   on f27b's training test split (existing for joint; measured during the
   retrain-sanity check for disjoint).
5. **combined_drop(α)** = mean(drop_ratio(joint, α), drop_ratio(disjoint, α)).
6. **α\*** = argmin over the three tested values of |combined_drop(α) − 0.5|.

If combined_drop(α\*) falls outside [0.25, 0.75], Amendment 1 discloses the
miss explicitly and either continues with α\* or adds one additional α
between the closest tested points; no silent re-tuning.

**Amendment 1 pins:** α\*, combined_drop(α) for every tested α, disjoint
train-world gains (§7 retrain-sanity), and the coupling-r values measured
on each pilot dataset. Written before any main-experiment measurement.

## 4. Inputs — existing and to-be-created

- **Joint checkpoints (existing):** `runs/f27b/{ctrl,base,g2,loud}_s{0,1,2}/model.safetensors`.
- **Disjoint checkpoints (TO BE RETRAINED):**
  `runs/f29_disjoint/{ctrl,base,g2,loud}_pair{0,1,2}/{render,behav}.safetensors`
  — 24 retrains at the **frozen F27b recipe**: lr 2e-4, 50 epochs, batch 32,
  image_size 128, seeds 0/1/2 shared with joint arms so init is comparable.
  Render pair: `--behavior-weight 0 --essence-weight 0`. Behav pair:
  `--render-weight 0`. ~35 min total on the M5 Pro at f27b timings.
- **Training datasets (existing):** `data/pm_f27_{ctrl,base,g2,loud}`.
- **Pilot datasets (to be generated):** `data/pm_f29_pilot_a{5,20,40}`,
  128 scenes × 4 arms each. Same MuJoCo pipeline as f27b with a
  material-rejection filter enforcing the α-tail cut.
- **Main held-out dataset (to be generated at α\*, after Amendment 1):**
  `data/pm_f29_holdout`, 128 scenes × 4 arms.

## 5. Measurements

1. **Retraining sanity** (per arm, per pair): train-world gain of each
   disjoint pair's `behav` model on f27b's training test split, using the
   same measurement code as `scripts/f27_measure.py`. Reported per pair;
   min across pairs feeds §7's `retraining` gate.

2. **Apparatus check** (per arm, per dataset): coupling r on the held-out
   scenes must match the training arm's r within `R_MATCH_TOL = 0.05`.
   Guards against physics-correlation drift in the extrapolation tail.

3. **Extrapolation check** (per arm, per dataset): 100% of held-out
   material vectors lie outside the training arm's convex hull in
   normalized (density, friction, restitution) space.

4. **Held-out gain per student per arm (primary):** encode held-out images
   through each student, run behavior head, compute held_out_gain per seed
   (joint) or per pair (disjoint), 3 each per arm.

5. **Advantage curve:** `advantage(W)` at each arm on the ladder.

6. **F28 tie-in prep (H4):** if H2 reads POSITIVE at any arm, per-arm
   directional-coherence values on the held-out dataset will be computed
   via a subsequent amendment and driver commit. NOT measured in the main
   F29 run; the fork is registered here so the answer isn't back-fittable.

## 6. Hypotheses and frozen decision rules

All Welch t computed from summary stats (mean, sd, n) across seeds/pairs;
per-seed and per-pair gains reported alongside the means.

**H0 — kill switch (loud arm only):**

    PASS   min(gain(joint, loud, seed_i)) >= 1.5   AND
           min(gain(disjoint, loud, pair_j)) >= 1.5
    FAIL   otherwise -> verdicts withheld, finding recorded as
           "below apparatus threshold for measuring transfer at this scale"
           (the scale-bound reading Dyna-2's low-scale ablation supports).

**H2 — primary contrast (loud, per-arm co-report):**

    POSITIVE     advantage(loud) >= +0.15  AND  |Welch t| >= 2.5
    NEGATIVE     advantage(loud) <= -0.15  AND  |Welch t| >= 2.5
    NULL         |advantage(loud)| < 0.05
    INCONCLUSIVE otherwise

Frozen interpretations:
- **POSITIVE:** unity has an *operational* signature at this scale — joint
  training carries transferable structure disjoint doesn't. Triggers H4
  tie-in.
- **NULL:** no operational advantage at this scale, consistent with F28
  zero and Dyna-2's 5k-hour ablation. Bounds unity below the apparatus,
  not death of unity.
- **NEGATIVE:** joint OVERFITS to training substrate; disjoint specialists
  transfer better. F27b's substrate entanglement is actively harmful, not
  just cosmetic.

**H3 — advantage curve across the ladder:**

    POSITIVE-COUPLING  Spearman(advantage, r) >= +0.8  AND
                       (advantage(loud) - advantage(ctrl)) >= +0.15
    NULL               |advantage(loud) - advantage(ctrl)| < 0.05
    INCONCLUSIVE       otherwise

**H4 — F28 tie-in (registered fork; verdict withheld here):** if H2
reads POSITIVE at any arm, a subsequent amendment commits a driver to
rerun F28's directional metric on the held-out dataset for that arm. Two
frozen readings:

- **F28 zero holds on held-out too** → joint's transferable structure
  lives outside global linear direction alignment. Publishable finding:
  *"unity is operational without being directional"* — extends F28's zero
  with a positive footprint.
- **F28 zero fails on held-out** → F28's zero was regime-specific. F28
  walk-back committed with equal prominence to its original entry.

**H4 alt (if H2 = NULL everywhere):** no tie-in triggered; F28's zero
and F29's null are jointly consistent with the scale-bound reading. This
alt is the modal expected path (P1).

## 7. Gates — all must pass before any verdict is read

    retraining      per arm: min disjoint train-world gain >= 1.5 across
                    all 3 pairs. Failure -> the failing pair is regenerated
                    ONCE with a different seed; systematic failure across
                    arms -> a driver commit questions the recipe (not
                    silent retry).
    apparatus       per arm: |r_held_out - r_train| < 0.05.
    extrapolation   per arm: 100% of held-out material vectors outside
                    training convex hull. Failure -> dataset regenerated
                    with tighter rejection sampler.
    collapse        per arm: PR >= 8 on held-out for every joint seed and
                    every disjoint pair (behav side). F28's floor.
    pilot           Amendment 1 committed with alpha* pinned and pilot
                    numbers disclosed BEFORE any main measurement runs.

## 8. Predictions — frozen before any data

- **P1 (60/40):** H2 → NULL at every arm. Consistent with F28 zero AND
  Dyna-2's low-scale ablation. Modal prediction; the odds sit at 60, not
  65, because Dyna-2's existence proof genuinely opens the door — a
  positive signal *is* achievable, we're just betting our scale isn't
  above the threshold.
- **P2 (25/75):** H2 → POSITIVE at loud. Registered because Dyna-2 shows
  unity IS achievable at scale, and our loud is the closest analog to
  Dyna-2's high-signal regime (strongest coupling, largest prediction
  gain in the program's history). Triggering P2 also triggers H4's
  tie-in fork.
- **P3 (15/85):** H2 → NEGATIVE at ctrl. If joint mostly glues heads
  together, at zero coupling that gluing is pure substrate overhead;
  disjoint specialists could beat joint on transfer specifically in
  weak-coupling worlds.
- **P4 (75/25):** H0 → PASS at loud. Held-out gain ≥ 1.5 for both
  students is plausible given f27b's loud joint gain 5.67 on training;
  registered so a failure here is a finding about the extrapolation
  region choice, not a silent restart.

## 9. Cost

Measurement-heavy but no scale ramp; phased on the M5 Pro:

- **Disjoint retrain** (24 pairs × ~87s at f27b timings): ~35 min MLX.
- **Pilot datasets** (3 datasets × 4 arms × 128 scenes): ~45 min MuJoCo.
- **Pilot measurement** (encode+predict on 3 pilot sets, both students,
  loud only): ~15 min MLX.
- **Amendment 1** (documentation, decision): ~10 min human time.
- **Main held-out dataset** (128 scenes × 4 arms at α\*): ~15 min MuJoCo.
- **Main measurement** (encode+predict on 4 arms × 6 students):
  ~20 min MLX.

Total: **~3 hr wall-clock**, no LLM, no unified-memory contention.

## 10. Honest limits

- One extrapolation axis (material). Held-out geometry stays box;
  held-out probes stay {drop, tilt, push} with fixed parameters. A null
  here doesn't refute unity along other extrapolation axes.
- The pilot criterion (50% combined drop) is arbitrary but honest. A
  different criterion could pin a different α\*; the criterion is
  disclosed in advance, not principled from first principles.
- n = 3 joint seeds, n = 3 disjoint pairs. Small; only large advantage
  effects clear t ≥ 2.5. Program-standard since F27.
- The F28 tie-in (H4) is a registered fork, not a fully-designed
  experiment. Its own preregistration follows if triggered.
- Scale-bound reading depends on Dyna-2's low-scale ablation being
  trustworthy. Dyna-2 is a single-lab tech report; the reading imports
  its uncertainty. If Dyna-2 fails to replicate, F29's null becomes
  harder to interpret.
- MLX 0.31.2 GPU matmul carries ~1e-3 relative error (F28 finding).
  Negligible against the ±0.15 gain bar.

## Amendment 1 (2026-08-12, post-pilot, pre-main-measurement) — α\* = 0.40 pinned; band MISSED low; limits on inference frozen

*Written after the §3 pilot ran (user-approved 2026-08-12) and before any
main-experiment measurement. The §3 procedure was followed exactly; this
amendment pins its outputs and discloses everything the procedure
required. Pilot artifacts: `runs/f29/pilot_report.json`,
`data/pm_f29_pilot_a{5,20,40}_{ctrl,base,g2,loud}` (12 datasets, 128
scenes each), `runs/f29_pilot_gen.log`, `runs/f29_pilot_measure.log`.*

**1. The pilot table (loud arm, per §3.2–3.5; gains = mean over 3
seeds/pairs; train anchors: joint 5.672 per-seed [5.040, 5.910, 6.067]
from `runs/f27b/f27_report.json`, disjoint 5.692 per-pair [5.222, 5.887,
5.967] measured per §3.4 — retrain sanity ≥ 1.5 satisfied at loud):**

| α | joint held | disjoint held | joint drop | disjoint drop | combined |
|---|---|---|---|---|---|
| 0.05 | 5.327 | 5.413 | 0.061 | 0.049 | 0.055 |
| 0.20 | 5.607 | 5.713 | 0.011 | −0.004 | 0.004 |
| 0.40 | 5.010 | 5.053 | 0.117 | 0.112 | 0.114 |

**α\* = 0.40** by the frozen §3.6 rule (argmin |combined − 0.5|).

**2. Band MISSED, disclosed, continue-with-α\* chosen (user decision
2026-08-12).** combined_drop(α\*) = 0.114 < 0.25. Of the two frozen
options, "add one α between the closest tested points" is arithmetically
unreachable — every interval drop lies in [0.004, 0.114], below the band
floor — so the run continues at α\* with this disclosure. Extending the
grid beyond 0.40 is not among the frozen options and was not taken.

**3. Structural diagnosis of the miss (recorded, not excused).** §3.4's
train-gain anchor is the f27 datasets' test split, which is
`EXTRAPOLATION_REGION_HOLDOUT` — the heavy+bouncy corner (density
[0.55, 1.01] × restitution [0.6, 0.9]). The anchor was therefore already
an extrapolation measurement, and the pilot's top-product tail is that
corner's neighborhood: small drops are structural. Independently
verified by the monitoring session against the dataset manifests. The
α = 0.20 non-monotonicity (less drop than α = 0.05) is sampling noise
at n = 3 seeds over 128-scene sets with different draws per α.

**4. LIMIT ON INFERENCE, frozen before the main run:** at drop ≈ 0.11
both students sit near their training-world ceiling on the held-out set,
compressing the room for joint–disjoint differentiation. **A main-run
H2 NULL at α\* = 0.40 is therefore weaker evidence against operational
unity than a NULL at drop ≈ 0.5 would have been**, and the F29 FINDINGS
entry must carry this sentence. A POSITIVE or NEGATIVE that clears the
±0.15 bar despite compression is, conversely, mildly strengthened.

**5. Pilot information leak, disclosed:** the pilot necessarily revealed
loud-arm held-out gains for both students; their difference (−0.04 to
−0.11, inside every frozen bar) previews P1's NULL shape at loud. The
H2/H3 rules were frozen before any of it, the main run measures all four
arms, and the main dataset uses a fresh seed (below), so no pilot scene
is reused.

**6. Main dataset, pinned:** `data/pm_f29_holdout_{arm}` (the per-arm
suffix refines §4's family name the same way the pilot datasets did —
`PseudoMarbleDataset` expects one directory per arm), 128 scenes × 4
arms at α\* = 0.40, **seed 2941** (fresh; the pilot's α = 0.40 seed was
2940, and reusing it would duplicate `pm_f29_pilot_a40_*` scene-for-
scene, importing the leak of §5 into the main measurement). Same frozen
filter, same matched-arms construction and asserts.

**7. Apparatus, carried:** r_held +0.997 / +0.996 / +0.995 at
α = 0.05 / 0.20 / 0.40 versus r_train +0.992 — inside the §5.2 gate;
outside-hull fraction 100% at every α; acceptance rates 5.2% / 6.3% /
9.1%, matching the generator's pre-registered dry-run exactly.

## Amendment 2 (2026-08-12, post-main-measurement) — two section-7 gates recalibrated; verdict bars untouched

*Written after the main run's verdicts were WITHHELD by the frozen gates
(user-approved recalibration). This amendment changes ONLY the two gate
constants shown below, inside `scripts/f29_verdicts.py`, in the same
commit as this text — the F28-Amendment-3 mechanism. The H0/H2/H3
verdict bars, the metric, and every measurement are untouched; the
rerun is verdicts-only on the already-written report.*

**1. What tripped, and why both trips are freeze-time gate errors:**

- **Retraining gate** (`disjoint_train_gain_min >= 1.5` at every arm)
  tripped at ctrl (1.064) and base (1.053). The floor is unachievable
  BY CONSTRUCTION at weak coupling: the JOINT students' own train
  gains there are 1.121 and 1.113 — frozen as `TRAIN_GAIN_JOINT` in
  the same file, three constants above the gate. §7 extended H0's
  loud-only 1.5 floor to all arms in contradiction of reference values
  the file itself carries. The retrains are healthy: disjoint matches
  joint capability at every arm.
- **Apparatus gate** (`|r_held − r_train| < 0.05` flat) tripped at ctrl
  (0.083) and g2 (0.053). The sd of a sample correlation is
  ≈ (1 − r²)/√n; at n = 128 and r ≈ 0 that is 0.088, so ctrl's 0.083
  is ~1σ of sampling noise, and g2's 0.053 is ~1.7σ. The flat bar
  ignored estimator variance; the pilot passed it only because loud's
  near-1 correlation has almost none.

**2. Recalibrated gates (frozen here, applied to the verdicts script in
this commit):**

    retraining   disjoint_train_gain_min >= 0.85 × TRAIN_GAIN_JOINT[arm]
                 (arm-relative: catches a broken retrain — wrong
                 checkpoint, failed convergence — without demanding the
                 structurally impossible; a pair that lost ≥ 15% of the
                 joint students' capability still fails)
    apparatus    |r_held − r_train| < max(0.05, 2 · (1 − r_train²)/√128)
                 (variance-aware 2σ bar with the old 0.05 as floor;
                 √128 is N_SCENES_PER_ARM, frozen in §3)

**Robustness of the recalibration (defusing post-hoc tuning):** the
observed disjoint/joint train-gain ratios are 0.949 / 0.946 / 0.886 /
0.921 (ctrl/base/g2/loud), so ANY retraining factor in [0.5, 0.88]
yields the identical all-pass gate outcome on this data; 0.85 sits
below the minimum observed ratio with margin and was not tuned to a
boundary. On the apparatus side the per-arm deviations in estimator-σ units are
0.94σ / 0.36σ / 1.73σ / 2.19σ (ctrl/base/g2/loud) — and notably
**loud, the LARGEST σ-deviation of the four, PASSED the old flat gate**
while ctrl's 0.94σ failed it:
the flat bar's strictness was r-dependent noise, not protection, and
no arm is being selected against. (Independently verified by the
monitoring session; the [0.5, 0.92] robustness band it first proposed
is corrected here to [0.5, 0.88] — g2's 0.886 ratio, which that
session lacked, caps it.)

**3. Disclosure:** the main-run outcome table (advantages −0.000 /
+0.000 / +0.112 / −0.069 at ctrl/base/g2/loud, |t| ≤ 1.45) was printed
and seen before this amendment — the verdicts script's layout prints
the table before the gate check, itself a flaw to fix in any future
verdicts script. The recalibration direction was chosen from the gate
analysis alone; note that no choice of these two gate constants can
alter any H0/H2/H3 verdict — gates only decide WHETHER verdicts are
read, and the verdict bars are untouched — so the seen table cannot
have steered the amendment toward a preferred outcome.

## 11. Contamination disclosure

All F27b gains, F28 directional numbers, F24 alignment results, and the
Dyna-2 report were seen before this design was written. NO F29 number
has been computed on any held-out material. The pilot procedure is
frozen (§3); α\* will be pinned in Amendment 1 after the pilot runs and
with its numbers disclosed. Pilot data is generated after this prereg
commit; joint checkpoints exist (F27b), disjoint checkpoints do not yet
exist. Predictions are registered at real odds informed by F28 + Dyna-2;
the F28 tie-in fork is committed pre-run so no answer is back-fittable.

Session record: this design emerged from the 2026-08-11 thread that
started with the [[dyna-2]] muscle-memory / grandmother-and-jar analogy,
proposed F32 (executive/muscle-memory stack) as the next fork, ran F32's
pilot, hit DESIGN VOID by the pilot's own frozen gate, and pivoted to
F29 as the sequel that picks up F28's directional zero more cleanly.
F32's parked branch remains a work-in-progress record and may be
resurrected with a different action-space design if the muscle-memory
question is worth pursuing after F29 reads out.
