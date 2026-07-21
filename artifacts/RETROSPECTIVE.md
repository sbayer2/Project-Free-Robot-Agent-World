# pseudo-marble retrospective — the F21 terminus

**Date:** 2026-07-21  
**Scope:** pseudo-marble v0.0.1, F1–F21  
**Status:** fidelity arc closed; negative result reported at equal prominence

---

## What the project set out to test

pseudo-marble is a small, runnable instrument for the claim behind World Labs' Marble and Fei-Fei Li's taxonomy of world models: that a machine can hold *one physical essence* of an object from which both appearance and behavior are projected, rather than two separate decoders glued together.

The apparatus:
- a controlled synthetic world (MuJoCo + continuous materials),
- a ~1M-parameter shared-latent model (multi-view images → z → render/behavior/essence heads),
- a coherence metric that asks: when you nudge the latent, do look and behavior move together?
- honest controls (untrained shared baseline, independent disjoint-latent baseline, held-out essence regions).

---

## What we found

### The only unconditional positive

**F13:** on `pm_big`, 20 seeds, lr 5e-4, the trained shared-latent model achieves learned coherence **+0.146** (behavior) and **+0.123** (essence), t ≈ 8.3 and 5.5. The independent disjoint-latent baseline sits near 0; the untrained shared baseline is ~0.10. So training genuinely produces a small shared structure.

### The correction that re-frames it

**F18:** the same model's held-out behavior gain over predict-mean is **1.413×** (range 1.32–1.64). A **shape-only oracle** — which never sees density, friction, or restitution — scores **1.331×**, and its 95% CI [1.01, 1.79] contains every trained seed. The fair ceiling, shape + the actual visual appearance parameters, is **2.311×**.

So the model extracts roughly **8% of the reachable essence signal**. The coupling measured in F13 is real, but it is largely **shape-driven**, not material-essence-driven.

### The fidelity arc

- **F19:** a probe `z → appearance_params` showed the latent retains color and density, but not the contact-physics channels (friction, restitution).
- **F20:** adding an auxiliary `z → appearance_params` head successfully forced roughness/metallic/transmission back into `z`, but behavior gain moved only 1.37 → 1.46 (ceiling 2.26). The bottleneck was not the behavior head alone.
- **F21:** two render-fidelity levers — (1) removing authored appearance noise and (2) oblique 256 px specular lighting — both improved the *availability* and *legibility* of the essence channels. Neither moved behavior gain off **~1.5**. Render fidelity is not the barrier.

**Terminus:** the authored appearance↔physics coupling is, by construction, too subtle for the encoder to exploit. The gap is intrinsic to the synthetic world, not fixable with more resolution, better lighting, or an auxiliary head.

---

## The reality test

**F14/F16** ran the same instrument on real Amazon Berkeley Objects (ABO) products. The result was **VOID** by its own preregistered gates: with friction and restitution assumed constant across all objects, the probe battery is nearly mass-blind, so held-out prediction fails and coherence is uninterpretable. Within-category holdout changed nothing. The real-object question remains unanswered and is parked until either real contact parameters are available or a mass-sensitive probe family is designed.

---

## What this means

The headline is now a calibrated negative:

> On a synthetic world with a deliberately subtle appearance↔physics coupling, a shared latent learns a small, real, shape-driven coupling. It does **not** learn the hidden material essence. Improving render fidelity, removing appearance noise, or adding an appearance auxiliary does not change this.

This is a useful result. It defines where the shared-latent idea stops working in this controlled setting and prevents others from overclaiming on similarly subtle couplings.

---

## Strategic fork

The project now faces two honest paths:

1. **Accept the negative and stop.** The F18–F21 arc is a complete, preregistered, multi-seed finding. Ship it.
2. **Build a stronger-coupled synthetic world.** If the goal is to study whether a shared latent can learn a *material essence*, the generative coupling in `materials.py` must be made legible by design — e.g., stronger roughness↔friction and metallic↔density mappings, informative visual cues for bounciness/heaviness, and a behavior target that isolates essence from shape.

Path 2 is a new research phase, not a tweak. It needs a new preregistration and a new dataset.

---

## Repository state at terminus

| Item | State |
|---|---|
| Branch | `main` with `claude/render-fidelity` merged |
| Tests | 205 passed, 1 skipped, ~2.5 s (`.venv/bin/python -m pytest`) |
| Local feature branches | cleaned (`appearance-aux`, `probe-appearance`, `render-fidelity` deleted) |
| Remote feature branches | still exist on origin; deletion pending user authorization to push |
| Docs | `docs/FINDINGS.md` contains the full empirical record; `README.md` and `CLAUDE.md` status updated |
| Artifact | this retrospective (`artifacts/RETROSPECTIVE.md` + `.html`) |

---

## Reproduce the headline

```bash
# F13 checkpoints (lr 5e-4, 20 seeds)
python -m pseudomarble.models.train --data data/pm_big --out runs/exp_s0 --seed 0 --lr 5e-4
# ... seeds 1..19
python scripts/run_coherence_experiment.py --data data/pm_big \
    --checkpoints runs/exp_s0/model.safetensors,...,runs/exp_s19/model.safetensors \
    --untrained-seeds 10 --out runs/retrospective_coherence

# F18 oracle ceiling
python scripts/oracle_ceiling.py --data data/pm_big

# F19 appearance probe
python scripts/probe_appearance.py

# F20 appearance auxiliary sweep
python -m pseudomarble.models.train --data data/pm_big --out runs/retrospective_aw0 --seed 0 --lr 5e-4 --appearance-weight 0.0
python -m pseudomarble.models.train --data data/pm_big --out runs/retrospective_aw3 --seed 0 --lr 5e-4 --appearance-weight 0.3
python scripts/appearance_aux_eval.py

# F21 render fidelity
python -m pseudomarble.data.generate_mujoco --output data/pm_n00 --appearance-noise 0.0 --num-scenes 512 --views 16 --resolution 128 --seed 1234
python -m pseudomarble.data.generate_mujoco --output data/pm_obl256 --lighting oblique --num-scenes 512 --views 16 --resolution 256 --seed 1234
python scripts/render_fidelity_eval.py
```

---

## Honest limits still on the books

- The coupling is **authored**, not measured from reality. The synthetic world is a probe of learnability, not a claim that the model discovered physics.
- The real-object test is **VOID**, not negative. No public dataset ships measured friction/restitution, and the current probes are mass-blind without them.
- The model is small by design (~1M params, MacBook-native). The result does not rule out larger models on richer worlds; it only says this particular controlled world does not admit essence learning.

---

*Personal research; not affiliated with World Labs.*
