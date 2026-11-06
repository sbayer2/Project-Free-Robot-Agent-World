# 2026-11-06 · Cloud → Mac: RESULTS, repo-wide restitution leak in the TILT probe: PRESENT (boxes); spheres' tilt is mostly restitution

From the cloud session "AI research project onboarding [859a12]". main @ 01643e1 (`generate_mujoco`, repo contact model: pyramidal cone, restitution → solref), MuJoCo 3.17.0, 512 scenes, seed 1234, `--keep-trajectory`, 1 view at 32 px (renders don't touch the physics). Worlds:
- `pm_big`-equivalent: default shapes;
- `pm_f31fix_loud`-equivalent: `--shapes box --coupling-alpha 1 --coupling-gain 8`.

20° tilt probe (the fixed ramp). "Should hold" = μ > tan 20° + 0.1 = 0.464 and not toppled. Field: `tilt.slid_distance` as stored. Shares: single-property cubic fit of log10(slid + 1 mm) over all scenes.

| world / shape | n | should hold | their slide: median / max | r(slide, e) | r(slide, μ) | all scenes: R²(log slide ← e) / ← μ | toppled |
|---|---|---|---|---|---|---|---|
| **pm_f31fix_loud-eq (box)** | 512 | 310 | **20.0 / 163.8 cm** | **+0.77** | −0.04 | 0.14 / 0.79 | 1 |
| pm_big-eq, all shapes | 512 | 110 | 115.5 / 452.7 cm | −0.05 | +0.12 | 0.04 / 0.25 | 302 |
| · box | 94 | 55 | 21.3 / 123.4 cm | **+0.78** | +0.21 | 0.17 / 0.78 | 0 |
| · capsule | 100 | 1 | — | — | — | 0.03 / 0.88 | 99 |
| · cylinder | 110 | 0 | — | — | — | 0.26 / 0.83 | 95 |
| · ellipsoid | 115 | 7 | 346.7 / 355.0 cm | +0.95 | +0.92 | 0.16 / 0.80 | 88 |
| · **sphere** | 93 | 47 | 444.2 / 452.7 cm | **+0.90** | +0.03 | **0.83 / 0.01** | 20 |

Sampled properties are independent: r(μ, e) +0.03, r(μ, ρ) +0.05, r(e, ρ) −0.04 (both worlds).

## Reading
1. **The F35 creep leak is in the repo's own worlds.** Boxes that should hold on the 20° ramp slide a median of about 20 cm (up to 1.6 m), set by restitution (r +0.77 / +0.78) and not by friction (r ≈ 0). Same mechanism as D6: soft pyramidal friction sharing the restitution-derived solref.
2. **Direction of the bias.** μ and e are independent, so the leak *cannot* manufacture friction signal. It adds restitution variance to tilt fields that are read as "friction", which dilutes friction readouts (conservative for friction claims) and lets the tilt block carry restitution. In the box world, friction still dominates the tilt slide (single-property R² 0.79 vs 0.14).
3. **Spheres: the tilt "slide" is mostly a restitution reading.** A sphere rolls whatever its μ (rolling needs only μ > (2/7) tan θ ≈ 0.10), so ideally every sphere travels the same distance. The observed spread is set by restitution (R² 0.83, against μ 0.01), presumably soft-contact rolling losses scaling with the restitution-derived damping. FINDINGS (line ~224) already calls sphere tilt "ill-posed (tilt just measures rolling)"; new is that **what it measures is restitution**. Capsules, cylinders and ellipsoids mostly topple and roll (88–99 of about 100–115), so "should hold" is undefined for them.
4. **Shape split (your ask):** the cube's should-hold restitution link (r +0.78 in pm_big, +0.77 in the box world) matches F35's (+0.87 at 38.7°, +0.75 at 24.2°). The non-box shapes mostly roll rather than creep, and for spheres the rolling distance itself is the restitution channel.

## What it touches (to check, not yet claims)
- **The tilt-bug report's variance shares** (friction 25% in pm_big, 87% in the box world): the 87% stands as "friction dominates the box world's behaviour", but part of the tilt block attributed to friction-or-shape carries restitution. And in pm_big, the sphere tilt block is a restitution channel.
- **F31 pilot (summary friction 0.83) and F31a:** friction readouts are, if anything, diluted, not inflated. No friction claim gets stronger. Restitution readouts that use tilt carry some of it.
- **F35 is unaffected:** D7 fixed this with the elliptic cone + box–ground `solreffriction` (frozen).
- Suggested record: a line in the tilt-bug report's open items, and a FINDINGS note when the owner next edits FINDINGS. Not done here (no edits outside these notes).

Reproduce: `generate_mujoco --output … --num-scenes 512 --views 1 --resolution 32 --seed 1234 --keep-trajectory [--shapes box --coupling-alpha 1 --coupling-gain 8]` (44 s each here, MUJOCO_GL=osmesa, one BLAS thread); then the should-hold filter on `behavior.probes[tilt]`.
