# 2026-11-06 · Cloud → Mac: notebook v18 (freeze) + starting the repo-wide restitution-leak check

From the cloud session "AI research project onboarding [859a12]". The owner says: keep coordinating through these notes, and proceed with all tests.

## Class notebook v18 published
https://claude.ai/artifact/G1eRdPJv4VZaaxavTzX7HJ. **Start any class-page edit from v18.**
- Entry 18 status: "both fixed Nov 4". A dated update says the firmer friction is now frozen, with ramp and kick restitution 0.00 / −0.01 and friction 0.98 / 1.00 over five seeds.
- Entry 19 status: "decided Nov 4". A dated note covers the 0.3 m drop onto the 20° ramp, keeping every box, and the same no-mask rule for kick3.0 (44 tippers, mostly grippy; slide still tracks μ at r −0.95).
- **New entry 20, "The tests are locked":**
  - the designated pairs and margins (density +0.06, the thinnest; friction and restitution +0.15);
  - dropramp not load-bearing;
  - 5 seeds and 3 configurations agree;
  - provenance recording;
  - the one-BLAS-thread speed-up;
  - next steps (preregistration and first training; R1 not frozen);
  - the lesson: sealing the answer key before the exam.
- A new discussion question about freezing.

## Starting now: the repo-wide leak check (my standing open item, zero training)
**Question:** do the repo's own worlds, built with `generate_mujoco.build_mjcf` (pyramidal cone, restitution → solref), carry the same restitution-driven creep in their **tilt** probe that D7 fixed in F35? If so, part of the "friction" credited to tilt since the ramp fix (#45) is restitution. That would touch:
- the tilt-bug report's friction shares (25% / 87%);
- the F31 pilot's summary friction 0.83;
- F31a.

**Plan:**
1. Regenerate a `pm_big`-equivalent (main @ 01643e1: 512 scenes, default shapes, seed 1234) with `--keep-trajectory`. Renders don't affect the physics, so 1 view at 32 px is enough.
2. Regenerate a `pm_f31fix_loud`-equivalent the same way.
3. On the 20° tilt probe, for boxes that should hold (μ > tan 20° + 0.1): creep median and max, and r with restitution and with μ.
4. How much of `tilt.slid_distance`'s variance tracks restitution vs friction overall, and among boxes that should hold.

Results in a new dated note.
