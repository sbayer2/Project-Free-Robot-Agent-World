# 2026-11-04 · Cloud → Mac: pre-freeze checklist (owner chose "apply recommendation, freeze")

From the cloud session "AI research project onboarding [859a12]". Checked against my D7 batteries at ad82352.

**1. Provenance is missing from the manifest. Fix before freezing.** The D7 `manifest.json` has `design_iteration: "D7"` but **no MuJoCo version, numpy version, git commit or platform.** D7 is the most version-sensitive battery so far (ramp-family sums move by up to +49 between 3.9.0 and 3.17.0, though no gate cell moves). This is also the tilt-bug report's "no manifest records which tilt code produced it" lesson. Proposal: the generator writes
`provenance: {mujoco: mujoco.__version__, numpy: np.__version__, python: platform.python_version(), platform: platform.platform(), git_commit: <HEAD sha or "unknown">, fields_version: "D7"}`
and `f35_seeds.py` / `f35_code_table.py` copy it into their outputs. Plus one test that the keys exist.

**2. Name the canonical build.** Freeze with "canonical: MuJoCo 3.9.0" (the Mac's), and state that 3.17.0 reproduces every gate cell and mean at two decimals (5 seeds, part 2 note). Optionally pin `mujoco==3.9.0` in a `[f35]` extra rather than the global `mujoco>=3.1`.

**3. The three open items, as the recommendation applies them** (my reading of the part 2 note; confirm your wording matches):
- (a) **kick friction target = kick1.0** (no tips, friction 0.96). kick3.0 stays a recorded family; its `log_stop` keeps the tip mask but is **descriptive only**, never a target for a verdict.
- (b) **The friction verdict is carried by ramp ↔ kick** (large margins). dropramp ↔ ramp/kick are reported but not load-bearing. That's because dropramp friction clears the 0.8 bar by only 0.02 (stable: sd 0.01 across 5 seeds × 2 versions).
- (c) The residual trajectory restitution (ramp 0.18, kick 0.13) is disclosed; no target field carries it.

**4. Freeze the numbers, not just the code:** gate bar 0.8, gate noise 5 mm, CV-α readout, seeds `1234 1 2 3 4`, 512 scenes, `FIELDS_VERSION "D7"`, the outcome-field list, and the five-seed D7 table (yours, reproduced) as the frozen reference values a future rerun must match.

**5. Keep R1 (§10) open items visible:** the framing rule (W ladder logged as a D-iteration first; tie-break rule), and F_track as the graded pixel bar with F_px only choosing W. If R1 freezes in the same commit, these need to be settled; if not, say "R1 not frozen" explicitly.

**6. Practical:** set `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` in `f35_seeds.py`'s subprocess env. Here that took the code table from 30+ min to about 1 min and outcomes from about 10 min to 7 s.

After you push the freeze commit, I'll cross-check it on both versions (cheap now: about 15 min per design per version) and confirm the frozen reference table reproduces.
