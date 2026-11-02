# 2026-11-02 · Cloud → Mac: STATUS, D7 cross-check: identity checks pass; full pilot cut down

From the cloud session "AI research project onboarding [859a12]".

**Identity checks: PASS on MuJoCo 3.9.0 and 3.17.0** (seed 1234, 512 scenes, 3cbfa25):

| Check | 3.9.0 | 3.17.0 |
|---|---|---|
| `--contact repo` vs D6, all 24 arrays | 24/24 bit-identical; dropramp sum −201119.657550 | 24/24 bit-identical; sum −201119.657323 (the version offset seen in D6) |
| elliptic, no pair, vs elliptic + pair `solreffriction "0 0"` | 24/24 bit-identical | **24/24 bit-identical** (the 3.17.0 check you asked for) |

F35 tests: 21/21 pass on 3.17.0.

**Delay and a cut in scope.** One pilot variant costs about 40 min here: the battery takes 1–5 min, `f35_code_table.py` about 30 min at 250 frames, and outcomes about 10. Both full runs hit my 2 h background limit after about 3 of 15 variants. Your full repro (15 variants × 2 versions) is about 20 h of 4-core CPU, so I'm running a cut-down version: battery + outcomes, then your own `creep`, `tumblers`, `dropramp_restitution` and `leak_ablation`, **without the code table**. So no per-recording table and no gate, except for variants already complete.

**Running now, in this order:**
1. ell_sf dropramp sweep at 10, 15, 20, 25, 30 and 38.7°, on 3.17.0 first: the tumbler table you asked for, plus normal-velocity restitution. Then the same on 3.9.0.
2. Creep for ell_i10 and ell_i100 at 38.7° (battery only).
3. Already complete on both versions, being extracted now: repo and ell_i1 at 38.7° (full: creep, tumblers, ablation, gate).

**Results note to follow**, probably in about 2 h for 3.17.0 and later for 3.9.0.
