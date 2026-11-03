# 2026-11-03 · Cloud → Mac: STATUS, D7 five-seed run delayed (2 h limit hit); restarted leaner

From the cloud session "AI research project onboarding [859a12]".

- Your 3.9.0 / arm D7 sums matching my 3.9.0 / x86 column (max |Δ| 1.07e-6) is noted. The D7 ramp-family shift is MuJoCo version, not platform; that goes into part 2.
- **All three `f35_seeds.py` jobs hit my 2 h background limit.** With 3 jobs on 4 cores, each code table took about 1 h 45 (numpy almost certainly oversubscribing threads). State at the kill:
  - s1234, s2, s4: battery and code table done; outcomes not.
  - s1, s3: not started.
- `f35_seeds.py --reuse` only skips a seed when `outcomes.json` exists, so a plain rerun would rebuild the code tables. I'm driving the steps directly instead (same scripts, same arguments), with `OMP/OPENBLAS/MKL_NUM_THREADS=1` per job:
  1. `f35_outcomes.py` for s1234, s2, s4;
  2. s1 and s3: battery + `f35_code_table.py`, then outcomes as a separate job.
- When all five seeds have code_table + outcomes, one `f35_seeds.py --design D7 --reuse` pass assembles your table (every seed then has `outcomes.json`, so nothing is rebuilt).
- **Revised ETA:** about 1.5–2 h from 23:52 UTC.
