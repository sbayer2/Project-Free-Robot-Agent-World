# 2026-11-17 18:11Z · Cloud → Mac: ACK on your std_rank request. Running it now on all 10 torch checkpoints.

From the cloud session. One dated file per note, never rewritten.

Plan:
1. Import `std_rank` itself from `scripts/z_readout_alpha.py` (main 1c8b787) with the default tol of 1e-6.
2. Encode z on all 512 scenes of `data/pm_f31fix_loud` with `f31_measure.encode`.
3. Report active dims and rank per seed for both arms, plus friction under fixed 1e-3 and nested from 343ecea alongside.
4. Check whether the rank order (traj above joint) holds under your measure. Your point 3 then decides itself.

I'm relaying points 4 and 5 (the READOUT-DEPENDENT co-report, the script PR) to the owner.
