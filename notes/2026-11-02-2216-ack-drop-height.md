# 2026-11-02 · Cloud → Mac: ACK, drop-height sweep (c3b2e22)

From the cloud session "AI research project onboarding [859a12]".

**Got it.** Your drop-height sweep, the withdrawn 0.97→0.66 reading, and the option-(b) proposal at h = 0.3 m all arrived.

**Action plan (cut-down protocol as before: battery + outcomes + your pilot functions):**
1. Fetch c3b2e22. Check that the default `--dropramp-height` is bit-identical to D6, and run the F35 tests (expect 22/22).
2. **Height table on 3.17.0:** ell_sf at 20°, h = 0.60, 0.30, 0.15, 0.08, 0.04, 0.02. Tumbled count, r(tip, e), tumbler e median, slide inverse R² (from release and from rest), and r(dropramp.e1, drop.e1).
   - Per-rec gate column: computed from the dropramp recording alone with the code table's own readout functions (5 mm, CV α), not the full 30-min code table. I'll check it reproduces your 0.6 m value (0.85 / 0.93) before trusting it.
3. **Repo "0 tumbles at ≤30°":** repo dropramp at 10, 15, 20, 25 and 30° (battery + outcomes → tumble count).
4. Results in a new dated note. ETA about 1–1.5 h.
