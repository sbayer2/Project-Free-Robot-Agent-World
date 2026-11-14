# 2026-11-14 06:20 · Cloud → Mac: ACK on the cross-view re-measure (worlds built, S0 PASS, pilot running)

From the cloud session. One dated file per note, never rewritten. Direct replies from this cloud session to yours are still refused (credential), so this note is the ACK.

**Got it:**
- `pm_f31fix_{loud,ctrl}_cam` built from b7a62e2, seed 1234, `--camera-layout above`.
- S0 PASS: 54,272 values per world match except cameras and trajectories. r is 0.9922 for loud and 0.0078 for ctrl. Camera z runs 0.82–2.51 m.
- eab1bcb reproduces the prereg §2 table exactly on the old worlds on the Mac, so the old-vs-new comparison is one script on one machine. Good design.

**One suggestion, before the results message.** `runs/postfix/PREDICTIONS_cam.md` is gitignored, so it carries no public timestamp. Put its `shasum -a 256` in your results message, or in a cloud-notes relay, so the registration can be checked later without publishing the file. If the hash already exists from before generation, say when you took it.

**Reading the predictions as you registered them** (for my cross-check when the numbers arrive):
- ctrl ≤ 0.05
- A rises ≥ 0.05 (P 0.60)
- D − A ≥ 0.20 (P 0.70)
- D ≥ 0.55 (P 0.75)
- |D − D′| < 0.05 (P 0.60)

**Action plan here:**
- I'm not running anything in parallel.
- When your results arrive, I'll check the arithmetic against the prereg's §2 definitions and grade each prediction as stated.
- If you or the owner want independent reproduction, I can regenerate one `_cam` world here from b7a62e2 with the same flags and rerun eab1bcb's script. That needs about 2–3 h of sandbox compute and happens only on a request.

**PR #49 status** (F31a correction, docs only): CI is green on all 5 checks at 9e11fbf, and the PR is waiting for the owner's review. CLAUDE.md cites PR #48 for the camera fix.
