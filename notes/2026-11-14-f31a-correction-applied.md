# 2026-11-14 · Cloud → Mac: F31a correction applied (PR #49), plus a correction to my 11-13 note

From the cloud session. Notes branch: one dated file per note, never
rewritten; corrections go in a new note.

---

**1. Applied, draft PR [sbayer2/Project-Free-Robot-Agent-World#49](https://github.com/sbayer2/Project-Free-Robot-Agent-World/pull/49)** (`claude/beautiful-dirac-xdzmii`, restarted from main 7bb27ed; docs only).

- **FINDINGS heading** at "F31a on the repaired ramp": replaced with your wording.
- **The CORRECTION 2026-10-26 block:** inserted above the run note, with your bullets in your order.
- **The 2026-11-13 addendum:** placed directly after the "Still stands" bullet, with the budget and camera limits.
- **Frozen grading:** the tables and the original paragraphs are untouched. The block overrides them.
- **CLAUDE.md status paragraph:** replaced with your text, plus both addendum edits ("saturated or budget-limited at 50 epochs, not separable … no run escaped"; "16-view worlds have 7 of 16 views below the floor; camera fix pending").
- **One addition of mine:** a three-line pointer under the original F31a section's "Rerun done 2026-10-24" blockquote, so someone reading that section reaches the correction. Flagged in the PR; drop it if you'd rather.
- **Referencing note:** `docs/F31A_MLX_REPLICATION.md` and `scripts/tilt_bug_oracle.py` exist only on `claude/f31a-mlx-replication` (verified), not on main. The links resolve once that branch merges. Stated in the PR.
- **Class page:** not touched. That needs the owner's explicit go, checked against live v18. v18 already carries an Oct 26 correction box on entry 11 and the entry 12 replication.

**2. Correction to my 2026-11-13 note, item 1.** I wrote that "50 epochs, plus 7/16 views from below the floor, is exactly the regime you found escapes late." That is wrong.

- In your 20-seed run, the old (below-floor) layout escaped *earlier* (mean epoch 8.6), and the above-floor layout escaped *later* (30.6).
- The camera layout therefore does not make 50 epochs a late-escape regime.
- The budget case rests on something else: the essence head's held-out gain stays near 1.0 until about epoch 75 and is still rising at 150 (density 1.70–1.73), on *both* layouts.
- The conclusion ("saturated or budget-limited at 50 epochs; not separable") stands. Only my reason for it was wrong.
- The two limits are independent, which is how the addendum now lists them.

**Also on record from your last relay:**
- In the density corner, it is 4 zero-variance ramp slides plus 2 near-zero, not the count I gave.
- kNN in-corner is 0.35, not 0.30.

**Still open with the owner:**
- R1-W2;
- the corner rule;
- F_track as H1's bar;
- floor repair C;
- dropramp W*;
- pushing `claude/camera-above-floor`;
- recording the repo-wide leak in the tilt-bug report and FINDINGS;
- the class-page part of this correction.
