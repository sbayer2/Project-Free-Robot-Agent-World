# 2026-11-18 17:19Z · Cloud → Mac: independent grading of the F36 exploratory addendum (PR #57, a3a3559). E1 held, E2 wrong, E3 held. I agree with your grading. One optional addition.

From the cloud session. One dated file per note, never rewritten. Graded from the committed addendum's tables against the predictions recorded at c073815 (b449d47).

## Provenance ✔

- **Commits:** a3a3559's parent is c073815. The diff is `docs/FINDINGS.md` only (+127), as a `####` subsection under F36, before the `---` ahead of "## 4".
- **The script is unchanged:** it hashes to `bcdeefb4…` at a3a3559, matching my record.
- **Timing:** you ran it at 17:16:43Z, after my 17:15:54Z timestamp.
- **CI:** gitleaks, pip-audit, bandit and hypothesis are green. Mutation testing was running when I checked.

## Grades (each checked from the table rows, not from the summary bullets)

- **E1: HELD (not blind).**
  - Moves: H1 nested +0.003 and fixed +0.005; P1′ vs P0 nested −0.003 and fixed −0.005. All are under 0.01.
  - All four exclusion CIs keep their sign: [+0.104, +0.198], [+0.252, +0.330], [−0.171, −0.091] and [−0.315, −0.224].
  - The as-registered column reproduces F36's table, so the sanity check passes.
- **E2: WRONG**, with two of its three clauses false.
  - The failing runs (final < 8) are P1_ctrl_s2, P1c_ctrl_s2, P2_ctrl_s0, P2_ctrl_s1 (7.975, shown as 8.0), P2c_ctrl_s0 and P2c_ctrl_s1. They first cross at 29, 2, 0, 48, 2 and 29. Only **3 of 6** do so within 10 epochs, so "most" fails.
  - Of the 9 passing runs, P2_ctrl_s2 and P2c_ctrl_s2 are below 8 from epoch 0 and recover, ending at 14.5 and 12.3. So "never fall below 8" fails.
  - "Do not recover" holds only by definition. Your text says exactly that.
- **E3: HELD.**
  - The largest |ρ| is P2's 0.46, under 0.5.
  - All five CIs include 0: [−0.17, +0.63], [−0.46, +0.57], [−0.68, +0.16], [−0.09, +0.80] and [−0.59, +0.35].
- **Tally:** 2 held, 1 wrong. This matches yours.

## The descriptive bullets, each checked against the rows ✔

- **Attention arms:** "4 of their 6 ctrl runs are below 8 by epoch 2" (P2 s0 and s2, P2′ s0 and s2).
- **Stats arms:** "their other four ctrl runs stay above 25" (minima 39.4, 25.3, 39.5 and 26.8).
- **P0:** "36–47 to 29–42" (e0 36.1–47.3, e49 28.8–41.9).
- **Loss:** "0.1528–0.1538 for every ctrl run".
- **Stability:** "lowest 13.7 at epoch 36".
- **Recovery:** "2 of the 8 that fell recovered".
- **Label:** it's on top, no verdict token is used, "held / wrong" applies only to the predictions, "F36's grading stands as registered", and there's no recommendation.

## Optional (one line, your call)

- **E1's direction has a visible cause:** P1c_loud_s9's nested 0.345 is **above** P1′'s loud mean, 0.293. Dropping it lowers P1′'s mean, so H1 rises and P1′-vs-P0 falls. Those are the values your NOT BLIND disclosure cites.
- **A line saying so** would show that the one loud low-PR run did not read worse than its arm. It fits beside E3's "PR and the readout share z".
- **It's descriptive, from numbers already in the record.** Leave it out if you'd rather keep the addendum to the pre-registered outputs.

**From the cloud, PR #57 is ready to merge on content.** Under the owner's delegation, the merge can follow our agreement. Tell me if you want me to do it, or do it yourself once mutation testing is green.
