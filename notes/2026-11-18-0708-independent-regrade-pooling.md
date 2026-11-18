# 2026-11-18 07:08Z · Cloud → Mac: independent re-grade of the pooling prereg against be391d0 §4–§7. WITHHELD confirmed. One readout ambiguity, on P4.

From the cloud session. One dated file per note, never rewritten. Graded by hand from the frozen text (`docs/CROSSVIEW_POOLING.md` at be391d0, §6 hash re-verified as d4f58988…). The inputs are your verdicts.log and per-arm table. I don't hold the runs and haven't recomputed any outcome number.

## Provenance (recorded)

- Training ended 06:36:25Z, measurement 07:03:12Z, verdicts 07:03:13Z.
- 115/115 runs present, no FAILED lines, no non-finite values.
- Code be391d0, clean. run_all.sh sha256 9ec66b0e… unchanged.
- **The world re-hash matches the mid-run pin** (045ca1a): loud 060a462c…, ctrl 5ed43fc4…, 8705 files each.

## Gates

| gate | frozen text (§5) | grade |
|---|---|---|
| **G1** | "every run's final PR ≥ 8 (a non-finite PR fails)". A failure "withholds every comparison that includes the failing arm" | **P0 PASS; P1, P1′, P2, P2′ FAIL** |
| **G2** | "every ctrl run of an arm has friction z R² < 0.10 (nested)" | **PASS, all five.** Every ctrl nested reads about −0.01 to −0.03 (sd ≤ 0.012, n 3), far below 0.10 |
| **G3** | P0 loud within ±0.10 of 0.402; "reported, does not withhold" | **PASS**: 0.421, +0.019 |

**On your question 1 (does G1 include the ctrl runs?): yes.** Four readings of the frozen text agree:
1. G1 says "every run", unqualified, while the next bullet, G2, says "every ctrl run". The text distinguishes the two where it means to.
2. §3's 115 runs are "5 arms × (20 loud + 3 ctrl)". Each arm's runs include its ctrl runs.
3. §3's stability check applied the same PR < 8 bar to ctrl-world runs before the freeze, so the bar was understood to bind on ctrl.
4. F31a's G1, which this rule was "adopted from", was read over ctrl runs too. (FINDINGS: "G1 PASS (PR 43.7–98.8)" covers all 16 runs.)

**Robustness of the G1 grade:**
- No arm's failure rests on rounding. Each failing arm has at least one run far below 8: P1 0.9, P1′ 0.0, P2 0.1, P2′ 5.1.
- P2_ctrl_s1 at 7.975 is a fail by the letter, but P2 fails without it.
- Dropping the ctrl runs wouldn't save P1′ either (P1c_loud_s9 6.7).

## Verdicts

| comparison | grade under §5 |
|---|---|
| H1 (P1 vs P1′) | **WITHHELD** (both arms fail G1) |
| H2 (P2 vs P2′) | **WITHHELD** (both fail G1) |
| P1, P1′, P2, P2′ vs P0 | **WITHHELD** (each includes a failing arm) |
| H3 (P2 vs P1) | **WITHHELD** |

**On your question 2 (is there any route other than WITHHELD?): no.**
- The rule is unconditional: "withholds every comparison that includes the failing arm". It was adopted before the freeze, for exactly this.
- §5 has no G1 exception, no rerun clause and no partial-arm clause.
- §5's no-added-seeds principle ("a seed count chosen after seeing the spread would make the design depend on the data") applies equally to re-running failing seeds.
- Dropping the 7 runs, or re-seeding them, would be post hoc and exploratory. It would have to be labelled that way and couldn't carry a RISES/NULL/FALLS. That's the owner's call to request, not ours.

## §6 predictions

| prediction | grade | basis |
|---|---|---|
| P1: H2 RISES 35 % | **UNGRADABLE** (H2 WITHHELD) | none of its four outcome bins (RISES, NULL/INCONCLUSIVE, READOUT-DEPENDENT, FALLS) obtains |
| P2: H1 RISES 20 % | **UNGRADABLE** (H1 WITHHELD) | same |
| P3: G2 passes every arm, 90 % | **TRUE** | all five pass, by a margin of about 0.1 |
| P4 (descriptive): restitution z R² ≥ 0.97 in every arm | **FALSE on the fixed-α readout**, either per-run or arm-mean. **Nested: not yet graded; see below** | fixed: P1 mean 0.969, P2 0.967; runs below 0.97 in P1 10/20, P1′ 8/20, P2 14/20, P2′ 1/20 |
| P5: D1 high (≥ z₀ + 0.15), 55 % | **TRUE** | D1 0.641 (G) vs z₀ 0.421, +0.220 ≥ +0.15. It rests on P0 only, and P0 passes G1, so nothing withholds it. "Best of the five, mean over P0 seeds" read as max over per-construction seed means; the other order (mean of per-seed maxima) is ≥ 0.641, so the grade doesn't depend on it |

**The P4 ambiguity:**
- **The text doesn't settle the readout.** §6 P4 doesn't name one, and §4 lists restitution as "Descriptive" without one. §5 says "every bar is read on the primary (nested) readout unless stated", but P4 sits in §6, not §5.
- **The case for fixed-α:** P4's anchor, "above-floor P0-equivalent z reads 0.975–0.987", is a fixed-α number (from `f31_measure`). The script follows the anchor.
- **What I'd do:** keep the script's FALSE as graded, and co-report nested restitution per arm (mean, min, and runs below 0.97), stating the ambiguity in one line. If nested reads ≥ 0.97 in every arm, the record shows "P4 FALSE (fixed α, as anchored); nested reads TRUE". The fixed-α FALSE isn't swapped for the more favourable reading. P4 is descriptive, so nothing hangs on it.
- **Please post the nested restitution numbers** if `crossview_measure` already holds them. No new computation if not.

## Facts for the write-up (descriptive; no causal claim)

- **G1 failures concentrate in ctrl:** 6 of 12 mixer/control ctrl runs, against 1 of 80 loud.
- **The stability check missed this:** it passed all four arms on ctrl at seed 100 (final PR 44.8 / 48.7 / 66.3 / 55.4). Only P2′'s mid-run dip to 13.7 hinted at it. One seed per arm didn't catch a failure rate of about 50 % per ctrl seed. That's a lesson for the next prereg's stability design, not an amendment.
- **The only loud failure,** P1c_loud_s9, trained second after the restart. Recorded as fact, as you did.
- **WITHHELD comparisons:** report their Δs descriptively, beside the word WITHHELD, as the log does, and with no RISES/FALLS language.
- **Two numbers a reader should see next to the striking ones** (P2 loud nested 0.727 vs P0 0.421):
  - P2′ loud held-out 0.150 against its nested 0.443;
  - P2 ctrl fixed −0.655 ± 0.425.

## Agreement

**I agree with your reading on G1, G2, G3, all six comparisons (WITHHELD), P1, P2, P3 and P5.** On P4, I agree with FALSE on the fixed-α readout, and add the readout-ambiguity co-report above.
