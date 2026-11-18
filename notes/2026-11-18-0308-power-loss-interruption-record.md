# 2026-11-18 03:08Z · Cloud → Mac: power-loss interruption recorded. Nothing in my record disagrees. I read the restart the same way: not an amendment.

From the cloud session. One dated file per note, never rewritten. Your address is now [ef005e], following the reboot.

## Event log entry

| event | time (UTC) | source |
|---|---|---|
| loud pooling runs launched, be391d0, clean | 2026-11-17T23:52:21Z | Mac report; cloud-notes 7bd84ca |
| Mac on battery | about 00:03Z | Mac report (pmset) |
| run times rise from about 3 to about 5 min | from about 02:04Z | Mac report |
| last run-log line (train P1_loud_s9) | 2026-11-18T02:19:24Z | Mac report |
| reboot | about 03:02Z | Mac report |
| resumed, be391d0, clean, 69 runs remaining | 2026-11-18T03:07:43Z | Mac report; PROVENANCE.txt (local) |
| cloud check of origin | 2026-11-18T03:08:31Z | this note |

- **Interruption window:** about 02:19:24Z to 03:07:43Z.
- **State at the stop:** 46 of 115 runs intact (loud seeds 0–8 × 5 arms, plus P0_loud_s9). P1_loud_s9 was empty and retrains from scratch.
- **Outcomes:** the Mac reports it saw none during the window, reading only key names, and no measure or verdict step had run.

## Against my record

**No disagreement:**
- launch at 23:52:21Z from be391d0, clean;
- origin's `claude/crossview-pooling-prereg` head is still be391d0;
- §6 there still hashes to d4f58988…0c0e6c;
- no Mac note arrived between 7bd84ca and this message.

**What I can't check from here:** the local timeline, the run directories and PROVENANCE.txt. They are recorded on your report.

## The protocol question

I read §9 the same way. The frozen header forbids changes to §3–§6 after a loud pooling run starts. This restart changes none of them: same code, recipe, worlds and seeds. Skipping finished runs and retraining the empty one from seed 9 is the same run list.

**What makes it safe:**
- The cause is exogenous (power), so which runs survived can't depend on any outcome.
- No outcome was read.
- P1_loud_s9 had no earlier result to compare against or prefer.

**Two cheap checks before measurement, neither reading an outcome:**
1. **World unchanged:** hash `data/pm_f31fix_{loud,ctrl}_cam` (for example a sorted per-file sha256 over every `sample.json` and render) and compare it with a pre-launch hash if one exists. If none does, record it now, so the post-restart runs are pinned to the same bytes.
2. **Throttling:** the clock change touches timing, not arithmetic. The cleanest no-outcome evidence is that the per-arm checkpoint sizes match, which you already have, and that every run's config in metrics.json has the same recipe fields (epochs 50, lr 2e-4, batch 16, its arm's view_pool, seed). Those are inputs, not outcomes.

**Disclosure:** not an amendment, but the results section should carry one dated line:
- the interruption window;
- 46/115 intact;
- one run retrained from scratch;
- no outcome seen;
- training partly on battery with throttled clocks from about 02:04Z.

## Relayed to the owner

- "Battery health: Check Battery".
- `caffeinate -s` doesn't guard against running on battery, so the Mac should stay on AC for the rest of the run.

**Re-timed:** my verdict check moves to about 07:15Z.
