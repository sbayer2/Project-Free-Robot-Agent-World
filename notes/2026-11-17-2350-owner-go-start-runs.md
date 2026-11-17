# 2026-11-17 23:50Z · Cloud → Mac: the owner's go to start the pooling runs.

From the cloud session. One dated file per note, never rewritten.

The owner, in this session just now, verbatim: **"start runs"**.

**Scope, as frozen at be391d0, §3 and §9:**
- 115 MLX runs: 5 arms (P0 `mean`, P1 `stats`, P1′ `stats_ctrl`, P2 `attn`, P2′ `attn_ctrl`) × (20 loud seeds 0–19 on `pm_f31fix_loud_cam` + 3 ctrl seeds 0–2 on `pm_f31fix_ctrl_cam`).
- 50 epochs, lr 2e-4, B16, joint only.
- `crossview_measure` on the frozen CPU encode, `--d1` for P0 loud, then `crossview_verdicts --root runs/xv`.
- Training on the Metal GPU, by §4.
- About 6 h at your measured pace.

**Notes:**
- **The F-number is still unassigned.** By the frozen header it's assigned in a commit that changes nothing else. That doesn't block the runs.
- **Locked from the first loud pooling run:** per the header, no §3–§6 amendment can be made after a loud pooling run starts. Anything that has to be amended must happen before you launch.
- **Shared memory pool:** don't run LLM eval while training (CLAUDE.md).

**From the cloud while you run:**
- No torch pooling runs. Torch isn't the registered trainer, and the owner hasn't asked for a mirror.
- I'll verify `crossview_verdicts` output and the §6 grading against the frozen text when you post the report.

Please post a note when the runs launch, with the commit they run from, and another when the verdicts are in.
