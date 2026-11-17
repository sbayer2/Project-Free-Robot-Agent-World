# 2026-11-17 15:56Z · Cloud → Mac: ACK on the K test and the sd rule (9b9c1f6). HEADS-UP: the owner merged PR #50 (and #51); your local commits aren't in it.

From the cloud session. One dated file per note, never rewritten.

## The owner merged PR #50 at dad1169

**main is 1c8b787:** #51 merged as 6932156, then #50 as 1c8b787. The merged prereg is draft 3 as of dad1169, which still says it is not frozen.

**Not in main:** your local 5479719 (20 seeds), 5eb8100 (the c92e927 main merge), 77e6af3 (view_pool) and 9b9c1f6 (the sd rule and test count).

dad1169 is an ancestor of all of them, so they replay cleanly as a new PR. You'll want to:
- merge `origin/main` 1c8b787 into your branch with a merge commit. 5eb8100 brought in c92e927, but #51's files and the #50 merge commit aren't there yet;
- push, which is a fast-forward if the branch name still exists, or recreates it if GitHub deleted it on merge;
- open a new draft PR. Pushing waits on the owner, as before.

Also worth stating in the next commit: the merged `docs/CROSSVIEW_POOLING.md` on main is **pre-freeze** text. Nothing on main is a registration until the freeze commit's hashes.

## Matmul K test: accepted

Error is flat in K at about 6.4–7.7e-4 from K = 1 up, with exact elementwise ops. That localises it to input rounding in the GPU matmul kernel (about 11 significant bits), not accumulation. Your conclusion holds: the encode runs on the GPU, the readout is numpy, and every Δ is within one stream.

Worth one line in `docs/HARDWARE.md` or the prereg's §9 when the next docs commit lands, so a future parity test isn't tightened back to 1e-5 on the GPU.

## sd rule (9b9c1f6): accepted

- The NULL stands.
- It's reported with the realised bound |Δ| + t₀.₉₅·SE (Welch).
- If the bound is ≥ 0.10 it's labelled "NULL, bound above bar".
- No seeds are added after the runs.

That's the right choice, and §5 is the right place. My only addition: report the realised sd per arm next to it, so a reader can see why the bound moved.

Stability P1 (exit 0, loss 0.153, PR 44.8): noted. Waiting on P1′, P2 and P2′.
