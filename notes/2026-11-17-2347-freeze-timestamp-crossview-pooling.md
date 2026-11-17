# 2026-11-17T23:47:35Z · Cloud timestamp record: cross-view pooling preregistration FROZEN at be391d0. §6 hash verified, MATCH.

From the cloud session. One dated file per note, never rewritten. Recorded on the owner's sequence: "after the push comes the freeze commit. I hash the §6 predictions, the cloud records the timestamp, and I assign the F-number."

## Record

| field | value |
|---|---|
| freeze commit | `be391d0608295034f5ba812643cc3560f9047b66` |
| branch / PR | `claude/crossview-pooling-prereg`, PR #52; origin head = be391d0 at verification |
| commit time (origin, committer) | 2026-11-17T17:45:04-06:00 = **2026-11-17T23:45:04Z** |
| file | `docs/CROSSVIEW_POOLING.md` at be391d0 (UTF-8, LF, no CR bytes) |
| range | first byte of the `## 6. ` line (line 719) up to, not including, the `## 7. ` line (line 755); trailing "\n\n" included; **1994 bytes** |
| Mac's SHA-256 | `d4f58988fb900aac788ed889e1aa3e0e96a5a4f981939cea67adf643c10c0e6c` |
| cloud recompute, awk + sha256sum | `d4f58988fb900aac788ed889e1aa3e0e96a5a4f981939cea67adf643c10c0e6c` |
| cloud recompute, Python byte-index | `d4f58988fb900aac788ed889e1aa3e0e96a5a4f981939cea67adf643c10c0e6c` |
| verdict | **MATCH** |
| cloud verification time | 2026-11-17T23:46:38Z (this note's GitHub commit time is the third-party timestamp) |

## Checks

- **Same §6 as the owner accepted:** §6 at f35a496 hashes to the same value, so the freeze commit didn't change §6.
- **The freeze changed only the doc:** be391d0's whole diff against f35a496 is `docs/CROSSVIEW_POOLING.md`, 37 lines added and 8 removed, all outside §6.
- **Ancestry:** 931adb4, f35a496 and f66943a are all ancestors of be391d0.
- **Tests at be391d0** (torch 2.15.0+cpu, MuJoCo 3.17.0):
  - `pytest tests/test_view_pool.py -k torch`: 5 passed;
  - full suite with OSMesa: 401 passed, 26 skipped.
  - My earlier parity run was at f66943a (9855e04).

## Statement on pooling-variant models

To my knowledge, no loud-world pooling-variant model exists from the cloud side.
- The only trained models in this container are the 10 torch F31a checkpoints in `runs/f31fix/` (loud_joint and loud_traj, seeds 0–4).
- Their configs predate `view_pool`, so they load as the default `"mean"`. They weren't trained under this prereg's recipe, world or budget.
- A disk-wide search finds no other `metrics.json`.
- No stats, attention or control pooling arm (P1, P1′, P2, P2′) has been trained here.
- Nothing has been trained under `claude/crossview-pooling-prereg`.

## Open

- **F-number:** the owner assigns it in a later commit that changes nothing else.
- **No runs start until the owner says so.**
