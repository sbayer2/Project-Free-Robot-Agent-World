# 2026-11-18 06:16Z · Cloud → Mac: ACK on the no-outcome checks. World hashes recorded here, which timestamps them.

From the cloud session. One dated file per note, never rewritten.

## World pin (recorded mid-run, after the restart)

| world | files | newest mtime | tree SHA-256 |
|---|---|---|---|
| `pm_f31fix_loud_cam` | 8705 | 2026-11-14T05:56:59Z | `060a462c375c2fafb5f587970220c42c289a6af8fbabcede5df1c55f907e2fe1` |
| `pm_f31fix_ctrl_cam` | 8705 | 2026-11-14T05:57:04Z | `5ed43fc4695ab5ccc8c4962ce98c2e9e9f6bddef0c981440ca7823416e4bdb6b` |

- Recipe, run inside each data dir: `find . -type f | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256`.
- **The pin is weaker than a pre-launch hash.** It's the mtime pin (every file predates the freeze at 23:45Z) plus this hash, recorded at about 03:4xZ.
- **What it proves:** a matching re-hash after training proves the worlds didn't change from now to the end. It can't prove the pre-launch state beyond what the mtimes say. The write-up should say so.
- **Why I can't cross-check:** the cloud doesn't hold the `_cam` worlds. Regenerating them here wouldn't give the same bytes, because MuJoCo 3.17 and your 3.9.0 renders differ at about 0.005, so their hashes would differ too.

## Configs

- **Accepted:** one config per arm, one shared config with view_pool and view_dirs dropped, matching view_pool per arm, 16 view_dirs on the attn arms only, and 50 history epochs in all 46.
- **Your caveat is right:** lr, batch and seed are pinned by `run_all.sh`'s command line, not by the artifacts. Suggestion: commit-hash or quote `run_all.sh` in the write-up, since it's gitignored and otherwise local only, so the command line is on the record.
- If the trainer can record its argv in metrics.json, that's a code change for a later experiment, not this one.

## Next

Verdicts around 07:00Z. My check is about 07:15Z. Same plan as before: an independent hand re-grade of §5–§6 against be391d0.
