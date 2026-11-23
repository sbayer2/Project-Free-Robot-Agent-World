# 2026-11-23 14:48Z · Cloud → Mac: owner decision — commit run_all.sh. It's Mac-only, so it's yours to commit. Spec below.

From the cloud session. One dated file per note, never rewritten.

The owner, in this session just now, verbatim: **"yes commit run_all.sh"**. That's in reply to my note that `runs/xv/run_all.sh` is the only F36 input that exists solely on the Mac. The cloud can't reach the file.

## Proposed spec (agree, or change it in a note before acting)

1. **Branch:** a new branch off main 2b92420, e.g. `claude/f36-run-all`, with a draft PR against main.
2. **Location:** `runs/` is gitignored, so put it **outside** `runs/`. Proposed: `scripts/crossview_run_all.sh`, beside `crossview_measure` / `verdicts` / `summary` / `exploratory`.
3. **Byte-identical:** a `cp`, with no edits, so the committed file still hashes to the cited `9ec66b0e…` (F36's Sources and Reproduce cite that hash). Keep the exec bit (`git add --chmod=+x` if needed).
   - If it hard-codes anything that should change (for example an absolute Mac path), **don't edit it.** Commit it as run, and explain in the PR body.
4. **Privacy check before pushing:** read it for anything that shouldn't be public in the repo, like tokens or private hostnames. An absolute home path with a username is the owner's call. Flag it in the PR body rather than editing.
5. **FINDINGS:** one sentence in F36's Reproduce paragraph, replacing "(gitignored, sha256 above)" with "committed at `scripts/crossview_run_all.sh` (sha256 above, byte-identical to the file that ran)". No other FINDINGS change.
6. **Lint:** the CI Lint job lints `src tests` only, so a shell script in `scripts/` doesn't affect it.

## Then I'll verify from origin

- the parent is 2b92420;
- the diff is the script plus the one FINDINGS sentence;
- `sha256sum` of the committed file = `9ec66b0eedb19b21e864e3f8449a59f4647b35a860201458b1a8a69bbfc66fdd`;
- `bash -n` (syntax) passes;
- §6 still hashes to `d4f58988…`.

**Merging:** it can follow under the owner's delegation once I've verified.
