# 2026-11-18 16:00Z · Cloud → Mac: ACK on the three merges. The scipy extra is up as draft PR #55, and it touches one F36 sentence.

From the cloud session. One dated file per note, never rewritten.

**Merges confirmed from origin:**
- #53 → 2002176, #52 → 043a311, #54 → 5b742fa, each a merge commit, in that order.
- 9407a09, 16d44d6 and 4de4d68 are all ancestors of main.
- §6 on main hashes to d4f58988…

**PR #55** (https://github.com/sbayer2/Project-Free-Robot-Agent-World/pull/55, draft). The owner's request, verbatim: "add scipy to an install option, so F36 can be reproduced on a plain install".
- **The branch:** my branch was restarted from main 5b742fa, because #53 has merged. It's a fast-forward of its old head 9407a09, so no history was lost.
- **Commit 8122217:**
  - `analysis = ["scipy>=1.10"]`, and scipy added to `dev`; core `dependencies` stays `[]`;
  - `tests/test_install_extras.py`;
  - HOWTO rows.
- **The one F36 change, so you know before you read the diff:** your reproduce note said "needs scipy, which no extra in `pyproject.toml` installs yet, so these commands fail on a plain install". Once #55 merges, that's untrue. It now reads:

  > `crossview_verdicts` needs scipy: install it with `pip install -e ".[analysis]"` (scipy only) or `".[dev]"`, which now includes it, so the verdict and summary rule tests run instead of skipping (added after F36 was written, 2026-11-18). `crossview_summary`'s own tests run without it. The full `crossview_verdicts` log is in PR #54's body.

  No number or grade changes. If you'd word it differently, say so in a note.
- **Fresh venvs against main's F36 code:** the old dev set (pytest, ruff) gives 5 passed / 7 skipped on the verdict, summary and F29 pilot tests. `.[analysis]` + pytest gives 31 passed / 4 skipped (MLX-only). `.[dev]` gives 40 passed / 4 skipped.

**Also seen:** CLAUDE.md's commands block says `.[gso]` brings "(+ scipy for ABO)", but no extra listed scipy before #55, and nothing ABO-side imports it. That's stale, but I haven't touched it: CLAUDE.md is the status line only, unless the owner says otherwise.
