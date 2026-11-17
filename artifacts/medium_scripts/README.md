# Medium draft build scripts

Rebuild the charts and the paste-ready HTML for `PSEUDO_MARBLE_MEDIUM.md`.

```bash
cd ~/Project-Free-Robot-Agent-World/artifacts

# charts (needs matplotlib + numpy; the project .venv has neither, ~/.venv does)
~/.venv/bin/python medium_scripts/medium_figs.py medium_figs

# HTML from the markdown (standard library only)
python3 medium_scripts/md2html.py PSEUDO_MARBLE_MEDIUM.md PSEUDO_MARBLE_MEDIUM.html
```

`medium_figs.py` holds every plotted number inline, copied from the project
records (FINDINGS, the tilt-bug propagation report, the F35 freeze table, the
lab notebook). Edit the numbers there, not in the PNGs.

Figure files are numbered in reading order (fig1 to fig7), matching the article.
