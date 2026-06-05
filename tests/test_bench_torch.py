"""Smoke test for the torch scale-bench tool (skips without torch)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch  # noqa: F401
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False


def test_run_case_completes():
    if not _HAVE_TORCH:  # pragma: no cover
        print("torch unavailable; skipping")
        return
    from scripts.bench_torch import run_case

    # Smallest possible case, 1 step — just confirm the tool runs end to end.
    run_case("smoke", 2, 2, 16, (4, 8), 16, steps=1)


if __name__ == "__main__":
    test_run_case_completes()
    print("ok  test_run_case_completes\n\n1 passed")
