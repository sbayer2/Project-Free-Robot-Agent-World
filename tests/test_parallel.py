"""Tests for the data-generation parallel scheduler (``data/parallel.py``).

Pure stdlib — no MuJoCo — so the fan-out logic is verified in any session. Worker
callables must be MODULE-LEVEL to be picklable under the 'spawn' start method
(macOS), which is exactly the constraint the real generators honor.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.data.parallel import (  # noqa: E402
    default_cpu_workers,
    default_render_workers,
    ordered_parallel_map,
    resolve_workers,
)


# Module-level (picklable) workers — stand-ins for build_scene.
def _square(x):
    return x * x


def _square_with_pid(x):
    return (x * x, os.getpid())


def test_resolve_workers_auto_uses_cpu_count_clamped_to_items():
    # 0/None ⇒ auto = cpu_count(), but never more workers than items.
    auto = os.cpu_count() or 1
    assert resolve_workers(0, 1000) == auto
    assert resolve_workers(None, 1000) == auto
    assert resolve_workers(0, 1) == 1            # clamped down to #items
    assert resolve_workers(0, 0) == 1            # never zero workers


def test_resolve_workers_explicit_is_clamped():
    assert resolve_workers(3, 1000) == 3
    assert resolve_workers(99, 4) == 4           # not more workers than items
    # A nonsense negative falls through to the auto path (same as 0/None).
    assert resolve_workers(-5, 8) == resolve_workers(0, 8)
    assert resolve_workers(1, 8) == 1


def test_resolve_workers_honors_phase_default():
    # default= sets the auto target when requested is 0/None; explicit still wins.
    assert resolve_workers(0, 1000, default=3) == 3
    assert resolve_workers(None, 1000, default=5) == 5
    assert resolve_workers(0, 2, default=8) == 2          # clamped to #items
    assert resolve_workers(7, 1000, default=3) == 7       # explicit overrides default
    # A nonsense default falls back to cpu_count (same as no default given).
    assert resolve_workers(0, 1000, default=0) == resolve_workers(0, 1000)


def test_render_default_is_capped_and_narrower_on_real_machines():
    # The GPU-bound phase must not oversubscribe the one shared GPU: its auto width
    # is capped small regardless of core count.
    cpu = default_cpu_workers()
    gpu = default_render_workers()
    assert gpu >= 1 and cpu >= 1
    assert gpu <= 4                         # capped small no matter how many cores
    assert cpu <= (os.cpu_count() or 1)     # never claims more than the machine has
    # On a real many-core target (e.g. the 18-core M5), render is strictly narrower
    # than sim — the whole point of splitting the phases.
    if (os.cpu_count() or 1) >= 8:
        assert gpu < cpu


def test_cpu_default_reserve_leaves_headroom():
    n = os.cpu_count() or 1
    assert default_cpu_workers(reserve=0) == n
    assert default_cpu_workers(reserve=2) == max(1, n - 2)
    assert default_cpu_workers(reserve=10_000) == 1        # never below 1


def test_serial_map_preserves_order_and_reports_progress():
    items = [1, 2, 3, 4, 5]
    seen = []
    out = ordered_parallel_map(_square, items, workers=1,
                               on_done=lambda i, r: seen.append((i, r)))
    assert out == [1, 4, 9, 16, 25]
    # on_done fires once per item, in submission order for the serial path.
    assert seen == [(0, 1), (1, 4), (2, 9), (3, 16), (4, 25)]


def test_parallel_map_preserves_input_order():
    # Results must come back in INPUT order regardless of finish order.
    items = list(range(12))
    out = ordered_parallel_map(_square, items, workers=4)
    assert out == [x * x for x in items]


def test_parallel_map_actually_uses_child_processes():
    items = list(range(8))
    results = ordered_parallel_map(_square_with_pid, items, workers=2)
    squares = [r[0] for r in results]
    pids = [r[1] for r in results]
    assert squares == [x * x for x in items]
    # Every unit of work ran in a child process, not the parent.
    assert all(pid != os.getpid() for pid in pids)


def test_parallel_map_fires_on_done_for_every_item():
    items = list(range(10))
    done_idx = []
    out = ordered_parallel_map(_square, items, workers=3,
                               on_done=lambda i, r: done_idx.append(i))
    assert out == [x * x for x in items]
    assert sorted(done_idx) == list(range(10))   # each index reported exactly once


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
