"""Process-parallel scheduling for data generation (generator-agnostic).

Each generated scene is wholly self-contained — it builds its own MuJoCo
model/render/sim context and writes its own output directory — so a dataset is
*embarrassingly parallel* across scenes. This module holds only the scheduler:
how many workers to use, and an order-preserving parallel map. It is pure stdlib
(no MuJoCo), so it is unit-testable in any session; the per-generator worker (what
one scene actually *is*) lives with its generator.

Processes, not threads, on purpose: a MuJoCo render/sim context is per-process and
not thread-safe, the work is CPU/GL-bound, and the GIL would serialize threads
anyway. ``ProcessPoolExecutor`` uses the 'spawn' start method on macOS, so worker
callables and their arguments must be picklable — keep workers at module level and
pass dataclass/tuple payloads.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence


def resolve_workers(requested: Optional[int], n_items: int) -> int:
    """How many worker processes to use.

    ``requested`` 0/None ⇒ auto = ``os.cpu_count()`` (one process per core — on the
    target M5 that's ~18-way); otherwise the explicit value. Always clamped to
    ``[1, max(1, n_items)]`` — never more workers than scenes, never below 1.
    """
    n_items = max(1, n_items)
    if requested and requested > 0:
        return max(1, min(requested, n_items))
    auto = os.cpu_count() or 1
    return max(1, min(auto, n_items))


def ordered_parallel_map(fn: Callable, items: Sequence, workers: int,
                         on_done: Optional[Callable[[int, object], None]] = None) -> List:
    """Map ``fn`` over ``items`` across ``workers`` processes, results in INPUT order.

    ``workers <= 1`` runs serially (no pool spawned — keeps small runs and the
    no-MuJoCo tests simple). ``on_done(i, result)`` fires on the *parent* as each
    item finishes (so progress prints don't interleave across workers), while the
    returned list stays in submission order regardless of finish order.
    """
    items = list(items)
    if workers <= 1:
        results: List = []
        for i, x in enumerate(items):
            r = fn(x)
            if on_done is not None:
                on_done(i, r)
            results.append(r)
        return results

    import concurrent.futures as cf

    results = [None] * len(items)
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        fut_to_idx = {ex.submit(fn, x): i for i, x in enumerate(items)}
        for fut in cf.as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            results[i] = fut.result()
            if on_done is not None:
                on_done(i, results[i])
    return results
