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

Phase-aware widths (the unified-memory point): rendering is GPU-bound and the Mac
has ONE GPU sharing the 64 GB / 307 GB-s pool with the CPU, while the drop/tilt/
push simulation is pure CPU (``mj_step``). Those have *opposite* optimal worker
counts, so a single number can't serve both — ``default_render_workers`` stays
small (don't oversubscribe the one GPU / the shared bus) while ``default_cpu_workers``
spreads across the performance cores. The primary generator runs render and sim as
two phases, each at its own width; ``resolve_workers(default=...)`` picks the auto
target per phase.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence


def default_cpu_workers(reserve: int = 2) -> int:
    """Auto width for a CPU-bound phase (e.g. MuJoCo ``mj_step`` simulation).

    Use most cores, leaving a little headroom for the OS / IO. On a unified-memory
    Mac the GPU is idle during a pure-sim phase, so spreading across the
    performance cores is fine — the work is genuinely CPU-parallel.
    """
    n = os.cpu_count() or 1
    return max(1, n - max(0, reserve))


def default_render_workers(cap: int = 4) -> int:
    """Auto width for a GPU-bound phase (Metal off-screen rendering).

    KEY unified-memory caveat: there is ONE GPU, and it shares the memory bus with
    the CPU. Spawning a worker per core just makes N processes queue on the same
    GPU and fight for bandwidth — more processes, not more throughput. A small pool
    keeps the GPU fed (overlapping CPU-side scene setup + PNG encode/IO while it
    draws) without oversubscribing it. Tune up explicitly only if the GPU is
    measurably underused; the real optimum is found by benchmarking on the Mac.
    """
    n = os.cpu_count() or 1
    return max(1, min(cap, n))


def resolve_workers(requested: Optional[int], n_items: int,
                    default: Optional[int] = None) -> int:
    """How many worker processes to use.

    ``requested`` > 0 is honored (clamped); ``0``/``None`` ⇒ auto. ``default`` sets
    the auto target — pass ``default_cpu_workers()`` for CPU-bound phases or
    ``default_render_workers()`` for GPU-bound ones; when omitted, auto falls back
    to ``os.cpu_count()``. The result is always clamped to ``[1, max(1, n_items)]``
    — never more workers than items, never below 1.
    """
    n_items = max(1, n_items)
    if requested and requested > 0:
        return max(1, min(requested, n_items))
    auto = default if (default and default > 0) else (os.cpu_count() or 1)
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
