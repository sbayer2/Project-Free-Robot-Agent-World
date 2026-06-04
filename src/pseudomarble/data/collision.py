"""Collision-shape generation for the physics path.

Deliberately *not* a convex hull by default. The canonical object in this whole
project is a cup, and a convex hull of a cup is a solid lump: no cavity, no rim
to grasp, no interior to hold liquid. Using convex-hull collision would throw
away exactly the concavity that distinguishes "container" from "blob".

So the default is an approximate **convex decomposition** (CoACD or V-HACD): a
union of convex pieces that preserves concavity while remaining cheap and stable
for the rigid-body solver. We fall back to a single convex hull only when no
decomposition backend is installed — and we warn loudly when we do, because that
fallback silently changes what the physics means.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CollisionResult:
    method: str  # "coacd" | "vhacd" | "convex_hull"
    n_parts: int
    part_paths: List[str] = field(default_factory=list)
    degraded: bool = False  # True if we fell back from the requested method
    note: str = ""


def compute_collision(
    mesh_path: str,
    out_dir: str,
    method: str = "coacd",
    max_hulls: int = 16,
) -> CollisionResult:
    """Produce a collision representation for ``mesh_path`` under ``out_dir``.

    Tries the requested decomposition method, then the other decomposition
    backend, then a convex hull as a last resort (with a warning).
    """
    method = method.lower()
    order = _fallback_order(method)
    last_note = ""
    for m in order:
        try:
            if m == "coacd":
                return _coacd(mesh_path, out_dir, max_hulls, degraded=m != method)
            if m == "vhacd":
                return _vhacd(mesh_path, out_dir, max_hulls, degraded=m != method)
            if m == "convex_hull":
                if method != "convex_hull":
                    warnings.warn(
                        "falling back to CONVEX HULL collision: concavity will be "
                        "lost (a cup becomes a solid lump). Install 'coacd' or "
                        "'trimesh[easy]' for faithful collision shapes.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                return _convex_hull(mesh_path, out_dir, degraded=method != "convex_hull")
        except Exception as exc:  # try next backend
            last_note = f"{m} failed: {exc}"
            continue
    raise RuntimeError(f"no collision backend succeeded; last: {last_note}")


def _fallback_order(method: str) -> List[str]:
    if method == "convex_hull":
        return ["convex_hull"]
    if method == "vhacd":
        return ["vhacd", "coacd", "convex_hull"]
    return ["coacd", "vhacd", "convex_hull"]


def _coacd(mesh_path: str, out_dir: str, max_hulls: int, degraded: bool) -> CollisionResult:
    import os

    import coacd  # type: ignore
    import trimesh  # type: ignore

    mesh = trimesh.load(mesh_path, force="mesh")
    cmesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(cmesh, max_convex_hull=max_hulls)
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    for i, (verts, faces) in enumerate(parts):
        p = os.path.join(out_dir, f"hull_{i:03d}.obj")
        trimesh.Trimesh(verts, faces).export(p)
        paths.append(p)
    return CollisionResult("coacd", len(paths), paths, degraded=degraded)


def _vhacd(mesh_path: str, out_dir: str, max_hulls: int, degraded: bool) -> CollisionResult:
    import os

    import trimesh  # type: ignore

    mesh = trimesh.load(mesh_path, force="mesh")
    # trimesh exposes V-HACD via the convex decomposition helper when the
    # testVHACD binary is on PATH.
    parts = mesh.convex_decomposition(maxConvexHulls=max_hulls)
    if not isinstance(parts, list):
        parts = [parts]
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, part in enumerate(parts):
        p = os.path.join(out_dir, f"hull_{i:03d}.obj")
        part.export(p)
        paths.append(p)
    return CollisionResult("vhacd", len(paths), paths, degraded=degraded)


def _convex_hull(mesh_path: str, out_dir: str, degraded: bool) -> CollisionResult:
    import os

    import trimesh  # type: ignore

    mesh = trimesh.load(mesh_path, force="mesh")
    hull = mesh.convex_hull
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "hull_000.obj")
    hull.export(p)
    return CollisionResult(
        "convex_hull",
        1,
        [p],
        degraded=degraded,
        note="concavity discarded",
    )
