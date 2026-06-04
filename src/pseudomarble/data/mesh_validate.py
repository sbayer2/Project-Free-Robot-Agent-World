"""Mesh sanity checks before a mesh is allowed into the dataset.

Objaverse (and most in-the-wild asset sets) are full of non-manifold,
non-watertight, double-sided meshes. For those, *volume* — and therefore mass
from ``density * volume`` — is undefined, and convex-decomposition can misbehave.
We gate on watertightness so the physics ground truth is actually meaningful.

``trimesh`` is an optional dependency: if it is missing we return a report that
says "unknown" rather than crashing, so the package stays importable in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MeshReport:
    path: str
    available: bool  # was trimesh importable / mesh loadable?
    watertight: Optional[bool] = None
    winding_consistent: Optional[bool] = None
    volume: Optional[float] = None  # m^3 (only trustworthy if watertight)
    n_vertices: Optional[int] = None
    n_faces: Optional[int] = None
    note: str = ""

    @property
    def usable(self) -> bool:
        """A mesh is usable for physics ground truth only if watertight."""
        return bool(self.available and self.watertight)


def validate_mesh(path: str) -> MeshReport:
    try:
        import trimesh  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on env
        return MeshReport(path=path, available=False, note=f"trimesh unavailable: {exc}")

    try:
        mesh = trimesh.load(path, force="mesh")
    except Exception as exc:
        return MeshReport(path=path, available=False, note=f"load failed: {exc}")

    if mesh.is_empty or len(mesh.faces) == 0:
        return MeshReport(path=path, available=True, watertight=False, note="empty mesh")

    watertight = bool(mesh.is_watertight)
    return MeshReport(
        path=path,
        available=True,
        watertight=watertight,
        winding_consistent=bool(mesh.is_winding_consistent),
        volume=float(abs(mesh.volume)) if watertight else None,
        n_vertices=int(len(mesh.vertices)),
        n_faces=int(len(mesh.faces)),
        note="" if watertight else "not watertight -> volume/mass undefined",
    )


def mass_from_mesh(path: str, density_kg_m3: float) -> Optional[float]:
    """Mass = density * volume, in kg. Returns None if the mesh is not usable.

    NOTE: this assumes *uniform* density and a watertight mesh. Hollow or
    composite objects (the canonical cup!) violate the uniform-density
    assumption; we accept that as a known limitation of the synthetic substrate.
    """
    report = validate_mesh(path)
    if not report.usable or report.volume is None:
        return None
    return density_kg_m3 * report.volume
