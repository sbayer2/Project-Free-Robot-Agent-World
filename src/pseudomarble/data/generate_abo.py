"""ABO (Amazon Berkeley Objects) generation — real products with listed weights.

The GSO experiment's gate 1 (measured mass) admitted 0/1033 objects from the
canonical Fuel distribution (docs/GSO_EXPERIMENT.md, 2026-07-11 amendment), so
the reality source is ABO: ~8k real products with 3D meshes (GLB) and listing
metadata that includes **item weight** for ~89% of modeled products —
manufacturer-listed rather than lab-measured, but real, object-specific, and
not derivable from geometry.

This module is a thin front-end to the GSO pipeline: it PREPARES object
directories that satisfy ``generate_gso.discover_objects``'s contract
(``meshes/model.obj`` + ``materials/textures/texture.png`` + ``meta.json``
with ``mass_kg``/``category``), then delegates generation — rendering,
drop/tilt/push probes, category holdout, manifest — to ``generate_gso.main``.
One pipeline, two reality sources.

Gates (fixed before selection; every rejection tallied, none silent):
  * usable listed weight (normalized_value parse > 0),
  * not in ``DEFORMABLE_CATEGORIES`` (rigid-body sim would mislabel a rug),
  * GLB loads and converts (trimesh),
  * density sanity when the mesh is watertight: mass / volume inside
    [``DENSITY_MIN``, ``DENSITY_MAX``] kg/m^3 — catches listing errors and
    shipping-weight-includes-packaging cases.

Selection is deterministic (no hand-picking): categories in alphabetical
order, round-robin, lexicographically smallest ids first, per-category cap.

Pure-Python parts (weight normalization, listing parsing, selection, density
gate) are unit-tested anywhere; GLB conversion needs ``trimesh`` and running
the pipeline needs ``mujoco`` (install ``.[gso]``).

Usage::

    python -m pseudomarble.data.generate_abo \
        --listings-dir ~/datasets/abo/listings/metadata \
        --glb-dir ~/datasets/abo/glb --prepared-root ~/datasets/abo/prepared \
        --output data/pm_abo --target 400 --camera-radius 0
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Pure: listing parsing, gates, selection.
# --------------------------------------------------------------------------- #
WEIGHT_TO_KG: Dict[str, float] = {
    "pounds": 0.45359237,
    "kilograms": 1.0,
    "grams": 1e-3,
    "milligrams": 1e-6,
    "ounces": 0.028349523,
    "tons": 907.18474,  # US short ton (Amazon listing convention)
}

# Product types whose objects are grossly deformable: rigid-body MuJoCo would
# produce FALSE behavior labels (a rigid rug). Fixed before selection.
DEFORMABLE_CATEGORIES = frozenset({
    "RUG", "PILLOW", "CUSHION", "BLANKET", "QUILT", "CURTAIN", "TOWEL",
    "MATTRESS", "BED_LINEN_SET", "SLEEPING_BAG", "TAPESTRY",
    # added after the generation smoke surfaced leaks (2026-07-11):
    "EXERCISE_MAT", "YOGA_MAT", "MOUSE_PAD",
})

# Effective-density plausibility band, kg/m^3. Lower bound admits hollow/foamy
# furniture (~20); upper bound is beyond solid steel (~7900) to admit dense
# small goods while catching unit errors (e.g. grams read as kilograms).
DENSITY_MIN = 10.0
DENSITY_MAX = 20000.0


def weight_to_kg(item_weight: Optional[List[Dict]]) -> Optional[float]:
    """Listed item weight in kg from an ABO ``item_weight`` list, else None."""
    for entry in item_weight or []:
        nv = entry.get("normalized_value", entry)
        unit, value = nv.get("unit"), nv.get("value")
        if unit in WEIGHT_TO_KG and value is not None:
            try:
                kg = float(value) * WEIGHT_TO_KG[unit]
            except (TypeError, ValueError):
                continue
            if kg > 0:
                return kg
    return None


def parse_listing(record: Dict) -> Optional[Dict]:
    """{model_id, kg, category} for a listing with a 3D model + usable weight."""
    mid = record.get("3dmodel_id")
    if not mid:
        return None
    kg = weight_to_kg(record.get("item_weight"))
    if kg is None:
        return None
    pt = record.get("product_type") or [{}]
    category = (pt[0].get("value") or "UNKNOWN").strip().upper()
    return {"model_id": mid, "kg": kg, "category": category}


def load_listings(listings_dir: str) -> Dict[str, Dict]:
    """Scan ABO ``listings_*.json.gz`` shards -> {model_id: {kg, category}}.
    First usable weight per model wins (duplicates across marketplaces)."""
    models: Dict[str, Dict] = {}
    for path in sorted(glob.glob(os.path.join(listings_dir, "listings_*.json.gz"))):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                parsed = parse_listing(json.loads(line))
                if parsed and parsed["model_id"] not in models:
                    models[parsed["model_id"]] = {
                        "kg": parsed["kg"], "category": parsed["category"]}
    return models


def select_models(models: Dict[str, Dict], target: int = 450,
                  cap_per_category: int = 12) -> Tuple[List[str], Dict[str, int]]:
    """Deterministic gated selection: drop deformable categories, then take
    round-robin across alphabetical categories (lexicographic ids within each)
    up to ``cap_per_category`` per category and ``target`` total.

    Returns (selected ids, rejection/selection tally)."""
    tally = collections.Counter()
    by_cat: Dict[str, List[str]] = collections.defaultdict(list)
    for mid, meta in models.items():
        if meta["category"] in DEFORMABLE_CATEGORIES:
            tally["rejected_deformable"] += 1
        else:
            by_cat[meta["category"]].append(mid)
    for ids in by_cat.values():
        ids.sort()
    selected: List[str] = []
    rank = 0
    while len(selected) < target and rank < cap_per_category:
        for cat in sorted(by_cat):
            ids = by_cat[cat]
            if rank < min(cap_per_category, len(ids)) and len(selected) < target:
                selected.append(ids[rank])
        rank += 1
    tally["candidates"] = sum(len(v) for v in by_cat.values())
    tally["selected"] = len(selected)
    return selected, dict(tally)


def density_gate(mass_kg: float, volume_m3: Optional[float]) -> Tuple[bool, Optional[float]]:
    """(passes, density). Non-watertight meshes (volume None) pass with
    density None — mass is still a real label; only a computable, implausible
    density rejects (listing/unit errors, packaging weights)."""
    if volume_m3 is None or volume_m3 <= 0:
        return True, None
    density = mass_kg / volume_m3
    return (DENSITY_MIN <= density <= DENSITY_MAX), density


# --------------------------------------------------------------------------- #
# Runtime: GLB -> object dir satisfying generate_gso's discovery contract.
# --------------------------------------------------------------------------- #
def prepare_object(glb_path: str, mass_kg: float, category: str,
                   obj_dir: str) -> Optional[str]:
    """Convert one GLB into a GSO-shaped object dir. Returns a rejection
    reason string, or None on success. Needs trimesh."""
    import trimesh  # type: ignore

    if os.path.exists(os.path.join(obj_dir, "meta.json")):
        return None  # already prepared (resumable)
    try:
        mesh = trimesh.load(glb_path, force="mesh")
    except Exception as e:  # noqa: BLE001 - external data; tally and skip
        return f"glb_load_failed:{type(e).__name__}"
    if mesh.is_empty or len(mesh.faces) == 0:
        return "empty_mesh"

    volume = float(abs(mesh.volume)) if mesh.is_watertight else None
    ok, density = density_gate(mass_kg, volume)
    if not ok:
        return f"density_implausible:{density:.1f}"
    if volume is None:
        # Non-watertight (all of ABO in practice): the convex-hull volume is an
        # UPPER bound on true volume, so mass/hull is a LOWER bound on density —
        # an implausibly high lower bound still catches unit errors.
        try:
            hull_vol = float(mesh.convex_hull.volume)
        except Exception:  # noqa: BLE001 - hull can fail on degenerate meshes
            hull_vol = 0.0
        if hull_vol > 0 and (mass_kg / hull_vol) > DENSITY_MAX:
            return f"density_implausible_lb:{mass_kg / hull_vol:.1f}"
        density = None if hull_vol <= 0 else mass_kg / hull_vol  # lower bound

    os.makedirs(os.path.join(obj_dir, "meshes"), exist_ok=True)
    tex_dir = os.path.join(obj_dir, "materials", "textures")
    # Bake the base-color texture out of the GLB, if present.
    texture = getattr(getattr(mesh, "visual", None), "material", None)
    img = getattr(texture, "baseColorTexture", None)
    if img is not None:
        os.makedirs(tex_dir, exist_ok=True)
        try:
            img.save(os.path.join(tex_dir, "texture.png"))
        except Exception:  # noqa: BLE001 - texture is optional
            pass
    try:
        mesh.export(os.path.join(obj_dir, "meshes", "model.obj"))
    except Exception as e:  # noqa: BLE001
        return f"obj_export_failed:{type(e).__name__}"
    with open(os.path.join(obj_dir, "meta.json"), "w") as fh:
        json.dump({"mass_kg": mass_kg, "category": category,
                   "mass_provenance": "abo_listing", "watertight": volume is not None,
                   "density_kg_m3": density if volume is not None else None,
                   "density_lower_bound_kg_m3": None if volume is not None else density},
                  fh, indent=2)
    return None


def prepare_all(glb_dir: str, models: Dict[str, Dict], selected: List[str],
                prepared_root: str) -> Dict[str, int]:
    """Prepare every selected object; returns the rejection tally (printed AND
    persisted — the design doc forbids silent drops)."""
    tally = collections.Counter()
    for mid in selected:
        glb = os.path.join(glb_dir, f"{mid}.glb")
        if not os.path.exists(glb):
            tally["rejected_glb_missing"] += 1
            continue
        reason = prepare_object(glb, models[mid]["kg"], models[mid]["category"],
                                os.path.join(prepared_root, mid))
        if reason is None:
            tally["prepared"] += 1
        else:
            tally["rejected_" + reason.split(":")[0]] += 1
            print(f"[abo] reject {mid}: {reason}", flush=True)
    return dict(tally)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pseudo-marble ABO (real-product) generation")
    p.add_argument("--listings-dir", required=True,
                   help="dir of ABO listings_*.json.gz shards")
    p.add_argument("--glb-dir", required=True, help="dir of <model_id>.glb meshes")
    p.add_argument("--prepared-root", required=True,
                   help="cache dir for converted object dirs (resumable)")
    p.add_argument("--output", default="data/pm_abo")
    p.add_argument("--target", type=int, default=450, help="selection size before gates")
    p.add_argument("--cap-per-category", type=int, default=12)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--views", type=int, default=8)
    p.add_argument("--camera-radius", type=float, default=0.0,
                   help="<=0 = adaptive to object size (metre-scale furniture)")
    p.add_argument("--collision-method", default="coacd")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--prepare-only", action="store_true",
                   help="stop after preparation (no rendering/sim)")
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    from pseudomarble.data import generate_gso

    args = parse_args(argv)
    models = load_listings(args.listings_dir)
    selected, sel_tally = select_models(models, args.target, args.cap_per_category)
    print(f"[abo] listings: {len(models)} weighted models; selection tally: {sel_tally}")
    prep_tally = prepare_all(args.glb_dir, models, selected, args.prepared_root)
    print(f"[abo] preparation tally: {prep_tally}")
    with open(os.path.join(args.prepared_root, "abo_tally.json"), "w") as fh:
        json.dump({"selection": sel_tally, "preparation": prep_tally}, fh, indent=2)
    if args.prepare_only:
        return
    generate_gso.main([
        "--gso-root", args.prepared_root, "--output", args.output,
        "--resolution", str(args.resolution), "--views", str(args.views),
        "--camera-radius", str(args.camera_radius),
        "--collision-method", args.collision_method,
        "--holdout-frac", str(args.holdout_frac),
        "--seed", str(args.seed), "--workers", str(args.workers),
    ])


if __name__ == "__main__":
    main(sys.argv[1:])
