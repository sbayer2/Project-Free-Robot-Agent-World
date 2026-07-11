"""Tests for the ABO loader's pure core (``data/generate_abo.py``).

No trimesh/mujoco/network: weight normalization, listing parsing, the
deterministic gated selection, the density gate, the adaptive camera radius,
and the meta.json contract that ``generate_gso``'s parsers must recover.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.data.generate_abo import (  # noqa: E402
    DEFORMABLE_CATEGORIES,
    DENSITY_MAX,
    DENSITY_MIN,
    density_gate,
    parse_listing,
    select_models,
    weight_to_kg,
)
from pseudomarble.data.generate_gso import (  # noqa: E402
    auto_camera_radius,
    read_category,
    read_mass,
)


def _w(unit, value):
    return [{"normalized_value": {"unit": unit, "value": value},
             "unit": unit, "value": value}]


# --------------------------------------------------------------------------- #
# Weight normalization.
# --------------------------------------------------------------------------- #
def test_weight_units_normalize_to_kg():
    assert abs(weight_to_kg(_w("pounds", 2.0)) - 0.90718) < 1e-4
    assert weight_to_kg(_w("kilograms", 3.5)) == 3.5
    assert abs(weight_to_kg(_w("grams", 500)) - 0.5) < 1e-9
    assert abs(weight_to_kg(_w("ounces", 16)) - 0.45359) < 1e-4


def test_weight_rejects_junk():
    assert weight_to_kg(None) is None
    assert weight_to_kg([]) is None
    assert weight_to_kg(_w("furlongs", 3)) is None
    assert weight_to_kg(_w("pounds", 0)) is None          # non-positive
    assert weight_to_kg(_w("pounds", "heavy")) is None    # non-numeric


def test_weight_takes_first_usable_entry():
    entries = _w("furlongs", 3) + _w("pounds", 1.0)
    assert abs(weight_to_kg(entries) - 0.45359) < 1e-4


# --------------------------------------------------------------------------- #
# Listing parsing.
# --------------------------------------------------------------------------- #
def test_parse_listing_requires_model_and_weight():
    good = {"3dmodel_id": "B01", "item_weight": _w("pounds", 1),
            "product_type": [{"value": "CHAIR"}]}
    assert parse_listing(good) == {"model_id": "B01", "kg": 0.45359237,
                                   "category": "CHAIR"}
    assert parse_listing({"item_weight": _w("pounds", 1)}) is None   # no model
    assert parse_listing({"3dmodel_id": "B01"}) is None              # no weight


def test_parse_listing_unknown_category():
    r = parse_listing({"3dmodel_id": "B02", "item_weight": _w("kilograms", 1)})
    assert r["category"] == "UNKNOWN"


# --------------------------------------------------------------------------- #
# Selection: deterministic, gated, capped.
# --------------------------------------------------------------------------- #
def _models():
    out = {}
    for i in range(5):
        out[f"CH{i}"] = {"kg": 1.0, "category": "CHAIR"}
        out[f"TB{i}"] = {"kg": 2.0, "category": "TABLE"}
    out["RG0"] = {"kg": 1.0, "category": "RUG"}  # deformable -> gated out
    return out


def test_select_models_gates_deformables_and_is_deterministic():
    sel1, tally = select_models(_models(), target=6, cap_per_category=3)
    sel2, _ = select_models(_models(), target=6, cap_per_category=3)
    assert sel1 == sel2                              # deterministic
    assert not any(m.startswith("RG") for m in sel1)  # rug gated
    assert tally["rejected_deformable"] == 1
    assert tally["selected"] == 6
    # round-robin: alternating categories, lexicographic ids within each
    assert sel1 == ["CH0", "TB0", "CH1", "TB1", "CH2", "TB2"]


def test_select_models_respects_per_category_cap():
    sel, _ = select_models(_models(), target=100, cap_per_category=2)
    assert sum(1 for m in sel if m.startswith("CH")) == 2
    assert sum(1 for m in sel if m.startswith("TB")) == 2


def test_deformable_list_is_frozen():
    assert "RUG" in DEFORMABLE_CATEGORIES and "PILLOW" in DEFORMABLE_CATEGORIES


# --------------------------------------------------------------------------- #
# Density gate.
# --------------------------------------------------------------------------- #
def test_density_gate_bounds():
    ok, d = density_gate(mass_kg=10.0, volume_m3=0.05)   # 200 kg/m^3: furniture
    assert ok and abs(d - 200.0) < 1e-9
    ok, d = density_gate(mass_kg=1000.0, volume_m3=0.001)  # 1e6: unit error
    assert not ok and d > DENSITY_MAX
    ok, d = density_gate(mass_kg=0.001, volume_m3=1.0)     # 0.001: packaging air
    assert not ok and d < DENSITY_MIN


def test_density_gate_passes_non_watertight():
    ok, d = density_gate(mass_kg=5.0, volume_m3=None)
    assert ok and d is None


# --------------------------------------------------------------------------- #
# Contracts shared with the GSO pipeline.
# --------------------------------------------------------------------------- #
def test_meta_json_contract_readable_by_gso_parsers(tmp_path):
    obj = tmp_path / "B0TEST"
    obj.mkdir()
    (obj / "meta.json").write_text(json.dumps(
        {"mass_kg": 4.25, "category": "CHAIR", "mass_provenance": "abo_listing"}))
    mass, source = read_mass(str(obj))
    assert mass == 4.25 and source == "json"
    assert read_category(str(obj)) == "chair"


def test_auto_camera_radius_rule():
    assert auto_camera_radius(0.15, 2.6) == 2.6      # configured wins
    assert auto_camera_radius(1.0, 0.0) == 4.0       # adaptive: 4x half-height
    assert auto_camera_radius(0.0, 0.0) == 0.2       # floor at 0.05 half-height
