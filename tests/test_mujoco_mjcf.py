"""Tests for the MuJoCo MJCF builder (pure-Python; no mujoco runtime needed).

These validate that the appearance<->physics coupling is faithfully written into
a single geom, and that the XML is well-formed — without requiring the mujoco
package, which is Apple-silicon-only and not present in generic CI.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble import materials as M  # noqa: E402
from pseudomarble.data.generate_mujoco import (  # noqa: E402
    DEFAULT_SHAPES,
    SHAPE_TO_GEOM,
    build_mjcf,
)


def _object_geom(xml: str):
    root = ET.fromstring(xml)
    for geom in root.iter("geom"):
        if geom.get("name") == "obj":
            return root, geom
    raise AssertionError("no object geom found")


def test_mjcf_is_well_formed_for_every_default_shape():
    for shape in DEFAULT_SHAPES:
        xml = build_mjcf(shape, M.get("steel"))
        ET.fromstring(xml)  # raises if malformed


def test_coupling_lives_in_one_geom():
    # density (-> mass), friction (physics), and material/rgba (appearance) must
    # all hang off the same object geom: the Marble idea in one element.
    steel = M.get("steel")
    xml = build_mjcf("box", steel)
    _, geom = _object_geom(xml)
    assert float(geom.get("density")) == steel.physics.density
    assert geom.get("friction").split()[0] == str(steel.physics.friction)
    assert geom.get("material") == f"m_{steel.name}"


def test_density_drives_mass_difference():
    # Two materials with different density yield different density attributes;
    # MuJoCo computes mass = density * volume from this.
    steel = float(_object_geom(build_mjcf("box", M.get("steel")))[1].get("density"))
    foam = float(_object_geom(build_mjcf("box", M.get("foam")))[1].get("density"))
    assert steel > foam


def test_restitution_maps_monotonically_to_solref_damping():
    # Bouncier material -> less contact damping (second solref term smaller).
    def damping(mat):
        _, geom = _object_geom(build_mjcf("sphere", M.get(mat)))
        return abs(float(geom.get("solref").split()[1]))

    assert damping("rubber") < damping("foam")  # rubber bounces, foam does not


def test_unknown_shape_is_rejected_with_guidance():
    try:
        build_mjcf("teapot", M.get("wood"))
    except ValueError as exc:
        assert "mesh asset" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-primitive shape")


def test_every_default_shape_is_known():
    for shape in DEFAULT_SHAPES:
        assert shape in SHAPE_TO_GEOM


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
