"""Tests for the principled material library (the authored coupling)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble import materials as M  # noqa: E402


def test_names_are_sorted_and_stable():
    assert M.names() == tuple(sorted(M.names()))
    assert "ceramic" in M.names()


def test_every_material_is_well_formed():
    for name in M.names():
        mat = M.get(name)
        assert mat.physics.density > 0
        assert 0.0 <= mat.physics.restitution <= 1.0
        assert mat.physics.friction >= 0.0
        assert 0.0 <= mat.visual.roughness <= 1.0
        assert len(mat.visual.base_color) == 4


def test_coupling_is_physically_sensible():
    # The whole project rests on appearance and physics co-varying sensibly.
    steel = M.get("steel")
    foam = M.get("foam")
    rubber = M.get("rubber")
    ice = M.get("ice")
    glass = M.get("glass")

    assert steel.physics.density > foam.physics.density  # metal heavier than foam
    assert steel.visual.metallic > foam.visual.metallic  # ...and looks it
    assert rubber.physics.restitution > steel.physics.restitution  # rubber bounces
    assert ice.physics.friction < rubber.physics.friction  # ice is slippery
    assert glass.visual.transmission > steel.visual.transmission  # glass is clear


def test_physics_vector_matches_material():
    d, f, r = M.physics_vector("rubber")
    rubber = M.get("rubber")
    assert (d, f, r) == (
        rubber.physics.density,
        rubber.physics.friction,
        rubber.physics.restitution,
    )


def test_unknown_material_raises_with_help():
    try:
        M.get("unobtainium")
    except KeyError as exc:
        assert "unobtainium" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")


def test_validation_rejects_bad_values():
    from pseudomarble.materials import PhysicsProps, VisualProps

    for bad in (
        lambda: VisualProps((0, 0, 0, 1), roughness=2.0, metallic=0.0),
        lambda: PhysicsProps(density=-1, friction=0.5, restitution=0.5),
        lambda: PhysicsProps(density=1, friction=0.5, restitution=5.0),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
