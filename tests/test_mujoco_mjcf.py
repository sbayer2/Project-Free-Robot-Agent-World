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
    assemble_scene,
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


def test_lighting_modes_are_well_formed_and_distinct():
    """F21 Arm 2: 'oblique' adds specular lights + low ambient; 'flat' is the
    historical single top-down light. Both must parse; default is 'flat'."""
    steel = M.get("steel")
    flat = build_mjcf("box", steel)  # default
    oblique = build_mjcf("box", steel, lighting="oblique")
    ET.fromstring(flat)
    ET.fromstring(oblique)
    assert flat.count("<light ") == 1 and oblique.count("<light ") == 3
    assert 'ambient="0.4 0.4 0.4"' in flat        # historical high ambient
    assert 'ambient="0.18 0.18 0.18"' in oblique  # low ambient so highlights pop
    assert 'specular="0.6 0.6 0.6"' in oblique    # lights emit specular
    # default is byte-identical to explicit flat (old checkpoints/data reproducible)
    assert build_mjcf("box", steel, lighting="flat") == flat


def test_lighting_rejects_unknown_mode():
    try:
        build_mjcf("box", M.get("steel"), lighting="spooky")
        raise AssertionError("expected ValueError for unknown lighting mode")
    except ValueError:
        pass


def test_coupling_lives_in_one_geom():
    # density (-> mass), friction (physics), and material/rgba (appearance) must
    # all hang off the same object geom: the Marble idea in one element.
    steel = M.get("steel")
    xml = build_mjcf("box", steel)
    _, geom = _object_geom(xml)
    assert float(geom.get("density")) == steel.physics.density
    assert geom.get("friction").split()[0] == str(steel.physics.friction)
    assert geom.get("material") == "m_obj"


def test_drop_sets_object_height():
    xml = build_mjcf("box", M.get("wood"), object_z=1.2)
    root = ET.fromstring(xml)
    body = next(b for b in root.iter("body") if b.get("name") == "object")
    assert abs(float(body.get("pos").split()[2]) - 1.2) < 1e-6


def test_tilt_rotates_the_ground_plane():
    xml = build_mjcf("box", M.get("ice"), ground_euler=(0.35, 0.0, 0.0))
    root = ET.fromstring(xml)
    ground = next(g for g in root.iter("geom") if g.get("name") == "ground")
    assert abs(float(ground.get("euler").split()[0]) - 0.35) < 1e-6


def test_every_shape_has_half_height():
    for shape in DEFAULT_SHAPES:
        assert SHAPE_TO_GEOM[shape]["half_height"] > 0


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


def test_assemble_scene_joins_render_and_sim_without_mujoco():
    # The split pipeline's join seam: given frames (GPU phase) + behavior (CPU
    # phase) it must assemble a valid sample.json — pure-Python, no mujoco runtime.
    import json
    import tempfile

    from pseudomarble.config import PhysicsConfig, RenderConfig
    from pseudomarble.materials import MaterialSampler

    sample = MaterialSampler(seed=7).sample(material_id="t")
    frames = [{"index": 0, "file": "view_000.png"}]        # stand-in render output
    behavior = [{"probe": {"kind": "drop"}, "outcome": {"toppled": False}}]
    with tempfile.TemporaryDirectory() as d:
        rec = assemble_scene("train_000000", "box", sample, "train", d,
                             RenderConfig(resolution=64, num_views=1),
                             PhysicsConfig(), frames, behavior)
        # record carries both projections and the split label...
        assert rec["scene_id"] == "train_000000" and rec["split"] == "train"
        assert rec["input"]["shape"] == "box"
        assert rec["appearance"]["frames"] == frames      # render phase output
        assert rec["behavior"]["probes"] == behavior      # sim phase output
        # ...and it was written to <out>/<scene_id>/sample.json verbatim.
        with open(f"{d}/train_000000/sample.json") as fh:
            on_disk = json.load(fh)
        assert on_disk["scene_id"] == "train_000000"
        assert on_disk["behavior"]["probes"] == behavior


def test_topple_jitter_flags_parse():
    from pseudomarble.data.generate_mujoco import parse_args
    ns = parse_args(["--topple-jitter-reps", "8", "--topple-jitter-impulse", "0.05",
                     "--topple-jitter-azimuth", "1.5"])
    assert ns.topple_jitter_reps == 8
    assert ns.topple_jitter_impulse == 0.05 and ns.topple_jitter_azimuth == 1.5
    assert parse_args([]).topple_jitter_reps == 0   # default = hard binary toppled


def test_run_probes_soft_topple_without_mujoco():
    # Soft-topple averages the push label over action-jittered runs. We monkeypatch
    # the probe runners with a fake whose push topples iff impulse exceeds the base
    # 1.5, so impulse jitter straddling 1.5 yields a probability strictly in (0,1) —
    # exercising the F8 mitigation without a MuJoCo runtime.
    from pseudomarble.config import PhysicsConfig
    from pseudomarble.data import generate_mujoco as G
    from pseudomarble.materials import MaterialSampler

    def fake_runner(shape, material, spec, physics_cfg):
        toppled = getattr(spec, "impulse", 0.0) > 1.5
        up = [1.0, 0.0, 0.02] if toppled else [0.0, 0.0, 1.0]
        return [{"t": 0.0, "pos": [0.0, 0.0, 0.2], "up": [0.0, 0.0, 1.0]},
                {"t": 0.1, "pos": [0.0, 0.0, 0.2], "up": up}]

    saved = dict(G._PROBE_RUNNERS)
    try:
        for k in ("drop", "tilt", "push"):
            G._PROBE_RUNNERS[k] = fake_runner
        material = MaterialSampler(seed=3).sample(material_id="t").material
        recs = G.run_probes("cylinder", material,
                            PhysicsConfig(topple_jitter_reps=40, topple_jitter_impulse_rel=0.10))
        push = next(r for r in recs if r["probe"] == "push")
        tp = push["outcome"]["toppled"]
        assert isinstance(tp, float) and 0.0 < tp < 1.0   # smooth probability, not a bool
        assert push["spec"]["topple_jitter"] == {
            "reps": 40, "impulse_rel": 0.10, "azimuth_deg": 2.0, "n_samples": 41}

        # default (reps=0) keeps the hard binary label and adds no jitter metadata
        hard = G.run_probes("cylinder", material, PhysicsConfig())
        hpush = next(r for r in hard if r["probe"] == "push")
        assert isinstance(hpush["outcome"]["toppled"], bool)
        assert "topple_jitter" not in hpush["spec"]
    finally:
        G._PROBE_RUNNERS.clear()
        G._PROBE_RUNNERS.update(saved)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
