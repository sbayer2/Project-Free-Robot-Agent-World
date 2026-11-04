"""Pins for the F35 battery generator (scripts/f35_generate_battery.py): the
ramp body sits flush on the tilted plane, the recording names are stable,
sampler mode is deterministic, and (where MuJoCo is installed) one scene runs
every probe on generate_mujoco's 2 s / 60 fps cadence."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import f35_generate_battery as G  # noqa: E402


def test_plane_normal_matches_mjcf_euler_about_x():
    n = G.plane_normal(30.0)
    assert np.isclose(np.linalg.norm(n), 1.0)
    assert np.allclose(n, [0.0, -0.5, math.cos(math.radians(30.0))])
    assert np.allclose(G.plane_normal(0.0), [0.0, 0.0, 1.0])


def test_ramp_angles_sit_below_the_cube_tipping_angle_and_ascend():
    assert list(G.RAMP_DEG) == sorted(G.RAMP_DEG) and max(G.RAMP_DEG) < 45.0


def test_probe_names_are_stable():
    names = G.probe_names()
    assert names[0] == "drop" and names[-3:] == ["kick1.0", "kick3.0", "dropramp20.0"]
    assert len(names) == 1 + len(G.RAMP_DEG) + len(G.PUSH_N) + 2 + 2 + 2 + len(G.KICK_MPS) + 1
    assert "ramp05.7" in names and "push0015" in names


def test_scale_spring_sinks_the_heaviest_box_within_the_plate_stroke():
    heaviest = 8000.0 * 0.3 ** 3                      # kg: top of DENSITY_RANGE, 0.3 m cube
    sink = (heaviest + G.SCALE_PLATE_MASS) * 9.81 / G.SCALE_K
    assert 0.3 < sink                                     # visible
    assert G.SCALE_Z - G.SCALE_PLATE_HALF - sink > 0.02   # the plate stays off the ground
    assert G.SCALE_PLATE_HALF >= 0.05                     # D4: too thick to tunnel through


def test_replace_once_refuses_ambiguity():
    assert G.replace_once("a<x>b", "<x>", "y") == "ayb"
    with pytest.raises(RuntimeError):
        G.replace_once("<x><x>", "<x>", "y")


def test_sampled_materials_are_deterministic():
    a = [m.physics for _, m in G.sampled_materials(7, 5)]
    b = [m.physics for _, m in G.sampled_materials(7, 5)]
    assert a == b and len({p.friction for p in a}) == 5


def test_one_scene_runs_every_probe():
    pytest.importorskip("mujoco")
    sid, norm, raw, out = G.simulate_scene(G.sampled_materials(0, 1)[0])
    assert sid == "sampled_000000" and len(norm) == len(raw) == 3
    assert set(out) == set(G.probe_names())
    for rec in out.values():
        assert len(rec) == 250 and len(rec[0]) == 7          # D5: 125 fps x 2 s


def test_ball_ground_pair_fixes_the_balls_ground_contact():
    """D5: the ball's ground contacts carry no material property of the box."""
    mujoco = pytest.importorskip("mujoco")
    from pseudomarble.data.generate_mujoco import build_mjcf, half_height

    _, mat = G.sampled_materials(0, 1)[0]
    h = half_height(G.SHAPE)
    ball = G._ball_xml("ball", (-(h + G.STRIKE_GAP + G.STRIKE_R), 0.0, h),
                       G.STRIKE_R, G.STRIKE_MASS, mat)
    m = mujoco.MjModel.from_xml_string(G.with_ball(build_mjcf(G.SHAPE, mat), ball))
    assert m.npair == 1
    assert {m.pair_geom1[0], m.pair_geom2[0]} == {m.geom("ball_geom").id, m.geom("ground").id}
    assert np.isclose(m.pair_friction[0, 0], G.BALL_GROUND_FRICTION)
    assert np.allclose(m.pair_solref[0], [float(v) for v in G.BALL_GROUND_SOLREF.split()])


def test_repo_contact_model_is_the_identity():
    mat = G.sampled_materials(0, 1)[0][1]
    try:
        G.set_contact("repo")
        xml = "<mujoco><option timestep='0.002'/></mujoco>"
        assert G.contact_xml(xml, mat) == xml
        with pytest.raises(ValueError):
            G.set_contact("repo", impratio=10.0)
        with pytest.raises(ValueError):
            G.set_contact("pyramidal")
    finally:
        G.set_contact(*G.DESIGNS["D7"][0])


def test_elliptic_contact_model_sets_cone_and_box_ground_pair():
    mujoco = pytest.importorskip("mujoco")
    mat = G.sampled_materials(0, 1)[0][1]
    try:
        G.set_contact("elliptic", impratio=10.0)
        m = mujoco.MjModel.from_xml_string(G.mjcf(mat))
        assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and m.opt.impratio == 10.0
        assert m.npair == 0
        G.set_contact("elliptic", solreffriction="0.01 1")
        m = mujoco.MjModel.from_xml_string(G.mjcf(mat))
        assert m.npair == 1
        assert {m.pair_geom1[0], m.pair_geom2[0]} == {m.geom("obj").id, m.geom("ground").id}
        assert np.allclose(m.pair_solreffriction[0], [0.01, 1.0])
        assert np.allclose(m.pair_solref[0], m.geom_solref[m.geom("obj").id])
        assert np.isclose(m.pair_friction[0, 0], mat.physics.friction)
    finally:
        G.set_contact(*G.DESIGNS["D7"][0])


def test_d7_is_the_default_design_and_d6_stays_reachable():
    """D7: elliptic cone + fixed friction solref, dropramp 20 deg from 0.3 m;
    the D6 preset keeps the repo contact model, 38.7 deg and 0.6 m."""
    assert G.design_name() == "D7"
    assert (G.CONTACT["model"], G.CONTACT["solreffriction"]) == ("elliptic", "0.01 1")
    assert (G.DROPRAMP_DEG, G.DROPRAMP_HEIGHT) == (20.0, 0.3)
    assert G.DESIGNS["D6"] == (("repo", 1.0, ""), 38.7, 0.6)
    try:
        G.init_worker(*G.DESIGNS["D6"])
        assert G.design_name() == "D6" and G.probe_names()[-1] == "dropramp38.7"
        G.set_dropramp(38.7, 0.15)
        assert G.design_name() == "pilot"
        with pytest.raises(ValueError):
            G.set_dropramp(20.0, 0.0)
    finally:
        G.init_worker(*G.DESIGNS["D7"])
    assert G.design_name() == "D7"


def test_provenance_records_versions_and_commit():
    pytest.importorskip("mujoco")
    import mujoco
    p = G.provenance()
    assert {"mujoco", "numpy", "python", "platform", "git_commit", "git_dirty",
            "canonical_mujoco", "is_canonical_build"} <= set(p)
    assert p["mujoco"] == mujoco.__version__ and p["canonical_mujoco"] == "3.9.0"
    assert p["is_canonical_build"] == (mujoco.__version__ == "3.9.0")
