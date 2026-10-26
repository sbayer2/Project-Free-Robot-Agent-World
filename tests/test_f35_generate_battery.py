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
    assert names[0] == "drop" and names[-2:] == ["strike", "strike_ball"]
    assert len(names) == 1 + len(G.RAMP_DEG) + len(G.PUSH_N) + 2
    assert "ramp05.7" in names and "push0015" in names


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
        assert len(rec) == 125 and len(rec[0]) == 7
