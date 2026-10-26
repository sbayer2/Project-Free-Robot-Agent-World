"""Pins for the F35 code-table pilot (scripts/f35_code_table.py): noise touches
only what it should, the contact frame is the first one at or after touchdown
(the leak the pilot first hit was a 'free fall' window ending 2 ms into
contact), and the inverse readout recovers a planted axis and not an absent one.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import f35_code_table as C  # noqa: E402


def _traj(n: int = 10) -> list[dict]:
    return [{"t": 0.016 * i, "pos": [0.0, 0.0, 1.0 - 0.01 * i], "up": [0.0, 0.0, 1.0]}
            for i in range(n)]


def test_resample_is_relative_to_frame_zero():
    X, t = C.resample(_traj(), 4)
    assert X.shape == (4, 6) and np.allclose(X[0, :3], 0.0)
    assert np.isclose(X[-1, 2], -0.09) and np.isclose(t[-1], 0.144)


def test_noise_is_identity_at_zero_and_scaled_per_channel():
    X = np.zeros((200, 8, 6))
    assert C.add_noise(X, 0.0) is X
    Y = C.add_noise(X, 5.0)
    assert np.isclose(Y[..., :3].std(), 0.005, rtol=0.05)
    assert np.isclose(Y[..., 3:].std(), C.UP_NOISE, rtol=0.05)


def test_contact_frame_never_ends_free_fall_inside_contact():
    times = np.linspace(0, 1.98, 64)
    c = C.first_contact_frame(times, 0.3497)
    assert times[c - 1] < 0.3497 <= times[c]
    assert C.first_contact_frame(times, times[5]) == 5


def test_inverse_r2_finds_planted_axis_only():
    rng = np.random.default_rng(0)
    E = rng.uniform(size=(300, 3))
    X = np.stack([E[:, 2], rng.normal(size=300)], 1)[:, None, :]
    r = C.inverse_r2(X, E)
    assert r[2] > 0.95 and r[0] < 0.1 and r[1] < 0.1


def test_family_grouping():
    names = ("drop", "ramp05.7", "push0960", "strike_ball", "scale_plate", "tap_ball", "tilt")
    assert [C.family(n) for n in names] == ["drop", "ramp", "push", "strike", "scale", "tap",
                                            "tilt"]
    z = np.zeros((3, 4, 6))
    T = {"ramp05.7": z, "ramp11.3": z + 1, "drop": z}
    F = C.families(T)
    assert F["ramp"].shape == (3, 48) and F["drop"].shape == (3, 24)


def test_gate_needs_a_shared_pair():
    table = {"drop": [0.0, 0.1, 0.99], "ramp": [0.0, 0.9, 0.85], "push": [0.85, 0.82, 0.1]}
    g = C.gate(table, 0.8)
    assert g["restitution"]["pass"] and g["friction"]["pass"] and not g["density"]["pass"]
    assert g["density"]["families"] == ["push"] and g["density"]["pairs"] == []
    assert set(g["friction"]["pairs"]) == {"ramp->push", "push->ramp"}


def test_pair_matrix_is_ordered_and_excludes_the_diagonal():
    table = {"drop": [0.0, 0.1, 0.99], "tap": [0.0, 0.0, 0.9]}
    m = C.pair_matrix(table, 0.8)
    assert m == {"drop->tap": ["restitution"], "tap->drop": ["restitution"]}


def test_load_npz_matches_resample(tmp_path):
    rng = np.random.default_rng(0)
    raw = np.concatenate([np.tile(np.arange(10)[None, :, None] * 0.016, (2, 1, 1)),
                          rng.normal(size=(2, 10, 6))], 2).astype(np.float32)
    np.savez_compressed(tmp_path / "battery.npz", essence=np.ones((2, 3)), traj_drop=raw)
    (tmp_path / "manifest.json").write_text(
        '{"probes": ["drop"], "drop_height": 0.6}')
    E, T, times, h = C.load_npz(str(tmp_path), 4)
    trajs = [[{"t": float(r[0]), "pos": r[1:4].tolist(), "up": r[4:7].tolist()} for r in s]
             for s in raw.astype(float)]
    for i, tr in enumerate(trajs):
        X, t = C.resample(tr, 4)
        assert np.allclose(T["drop"][i], X, atol=1e-6) and np.allclose(times["drop"], t)
    assert h == 0.6 and E.shape == (2, 3)
