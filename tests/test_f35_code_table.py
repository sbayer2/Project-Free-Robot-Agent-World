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
