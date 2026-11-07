"""Pins for F_track's classical tracker (scripts/f35_r1_track.py;
docs/CROSS_INTERACTION.md §10.2a): background model, centroid, the
fourth-order orientation moment, two-body separation and the pixel-to-plane
mapping."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import f35_r1_floor as R  # noqa: E402
import f35_r1_frames as F  # noqa: E402
import f35_r1_track as K  # noqa: E402


def _square(cx: float, cy: float, half: float, theta: float = 0.0, ss: int = 8) -> np.ndarray:
    """Coverage image (0-1) of a rotated square, supersampled ss x ss per pixel."""
    g = (np.arange(K.R.PX * ss) + 0.5) / ss - 0.5
    X, Y = np.meshgrid(g, g)
    c, s = np.cos(theta), np.sin(theta)
    u = (X - cx) * c + (Y - cy) * s
    v = -(X - cx) * s + (Y - cy) * c
    inside = (np.abs(u) <= half) & (np.abs(v) <= half)
    return inside.reshape(K.R.PX, ss, K.R.PX, ss).mean((1, 3))


def _adc(cover: np.ndarray, level: float = K.BOX_LEVEL) -> np.ndarray:
    return np.round(cover * level).astype(np.uint16)


def test_background_model_black_scene():
    mean, sd = K.background_model(np.zeros((2, 4, 4), np.uint8))
    assert (mean == 0).all()
    assert sd[0, 0] == pytest.approx(np.sqrt(F.READ_E ** 2 + K.STEP ** 2 / 12) / K.STEP)
    assert K.K_SIGMA * sd[0, 0] < 0.02 * K.BOX_LEVEL       # the box sits far above threshold


def test_centroid_is_subpixel_on_a_coverage_rendered_square():
    img = _adc(_square(40.3, 51.7, 7.2))
    mean, sd = K.background_model(np.zeros((2, 96, 96), np.uint8))
    got = K.track_frame(img, mean, K.K_SIGMA * sd, two_bodies=False)["box"]
    assert abs(got[0] - 40.3) < 0.02 and abs(got[1] - 51.7) < 0.02


@pytest.mark.parametrize("theta", [0.0, 0.2, 0.6, -0.4])
def test_fourth_moment_reads_square_angle_mod_90(theta):
    img = _adc(_square(48.0, 48.0, 10.0, theta))
    mean, sd = K.background_model(np.zeros((2, 96, 96), np.uint8))
    m4 = K.track_frame(img, mean, K.K_SIGMA * sd, two_bodies=False)["box"][2]
    # an axis-aligned square has m4 < 0 (E x^4 + E y^4 - 6 E x^2 y^2 = -4 h^4 / 15),
    # so theta = arg(-m4) / 4; image rows grow downward, so a turn of +theta in
    # (col, row) is -theta in the tracker's (col, -row) frame
    ang = np.angle(-m4) / 4
    d = (ang + theta + np.pi / 4) % (np.pi / 2) - np.pi / 4
    assert abs(d) < 0.01


def test_second_moments_carry_no_angle_for_a_square():
    """Why the tracker uses m4: a square's inertia is isotropic."""
    for theta in (0.0, 0.5):
        cov = _square(48.0, 48.0, 10.0, theta)
        r, c = np.nonzero(cov > 0)
        w = cov[r, c]
        z = (c - (w * c).sum() / w.sum()) + 1j * (r - (w * r).sum() / w.sum())
        assert abs((w * z ** 2).sum()) / (w * abs(z) ** 2).sum() < 0.01


def test_two_bodies_box_and_dimmer_ball_are_separated():
    img = _adc(_square(30.0, 60.0, 14.0)).astype(float)
    yy, xx = np.mgrid[:96, :96]
    ball = ((xx - 60.0) ** 2 + (yy - 60.0) ** 2) <= 4.8 ** 2
    img[ball] = 0.4 * K.BOX_LEVEL
    mean, sd = K.background_model(np.zeros((2, 96, 96), np.uint8))
    got = K.track_frame(img.astype(np.uint16), mean, K.K_SIGMA * sd, two_bodies=True)
    assert set(got) == {"box", "ball"}
    assert abs(got["box"][0] - 30.0) < 0.05 and abs(got["ball"][0] - 60.0) < 0.05
    # touching (contact): still two bodies, the box from its bright core
    img2 = _adc(_square(30.0, 60.0, 14.0)).astype(float)
    ball2 = ((xx - 48.5) ** 2 + (yy - 60.0) ** 2) <= 4.8 ** 2
    img2[ball2 & (img2 == 0)] = 0.4 * K.BOX_LEVEL
    got = K.track_frame(img2.astype(np.uint16), mean, K.K_SIGMA * sd, two_bodies=True)
    assert set(got) == {"box", "ball"} and abs(got["box"][0] - 30.0) < 0.3


def test_empty_frame_finds_nothing_under_noise():
    pair = np.zeros((2, 96, 96), np.uint8)
    rng = np.random.default_rng(0)
    mean, sd = K.background_model(pair)
    hits = sum(bool(K.track_frame(F.sensor(pair, rng), mean, K.K_SIGMA * sd, False))
               for _ in range(200))
    assert hits == 0


def test_pixel_to_plane_inverts_the_camera():
    cam = F.camera("kick", 2.0)
    win = R.window(R.FAMILIES["kick"]["place"], 2.0)
    p = np.array([0.37, 0.0, 0.41])
    col, row = F.project(p, cam)
    assert np.allclose(K._px_to_plane(col, row, win, 2.0), (0.37, 0.41))
