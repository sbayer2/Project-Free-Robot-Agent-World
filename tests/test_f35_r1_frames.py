"""Pins for the R1 frame generator (scripts/f35_r1_frames.py;
docs/CROSS_INTERACTION.md §10.2, §10.2a, §10.6 G1): camera placement against
the framing rule's window, silhouette geometry, the exposure merge's
linearity across the seam (headlight scaled), the sensor model and G1's
frame selection."""

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


def _plane_to_world(fam: str, u: float, v: float) -> np.ndarray:
    o = F.start_origin(fam)
    if R.FAMILIES[fam]["plane"] == "xz":
        return np.array([o[0] + u, 0.0, v])
    return np.array([0.0, o[1] - u, o[2] + v])


@pytest.mark.parametrize("fam,w", [("kick", 2.0), ("drop", 4.0), ("strike", 1.0),
                                   ("dropramp", 0.5), ("dropramp", 1.0)])
def test_camera_frames_the_rule_window_exactly(fam, w):
    cam = F.camera(fam, w)
    u_lo, u_hi, v_lo, v_hi = R.window(R.FAMILIES[fam]["place"], w)
    # window corners land on the image corners (pixel edges at -0.5 and 95.5)
    assert np.allclose(F.project(_plane_to_world(fam, u_lo, v_hi), cam), [-0.5, -0.5])
    assert np.allclose(F.project(_plane_to_world(fam, u_hi, v_lo), cam), [95.5, 95.5])
    # the camera looks along the plane normal: out-of-plane offsets do not move a point
    p = _plane_to_world(fam, 0.1, 0.1)
    assert np.allclose(F.project(p + 0.7 * cam["view"], cam), F.project(p, cam))


def test_box_start_sits_where_rule_2_puts_it():
    col, row = F.project(F.start_origin("kick"), F.camera("kick", 2.0))
    assert abs((col + 0.5) / 96 - 0.15) < 1e-12          # 15 % from the trailing edge
    col, row = F.project(F.start_origin("dropramp"), F.camera("dropramp", 1.0))
    assert abs((col + 0.5) / 96 - 0.15) < 1e-12 and abs((row + 0.5) / 96 - 0.15) < 1e-12


def test_polygon_clip_area_and_centroid():
    sq = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
    a, c = F.area_centroid(sq)
    assert a == pytest.approx(8.0) and np.allclose(c, [2.0, 1.0])
    half = F.clip(sq, np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 3.0], [-1.0, 3.0]]))
    a, c = F.area_centroid(half)
    assert a == pytest.approx(2.0) and np.allclose(c, [0.5, 1.0])
    assert len(F.clip(sq, sq + 10.0)) == 0                               # disjoint
    assert F.area_centroid(F.hull(np.vstack([sq, [[2.0, 1.0]]])))[0] == pytest.approx(8.0)


def test_raster_samples_pixel_centres():
    sq = np.array([[1.6, 1.6], [4.4, 1.6], [4.4, 3.4], [1.6, 3.4]])     # covers cols 2-4, rows 2-3
    px = F.raster(sq)
    assert len(px) == 6 and np.allclose(px.mean(0), [3.0, 2.5])


def test_box_silhouette_centroid_is_projected_centre():
    cam = F.camera("kick", 2.0)
    rot = np.array([[np.cos(0.4), 0, np.sin(0.4)], [0, 1, 0], [-np.sin(0.4), 0, np.cos(0.4)]])
    pos = np.array([0.3, 0.02, 0.4])
    poly = F.silhouette("box", pos, rot, np.array([0.15, 0.15, 0.15]), cam)
    face = (0.3 * 96 / 2.0) ** 2
    assert np.allclose(F.area_centroid(poly)[1], F.project(pos, cam))
    assert F.area_centroid(poly)[0] == pytest.approx(face)              # in-plane turn: same area
    yaw = np.array([[np.cos(0.4), -np.sin(0.4), 0], [np.sin(0.4), np.cos(0.4), 0], [0, 0, 1]])
    poly = F.silhouette("box", pos, yaw, np.array([0.15, 0.15, 0.15]), cam)
    assert np.allclose(F.area_centroid(poly)[1], F.project(pos, cam))
    assert F.area_centroid(poly)[0] > 1.2 * face                         # out of plane: wider


def test_merge_uses_bright_below_knee_and_four_times_dim_above():
    pair = np.zeros((2, 2, 2), np.uint8)
    pair[0] = [[100, 249], [250, 255]]
    pair[1] = [[25, 62], [70, 200]]
    assert F.merge(pair).tolist() == [[100.0, 249.0], [280.0, 800.0]]


def _renderer_or_skip():
    import mujoco

    try:
        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        mujoco.Renderer(m, 8, 8).close()
    except Exception as e:                                              # no GL context
        pytest.skip(f"no offscreen GL: {e}")


def test_merge_is_linear_across_the_seam_with_the_headlight_scaled():
    """§10.2: the merged exposure pair is linear in light across the 250
    seam, every light (headlight included) scaled. A matte grey plate faces
    the camera under a headlight and a light; the light level k sweeps the
    x 1 frame from mid-scale through clipping."""
    _renderer_or_skip()
    import mujoco

    xml = """<mujoco><visual><headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4"
      specular="0 0 0"/></visual><asset><material name="m_obj" rgba="0.5 0.5 0.5 1"
      specular="0" shininess="0" reflectance="0"/></asset><worldbody>
      <light pos="0 -3 0" dir="0 1 0" diffuse="0.8 0.8 0.8" specular="0 0 0"/>
      <camera name="r1cam" projection="orthographic" fovy="1" pos="0 -5 0" xyaxes="1 0 0 0 0 1"/>
      <geom name="obj" type="box" size="2 0.05 2" material="m_obj"/></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    F._JOB["light0"] = (model.light_ambient.copy(), model.light_diffuse.copy(),
                        model.light_specular.copy())
    h = model.vis.headlight
    F._JOB["head0"] = (np.array(h.ambient), np.array(h.diffuse), np.array(h.specular))
    rnd = mujoco.Renderer(model, 16, 16)
    ks, merged, bright = np.linspace(0.6, 2.2, 17), [], []
    for k in ks:
        pair = []
        for e in F.EXPOSURES:
            F._scale_lights(model, k * e)
            rnd.update_scene(data, camera="r1cam")
            pair.append(F._grey(rnd.render()))
        merged.append(F.merge(np.stack(pair))[8, 8])
        bright.append(pair[0][8, 8])
    rnd.close()
    merged, bright = np.array(merged), np.array(bright)
    assert bright.min() < F.MERGE_KNEE and bright.max() == 255          # the sweep crosses the seam
    lo = bright < F.MERGE_KNEE
    slope = (merged[lo] / ks[lo]).mean()
    resid = merged - slope * ks
    # One slope on both sides of the seam. The renderer's 8-bit output is good
    # to about one step (measured: it rounds down about as often as to nearest),
    # so the bright branch is good to ~1 and the dim branch, 4 x an 8-bit value,
    # to ~4 in x 1 units: no residual beyond one dim step (plus the slope's own
    # error), and no offset at the seam beyond half a dim step on average.
    assert np.abs(resid[lo]).max() <= 1.0
    assert np.abs(resid[~lo]).max() <= 4.5
    assert abs(resid[~lo].mean()) <= 2.0
    assert merged.max() > 450                          # well beyond the 8-bit clip


def test_sensor_shot_and_read_noise_and_adc():
    pair = np.zeros((2, 64, 64), np.uint8)
    pair[0], pair[1] = 128, 32
    out = F.sensor(np.broadcast_to(pair, (50, 2, 64, 64)), np.random.default_rng(0))
    assert out.dtype == np.uint16 and out.max() <= F.ADC_MAX
    step = F.FULL_WELL / F.ADC_MAX                                        # e- per DN
    e = 128 / 255 * F.E_WHITE
    assert out.mean() * step == pytest.approx(e, rel=2e-3)
    var_e = (out.astype(float) * step).var()
    assert var_e == pytest.approx(e + F.READ_E ** 2 + step ** 2 / 12, rel=0.03)


def test_sensor_clips_at_full_well_and_is_seeded():
    pair = np.full((2, 8, 8), 255, np.uint8)                             # merged 1020 = full well
    out = F.sensor(pair, np.random.default_rng(1))
    assert out.max() == F.ADC_MAX
    a = F.sensor(pair, np.random.default_rng(3))
    assert (a == F.sensor(pair, np.random.default_rng(3))).all()


def test_frames_must_be_grey():
    img = np.zeros((4, 4, 3), np.uint8)
    assert F._grey(img).shape == (4, 4)
    img[0, 0, 1] = 1
    with pytest.raises(RuntimeError):
        F._grey(img)


def test_g1_scores_unoccluded_visible_frames_only():
    # columns: seg col,row,px | expected col,row | visible, full | occluded |
    #          raster col,row,px | differing px
    good = [10.2, 20.0, 50, 10.0, 20.0, 50, 50, 0, 10.2, 20.0, 50, 0]
    occl = [30.0, 20.0, 10, 10.0, 20.0, 50, 50, 1, 10.0, 20.0, 50, 40]
    sliver = [0.1, 20.0, 2, 0.0, 20.0, 5, 50, 0, 0.0, 20.0, 2, 0]      # 10 % visible: not scored
    cent = np.array([[good, occl, sliver]], dtype=float)
    proj = np.array([[[10.0, 20.0]] * 3])
    g = F.g1(cent, proj)
    assert g["frames_used"] == 1 and g["frames_occluded"] == 1
    assert g["rms_px"] == pytest.approx(0.2) and g["pass"]
    assert g["presence_mismatch"] == 0


def test_init_worker_is_idempotent_so_fork_does_not_recurse():
    """A forked worker inherits the hooked module and calls init_worker
    again; B.mjcf must still reach the real builder exactly once."""
    from pseudomarble import materials as M

    orig = (F.B.mjcf, F.B._record, dict(F._JOB))
    try:
        F.init_worker("D7")
        F.init_worker("D7")                          # what a forked worker does
        assert F._JOB["orig_mjcf"] is not F._mjcf_with_camera
        F._JOB["cam"] = F.camera("kick", 2.0)
        mat = M.MaterialSampler(seed=1234).sample(0).material
        xml = F.B.mjcf(mat)
        assert xml.count('name="r1cam"') == 1
    finally:
        F.B.mjcf, F.B._record = orig[0], orig[1]
        F._JOB.clear()
        F._JOB.update(orig[2])
