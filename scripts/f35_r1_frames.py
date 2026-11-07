"""F35 R1 step (c): the frame generator and gate G1 (docs/CROSS_INTERACTION.md
§10.2, §10.2a, §10.6).

Re-simulates the frozen D7 battery's R1 input recordings with the battery's
own code (scripts/f35_generate_battery.py, untouched) and renders each
recorded instant through one fixed orthographic camera per input family,
placed by the framing rule (§10.2a rules 1-2) at a given window W. Two
hooks are installed in the battery module for the run: its scene builder
gains a <camera> (no body, no physics) and its recorder renders at every
recorded step. Appearance is set on the compiled model (one fixed box
material for every scene), so physics cannot change by construction; the
run also checks it: the re-simulated T0 must equal the stored battery bit
for bit (gate G0 here).

Per recorded instant, three renders at 96 x 96:
  - exposure x 1 and x 1/4: every light's ambient, diffuse and specular
    scaled, the headlight included; one channel is kept (the scene is grey
    under white light, so R = G = B, checked on every frame);
  - a segmentation render, from which G1 takes each body's area centroid.

G1 (apparatus): over every frame in which a body's silhouette lies fully
inside the image, the RMS distance between its segmentation centroid and
its projected T0 position is <= 0.5 px, per recording and body. A cube or a
sphere is centrally symmetric, so under orthographic projection the
silhouette's centroid is the projected centre exactly; what remains is
pixelization and any frame-timing or camera error, which is what G1 checks.

The sensor model (§10.2) is applied at load time by `sensor()`, from a
per-scene seed, so disk holds only the noiseless exposure pair.

    python scripts/f35_r1_frames.py --battery <frozen D7 battery, seed 1234> \
        --out data/r1_frames_s1234 --w kick=2 drop=4 strike=1 dropramp=0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import f35_generate_battery as B  # noqa: E402
from f35_r1_floor import BODY_SIZE, FAMILIES, PX, admissible, window  # noqa: E402

EXPOSURES = (1.0, 0.25)          # light scale of the bright and the dim render
MERGE_KNEE = 250                 # bright pixel used below this, else 4 x dim
BOX_ALBEDO = 0.5                 # one grey, matte, non-reflective finish for every box
CAM_DIST = 20.0                  # m from the motion plane; orthographic, so only clipping
G1_PX = 0.5                      # RMS bound
# sensor (§10.2): nominal white (255 in the x 1 exposure) = 2 500 e-
E_WHITE, FULL_WELL, READ_E, ADC_MAX = 2500.0, 10000.0, 3.0, 4095
BODY_GEOM = {"box": "obj", "ball": "ball_geom"}


# --- camera -------------------------------------------------------------------

def start_origin(fam: str) -> np.ndarray:
    """The box's nominal start position (world), the fixed camera's anchor.
    The floor (f35_r1_floor) anchors on each scene's frame-0 position, which
    differs from this by the first 2 ms of motion (microns; reported)."""
    if FAMILIES[fam]["plane"] == "ramp":
        return (B.plane_normal(B.DROPRAMP_DEG) * (0.15 + 0.0005)
                + np.array([0.0, 0.0, B.DROPRAMP_HEIGHT]))
    return np.zeros(3)


def camera(fam: str, w: float) -> dict:
    """Orthographic camera for a family at window w: centre (world), right
    and up axes (world), and the window in plane coordinates."""
    u_lo, u_hi, v_lo, v_hi = win = window(FAMILIES[fam]["place"], w)
    uc, vc = 0.5 * (u_lo + u_hi), 0.5 * (v_lo + v_hi)
    o = start_origin(fam)
    if FAMILIES[fam]["plane"] == "xz":          # u = x - x0, v = z; looks along +y
        right, up = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
        centre = np.array([o[0] + uc, 0.0, vc])
    else:                                       # u = -(y - y0), v = z - z0; looks along +x
        right, up = np.array([0, -1.0, 0]), np.array([0, 0, 1.0])
        centre = np.array([0.0, o[1] - uc, o[2] + vc])
    view = np.cross(up, right)                  # camera looks along -(right x up)
    return {"centre": centre, "right": right, "up": up, "view": view, "w": w,
            "window": win, "pos": centre - CAM_DIST * view}


def camera_xml(cam: dict) -> str:
    p, r, u = cam["pos"], cam["right"], cam["up"]
    return (f'<camera name="r1cam" projection="orthographic" fovy="{cam["w"]:g}" '
            f'pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
            f'xyaxes="{r[0]:g} {r[1]:g} {r[2]:g} {u[0]:g} {u[1]:g} {u[2]:g}"/>')


def project(pos: np.ndarray, cam: dict) -> np.ndarray:
    """World positions (..., 3) -> pixel (col, row), pixel centres at integers,
    row 0 at the top."""
    d = pos - cam["centre"]
    s = PX / cam["w"]
    col = (d @ cam["right"]) * s + 0.5 * PX - 0.5
    row = 0.5 * PX - 0.5 - (d @ cam["up"]) * s
    return np.stack([col, row], -1)


# --- silhouette geometry (G1's expected centroid) -----------------------------

IMAGE_RECT = np.array([[-0.5, -0.5], [PX - 0.5, -0.5], [PX - 0.5, PX - 0.5], [-0.5, PX - 0.5]])


def hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points, counter-clockwise (monotone chain)."""
    p = sorted(map(tuple, np.round(pts, 12)))
    if len(p) < 3:
        return np.array(p)

    def half(seq):
        h: list = []
        for q in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h
    lo, hi = half(p), half(p[::-1])
    return np.array(lo[:-1] + hi[:-1])


def _area2(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())


def clip(poly: np.ndarray, conv: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: poly clipped by the convex polygon conv. Both are
    made counter-clockwise first (pixel rows grow downward, so orientation
    is whatever the input happens to be)."""
    if len(poly) < 3:
        return np.zeros((0, 2))
    if _area2(conv) < 0:
        conv = conv[::-1]
    out = poly
    for i in range(len(conv)):
        a, b = conv[i], conv[(i + 1) % len(conv)]
        if len(out) == 0:
            break

        def inside(q, a=a, b=b):
            return (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) >= 0
        new = []
        for j in range(len(out)):
            p, q = out[j], out[(j + 1) % len(out)]
            ip, iq = inside(p), inside(q)
            if ip:
                new.append(p)
            if ip != iq:
                d = (q - p)
                e = (b - a)
                t = ((a[0] - p[0]) * e[1] - (a[1] - p[1]) * e[0]) / (d[0] * e[1] - d[1] * e[0])
                new.append(p + t * d)
        out = np.array(new) if new else np.zeros((0, 2))
    return out


def area_centroid(poly: np.ndarray) -> tuple[float, np.ndarray]:
    """|area| and centroid of a simple polygon; (0, nan) if degenerate."""
    if len(poly) < 3:
        return 0.0, np.full(2, np.nan)
    x, y = poly[:, 0], poly[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cr.sum() / 2.0
    if abs(a) < 1e-12:
        return 0.0, np.full(2, np.nan)
    c = np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()]) / (6 * a)
    return abs(a), c


def silhouette(kind: str, pos: np.ndarray, mat: np.ndarray, size: np.ndarray,
               cam: dict) -> np.ndarray:
    """Projected silhouette polygon (pixel coords) of a box (half sizes) or a
    sphere (radius) at a world pose."""
    if kind == "box":
        signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        pts = pos + (signs * size[:3]) @ mat.T
        return hull(project(pts, cam))
    ang = np.linspace(0.0, 2 * np.pi, 96, endpoint=False)
    ring = (np.cos(ang)[:, None] * cam["right"] + np.sin(ang)[:, None] * cam["up"]) * size[0]
    return project(pos + ring, cam)


_GRID = np.stack(np.meshgrid(np.arange(PX), np.arange(PX)), -1).reshape(-1, 2).astype(float)


def raster(poly: np.ndarray) -> np.ndarray:
    """Pixel centres inside a convex polygon, (N, 2) [col, row]: what a
    sampler at pixel centres draws, without its tie rules on edges."""
    if len(poly) < 3:
        return np.zeros((0, 2))
    a, b = poly, np.roll(poly, -1, 0)
    cr = ((b[:, 0] - a[:, 0])[None] * (_GRID[:, 1:2] - a[:, 1][None])
          - (b[:, 1] - a[:, 1])[None] * (_GRID[:, 0:1] - a[:, 0][None]))
    inside = (cr >= 0).all(1) | (cr <= 0).all(1)
    return _GRID[inside]


def nearest_depth(kind: str, pos: np.ndarray, mat: np.ndarray, size: np.ndarray,
                  cam: dict) -> float:
    """Distance along the view direction to the body's nearest point."""
    if kind == "box":
        signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        return float(((pos + (signs * size[:3]) @ mat.T - cam["pos"]) @ cam["view"]).min())
    return float((pos - cam["pos"]) @ cam["view"] - size[0])


# --- rendering hooks ----------------------------------------------------------

_JOB: dict = {}                  # per-process: the active camera and the frames stash


def _mjcf_with_camera(mat, **kw) -> str:
    xml = _JOB["orig_mjcf"](mat, **kw)
    return B.replace_once(xml, "<worldbody>", "<worldbody>\n    " + camera_xml(_JOB["cam"]))


def _appearance(model) -> None:
    """One fixed box finish for every scene (visual only)."""
    m = model.mat("m_obj").id
    model.mat_rgba[m] = (BOX_ALBEDO, BOX_ALBEDO, BOX_ALBEDO, 1.0)
    model.mat_specular[m] = 0.0
    model.mat_shininess[m] = 0.0
    model.mat_reflectance[m] = 0.0


def _scale_lights(model, k: float) -> None:
    model.light_ambient[:] = _JOB["light0"][0] * k
    model.light_diffuse[:] = _JOB["light0"][1] * k
    model.light_specular[:] = _JOB["light0"][2] * k
    model.vis.headlight.ambient[:] = _JOB["head0"][0] * k
    model.vis.headlight.diffuse[:] = _JOB["head0"][1] * k
    model.vis.headlight.specular[:] = _JOB["head0"][2] * k


def _grey(img: np.ndarray) -> np.ndarray:
    if not (np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2])):
        raise RuntimeError("frame is not grey: R, G and B differ")
    return img[..., 0]


def _record_with_frames(model, data, bodies, seconds=B.SIM_SECONDS, fps=B.FPS, force_fn=None):
    """B._record's loop, unchanged, plus three renders at every recorded step."""
    import mujoco

    from pseudomarble.data.generate_mujoco import _quat_to_up

    _appearance(model)
    _JOB["light0"] = (model.light_ambient.copy(), model.light_diffuse.copy(),
                      model.light_specular.copy())
    h = model.vis.headlight
    _JOB["head0"] = (np.array(h.ambient), np.array(h.diffuse), np.array(h.specular))
    rnd = mujoco.Renderer(model, PX, PX)
    seg_ids = {b: model.geom(g).id for b, g in BODY_GEOM.items()
               if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) >= 0}

    def grab():
        out = []
        for k in EXPOSURES:
            _scale_lights(model, k)
            rnd.update_scene(data, camera="r1cam")
            out.append(_grey(rnd.render()))
        _scale_lights(model, 1.0)
        rnd.enable_segmentation_rendering()
        rnd.update_scene(data, camera="r1cam")
        seg = rnd.render()
        rnd.disable_segmentation_rendering()
        cam = _JOB["cam"]
        geo = {b: ("box" if b == "box" else "ball", data.geom_xpos[g].copy(),
                   data.geom_xmat[g].reshape(3, 3).copy(), model.geom_size[g].copy())
               for b, g in seg_ids.items()}
        polys = {b: silhouette(k, p, m, s, cam) for b, (k, p, m, s) in geo.items()}
        cents = {}
        for b, gid in seg_ids.items():
            mask = (seg[..., 0] == gid) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            rows, cols = np.nonzero(mask)
            full, _ = area_centroid(polys[b])
            vis_area, vis_c = area_centroid(clip(polys[b], IMAGE_RECT))
            # occluded: another body overlaps it in the image and is nearer
            occl = any(area_centroid(clip(polys[b], hull(polys[o])))[0] > 0
                       and nearest_depth(*geo[o], cam) < nearest_depth(*geo[b], cam)
                       for o in seg_ids if o != b)
            seg_c = (cols.mean(), rows.mean()) if len(rows) else (np.nan, np.nan)
            ras = raster(polys[b])
            ras_c = ras.mean(0) if len(ras) else np.full(2, np.nan)
            # [seg col, seg row, seg px, expected col, expected row, expected
            #  visible area, full silhouette area, occluded, raster col,
            #  raster row, raster px, pixels where render and raster differ]
            diff = (len(set(zip(cols.tolist(), rows.tolist(), strict=True))
                        ^ set(map(tuple, ras.astype(int).tolist()))))
            cents[b] = (*seg_c, float(len(rows)), *vis_c, vis_area, full, float(occl),
                        *ras_c, float(len(ras)), float(diff))
        return np.stack(out), cents

    ids = {b: model.body(b).id for b in bodies}
    n_steps = int(seconds / model.opt.timestep)
    every = max(1, round((1.0 / fps) / model.opt.timestep))
    out: dict[str, list] = {b: [] for b in bodies}
    frames, cents = [], []
    for step in range(n_steps):
        t = step * model.opt.timestep
        if force_fn is not None:
            data.xfrc_applied[ids["object"]] = force_fn(t)
        mujoco.mj_step(model, data)
        if step % every == 0:
            for b, i in ids.items():
                out[b].append([t, *map(float, data.xpos[i]), *_quat_to_up(data.xquat[i])])
            f, c = grab()
            frames.append(f)
            cents.append(c)
    # the empty scene (tracker background, step (d)): bodies hidden, same lights
    opt = mujoco.MjvOption()
    for gid in seg_ids.values():
        model.geom_group[gid] = 5
    opt.geomgroup[5] = 0
    bg = []
    for k in EXPOSURES:
        _scale_lights(model, k)
        rnd.update_scene(data, camera="r1cam", scene_option=opt)
        bg.append(_grey(rnd.render()))
    _scale_lights(model, 1.0)
    rnd.close()
    _JOB["stash"].append({"frames": np.stack(frames), "cents": cents, "bg": np.stack(bg)})
    return out


def init_worker(design: str) -> None:
    (c, i, s), deg, hgt = B.DESIGNS[design]
    B.init_worker((c, i, s), deg, hgt)
    _JOB["orig_mjcf"] = B.mjcf
    B.mjcf = _mjcf_with_camera
    B._record = _record_with_frames


def _run(rec: str, mat):
    """The battery's own run_* for one recording; returns {body: T0 list}."""
    if rec == "drop":
        return {"box": B.run_drop(mat)}
    if rec.startswith("kick"):
        return {"box": B.run_kick(mat, float(rec[4:]))}
    if rec == "strike":
        box, ball = B.run_strike(mat)
        return {"box": box, "ball": ball}
    if rec.startswith("dropramp"):
        return {"box": B.run_ramp(mat, B.DROPRAMP_DEG, drop=B.DROPRAMP_HEIGHT)}
    raise ValueError(rec)


def render_scene(job: tuple) -> dict:
    """One scene, one input family at window w: frames per recording,
    re-simulated T0, segmentation centroids, projected T0, background."""
    fam, w, item = job
    _JOB["cam"] = cam = camera(fam, w)
    res = {}
    for rec, _ in _family_recs(fam):
        if rec == "strike_ball":
            continue
        _JOB["stash"] = []
        traj = _run(rec, item[1])
        st = _JOB["stash"][0]
        for body, t0 in traj.items():
            name = "strike_ball" if (rec == "strike" and body == "ball") else rec
            arr = np.array(t0, dtype=np.float64)
            res[name] = {"t0": arr.astype(np.float32),
                         "proj": project(arr[:, 1:4], cam).astype(np.float32),
                         "cent": np.array([c[body] for c in st["cents"]], dtype=np.float32)}
        res[rec]["frames"] = st["frames"]
        res[rec]["bg"] = st["bg"]
    return res


def _family_recs(fam: str) -> list[tuple[str, str]]:
    return [(r.format(deg=B.DROPRAMP_DEG), b) for r, b in FAMILIES[fam]["recs"]]


# --- exposure merge and sensor model (applied at load) -----------------------

def merge(pair: np.ndarray) -> np.ndarray:
    """(..., 2, H, W) uint8 [x 1, x 1/4] -> linear radiance in x 1 units, 0-1020."""
    bright = pair[..., 0, :, :].astype(np.float64)
    dim = pair[..., 1, :, :].astype(np.float64)
    return np.where(bright < MERGE_KNEE, bright, 4.0 * dim)


def sensor(pair: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Noiseless exposure pair -> the model's input, a 12-bit single-channel
    mosaic (§10.2 steps 1-5): electrons, Poisson shot noise, Gaussian read
    noise, full-well clip, 12-bit ADC. The scene is grey, so the RGGB mosaic
    samples equal channels and is the identity on the stored grey channel."""
    e = merge(pair) / 255.0 * E_WHITE
    e = rng.poisson(e).astype(np.float64) + rng.normal(0.0, READ_E, e.shape)
    e = np.clip(e, 0.0, FULL_WELL)
    return np.clip(np.round(e / FULL_WELL * ADC_MAX), 0, ADC_MAX).astype(np.uint16)


def load_frames(path: str, rec: str, noise_seed: int, scenes: np.ndarray | None = None):
    """Sensor-model frames (S, F, H, W) uint16 for one recording, one seeded
    generator per scene: rng([noise_seed, scene index, crc of rec])."""
    import zlib

    pair = np.load(path)[f"frames_{rec}"]
    idx = np.arange(len(pair)) if scenes is None else scenes
    tag = zlib.crc32(rec.encode())
    return np.stack([sensor(pair[i], np.random.default_rng([noise_seed, int(i), tag]))
                     for i in idx])


# --- G1 -----------------------------------------------------------------------

G1_MIN_VISIBLE = 0.25          # of the full silhouette area, for a frame to count


def g1(cent: np.ndarray, proj: np.ndarray) -> dict:
    """G1 per recording and body. Scored frames: not occluded and at least
    G1_MIN_VISIBLE of the silhouette inside the image. Error = segmentation
    centroid minus the centroid of the projected silhouette clipped to the
    image (equal to the projected T0 position whenever the body is fully in
    view; reported separately as `rms_px_full_view`). What G1 grades is that
    analytic error, which includes pixelization (a resting, axis-aligned box
    keeps the same sub-pixel offset for many frames). Diagnostics, not
    graded: the same silhouette sampled at pixel centres (`raster`) against
    the render, centroid and pixel by pixel, which separates pixelization
    from camera or timing error; and presence mismatches, frames where the
    render shows a body that the raster does not, or the reverse."""
    seg_c, seg_a = cent[..., :2], cent[..., 2]
    exp_c, vis_a, full_a, occl = cent[..., 3:5], cent[..., 5], cent[..., 6], cent[..., 7] > 0
    ras_c, ras_n, px_diff = cent[..., 8:10], cent[..., 10], cent[..., 11]
    ok = ~occl & (vis_a >= G1_MIN_VISIBLE * full_a) & (seg_a > 0)
    err = np.linalg.norm(seg_c - exp_c, axis=-1)[ok]
    full = ok & (vis_a >= full_a - 1e-6)
    err_full = np.linalg.norm(seg_c - proj, axis=-1)[full]
    err_ras = np.linalg.norm(seg_c - ras_c, axis=-1)[ok & (ras_n > 0)]
    mism = ~occl & (((seg_a == 0) & (ras_n > 0)) | ((seg_a > 0) & (ras_n == 0)))
    seen = ~occl & ((seg_a > 0) | (ras_n > 0))

    def rms(e):
        return float(np.sqrt((e ** 2).mean())) if e.size else None
    return {"rms_px": rms(err), "max_px": float(err.max()) if err.size else None,
            "rms_px_full_view": rms(err_full), "frames_full_view": int(full.sum()),
            "frames_used": int(ok.sum()), "frames_total": int(ok.size),
            "frames_occluded": int(occl.sum()),
            "raster_rms_px": rms(err_ras),
            "raster_pixels_differing_frac": float(px_diff[seen].sum()
                                                  / max(1.0, seg_a[seen].sum())),
            "presence_mismatch": int(mism.sum()),
            "pass": bool(err.size and rms(err) <= G1_PX)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True, help="dir with battery.npz + manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", nargs="+", required=True, metavar="FAM=W")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.battery, "manifest.json")))
    design = man["design_iteration"]
    if design not in B.DESIGNS:
        raise SystemExit(f"battery design {design!r} is not a frozen design")
    seed = int(man["source"].rsplit(" ", 1)[1])
    stored = np.load(os.path.join(args.battery, "battery.npz"))
    items = B.sampled_materials(seed, man["n_scenes"])[:args.limit]
    if [i for i, _ in items] != list(stored["scene_ids"][:len(items)]):
        raise SystemExit("scene ids differ from the stored battery")
    init_worker(design)                      # parent too, for DROPRAMP_DEG
    os.makedirs(args.out, exist_ok=True)
    report = {"battery": args.battery, "battery_provenance": man.get("provenance"),
              "provenance": B.provenance(), "design": design, "px": PX,
              "exposures": EXPOSURES, "merge_knee": MERGE_KNEE, "box_albedo": BOX_ALBEDO,
              "sensor": {"e_white": E_WHITE, "full_well": FULL_WELL, "read_e": READ_E,
                         "adc_max": ADC_MAX}, "families": {}}
    for spec in args.w:
        fam, w = spec.split("=")
        w = float(w)
        if not admissible(fam, w):
            raise SystemExit(f"{fam} at W={w} is not admissible (a body < 2 px)")
        t0 = time.time()
        with ProcessPoolExecutor(args.workers, initializer=init_worker,
                                 initargs=(design,)) as ex:
            res = list(ex.map(render_scene, [(fam, w, it) for it in items], chunksize=2))
        cam = camera(fam, w)
        fam_rep = {"w": w, "m_per_px": w / PX, "window_plane": cam["window"],
                   "camera_pos": cam["pos"].tolist(), "camera_right": cam["right"].tolist(),
                   "camera_up": cam["up"].tolist(), "start_origin": start_origin(fam).tolist(),
                   "body_px": {b: BODY_SIZE[b] * PX / w for _, b in _family_recs(fam)},
                   "recordings": {}}
        arrays = {}
        for rec, body in _family_recs(fam):
            t0s = np.stack([r[rec]["t0"] for r in res])
            same = np.array_equal(t0s, stored[f"traj_{rec}"][:len(items)])
            maxdiff = float(np.abs(t0s - stored[f"traj_{rec}"][:len(items)]).max())
            cent = np.stack([r[rec]["cent"] for r in res])
            proj = np.stack([r[rec]["proj"] for r in res])
            o_dev = float(np.abs(t0s[:, 0, 1:4] - start_origin(fam)).max())
            fam_rep["recordings"][rec] = {"body": body, "t0_bit_identical": bool(same),
                                          "t0_max_abs_diff": maxdiff,
                                          "frame0_vs_start_origin_m": o_dev,
                                          "g1": g1(cent, proj)}
            arrays[f"cent_{rec}"] = cent
            arrays[f"proj_{rec}"] = proj
            if "frames" in res[0][rec]:
                arrays[f"frames_{rec}"] = np.stack([r[rec]["frames"] for r in res])
                bgs = np.stack([r[rec]["bg"] for r in res])
                if not (bgs == bgs[0]).all():
                    raise RuntimeError(f"{rec}: empty-scene frame differs across scenes")
                arrays[f"background_{rec}"] = bgs[0]
                fam_rep["recordings"][rec]["frame_max"] = int(arrays[f"frames_{rec}"].max())
                fam_rep["recordings"][rec]["merged_above_knee_frac"] = float(
                    (arrays[f"frames_{rec}"][:, :, 0] >= MERGE_KNEE).mean())
        path = os.path.join(args.out, f"frames_{fam}_W{w:g}.npz")
        np.savez_compressed(path, scene_ids=stored["scene_ids"][:len(items)], **arrays)
        fam_rep["file"] = os.path.basename(path)
        fam_rep["seconds"] = round(time.time() - t0, 1)
        report["families"][f"{fam}_W{w:g}"] = fam_rep
        print(f"[r1-frames] {fam} W={w:g} m: {len(items)} scenes in {fam_rep['seconds']} s")
        for rec, rr in fam_rep["recordings"].items():
            g = rr["g1"]

            def f(x):
                return "  n/a" if x is None else f"{x:.3f}"
            print(f"  {rec:<14} T0 identical {rr['t0_bit_identical']}  G1 rms {f(g['rms_px'])} "
                  f"px (max {f(g['max_px'])}; full view {f(g['rms_px_full_view'])} on "
                  f"{g['frames_full_view']}) {g['frames_used']}/{g['frames_total']} frames, "
                  f"{g['frames_occluded']} occluded  {'PASS' if g['pass'] else 'FAIL'}\n"
                  f"  {'':<14} raster: centroid rms {f(g['raster_rms_px'])} px, pixels "
                  f"differing {g['raster_pixels_differing_frac']:.4f}, presence mismatches "
                  f"{g['presence_mismatch']}")
    json.dump(report, open(os.path.join(args.out, "frames_manifest.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
