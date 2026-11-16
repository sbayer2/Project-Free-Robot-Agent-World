"""Which camera views carry which material property, and what survives pooling.

A pixel-level diagnostic for the camera-layout question (PR #48, Step 8 of the
cross-view re-measure). It re-renders the loud-world materials with the
generator's own ``build_mjcf`` and reads the 16x16 (``--block``) block-mean
pixels of each view with a standardized ridge readout:

* ``perview``  - every view of a real 16-view layout on its own;
* ``mean``     - the 16-view mean of those features (a view average that keeps
                 each view's spatial layout, unlike the trained encoder, which
                 pools each view over H x W first);
* ``concat``   - all views concatenated (p >> n; a lower bound on content);
* ``sweep``    - fixed elevations x azimuths (``--elevations``/``--azimuths``),
                 mean over azimuths per elevation.

Readout (fixed after an over-regularization error in the first cloud run,
cloud note f2c5e25): the penalty is ``alpha * I`` on standardized columns
(unscaled, NOT alpha * n); the grid includes alpha = 0 (minimum-norm least
squares); the primary score is NESTED (alpha chosen by inner k-fold CV on each
outer training fold), mean over ``--fold-seeds`` outer shuffles. Best-of-grid
is co-reported, with every pick, so a grid edge shows.

Targets are the essence axes as ``load_world`` returns them (physics divided
by ``PHYSICS_NORMALIZERS``); ``--log-density`` reads log density instead.

Box faces are axis-aligned and the yaw is fixed in every render: the object
body has no orientation attribute and the photo is taken after mj_forward only.

    MUJOCO_GL=osmesa python scripts/specular_view_diag.py layouts \\
        --layouts sphere above --out runs/specular/layouts.json
    MUJOCO_GL=osmesa python scripts/specular_view_diag.py sweep \\
        --elevations -45 -25 -10 -2 5 15 --azimuths 45 135 225 315 \\
        --out runs/specular/sweep.json

Elevations here are MuJoCo's: NEGATIVE looks down from above the floor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

DEFAULT_ALPHAS = (0.0,) + tuple(10.0 ** k for k in range(-5, 7))


# --------------------------------------------------------------------------- #
# Readout (numpy only)
# --------------------------------------------------------------------------- #
def _standardize(xtr: np.ndarray, xte: np.ndarray):
    mu = xtr.mean(0)
    sd = xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)  # constant columns stay constant (0)
    return (xtr - mu) / sd, (xte - mu) / sd


def ridge_path(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray,
               alphas) -> np.ndarray:
    """Predictions on ``xte`` for every alpha, one thin SVD of the standardized
    training matrix. Penalty ``alpha * I`` (unscaled); alpha = 0 is the
    minimum-norm least-squares fit. Returns (len(alphas), n_test)."""
    xs, xt = _standardize(xtr, xte)
    ym = ytr.mean()
    u, s, vt = np.linalg.svd(xs, full_matrices=False)
    tol = max(xs.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    keep = s > tol
    u, s, vt = u[:, keep], s[keep], vt[keep]
    uty = u.T @ (ytr - ym)
    xv = xt @ vt.T
    out = np.empty((len(alphas), xte.shape[0]))
    for i, a in enumerate(alphas):
        out[i] = xv @ (s / (s * s + a) * uty) + ym
    return out


def _folds(n: int, k: int, seed: int):
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    return np.array_split(idx, k)


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def cv_readout(x: np.ndarray, y: np.ndarray, alphas=DEFAULT_ALPHAS, k: int = 5,
               inner_k: int = 4, fold_seeds=(0, 1, 2)) -> dict:
    """Nested (primary) and per-alpha k-fold R2, mean over outer fold seeds.

    For each outer fold, alpha is chosen by inner ``inner_k``-fold CV on the
    outer training rows only (pooled inner R2), refit there, and scored on the
    held-out rows. Returns nested R2, the per-alpha R2 curve, best-of-grid and
    every nested pick."""
    alphas = tuple(float(a) for a in alphas)
    nested, grid_scores, picks = [], [], []
    for fs in fold_seeds:
        pred_n = np.zeros_like(y, dtype=float)
        pred_g = np.zeros((len(alphas), len(y)))
        for f in _folds(len(y), k, fs):
            tr = np.setdiff1d(np.arange(len(y)), f)
            pred_g[:, f] = ridge_path(x[tr], y[tr], x[f], alphas)
            inner = np.zeros((len(alphas), len(tr)))
            for g in _folds(len(tr), inner_k, 1000 + fs):
                itr = np.setdiff1d(np.arange(len(tr)), g)
                inner[:, g] = ridge_path(x[tr][itr], y[tr][itr], x[tr][g], alphas)
            j = int(np.argmax([r2(y[tr], p) for p in inner]))
            picks.append(alphas[j])
            pred_n[f] = pred_g[j, f]
        nested.append(r2(y, pred_n))
        grid_scores.append([r2(y, p) for p in pred_g])
    curve = np.mean(grid_scores, 0)
    b = int(np.argmax(curve))
    return {"nested": float(np.mean(nested)), "nested_by_seed": nested,
            "picks": picks, "alphas": list(alphas), "by_alpha": curve.tolist(),
            "best_of_grid": float(curve[b]), "best_alpha": alphas[b]}


# --------------------------------------------------------------------------- #
# Layouts, materials, rendering (mujoco imported lazily)
# --------------------------------------------------------------------------- #
def layout_angles(name: str, num_views: int = 16, radius: float = 2.6):
    """MuJoCo (azimuth, elevation) per view, exactly as ``render_views`` drives
    the free camera for that layout."""
    from pseudomarble.data import samples

    if name == "sphere":  # legacy: elevation passed through unchanged
        return [samples.pose_to_az_el_dist(p)[:2]
                for p in samples.fibonacci_sphere_poses(num_views, radius)]
    if name == "above":
        fn = getattr(samples, "above_floor_camera_angles", None)
        if fn is None:
            raise SystemExit("layout 'above' needs samples.above_floor_camera_angles "
                             "(PR #48, claude/camera-above-floor)")
        return [(az, -el) for az, el in fn(num_views)]  # positive-up -> MuJoCo
    raise ValueError(f"unknown layout {name!r}")


def sample_materials(n: int, seed: int, alpha: float, gain: float):
    from pseudomarble.materials import MaterialSampler

    sampler = MaterialSampler(seed=seed, coupling_alpha=alpha, coupling_gain=gain)
    return [sampler.sample(material_id=f"scene_{i:06d}") for i in range(n)]


def block_features(img: np.ndarray, block: int) -> np.ndarray:
    h, w, c = img.shape
    return img.reshape(block, h // block, block, w // block, c).mean((1, 3)).ravel()


def coarsen(feats: np.ndarray, block_from: int, block_to: int) -> np.ndarray:
    """Exact block means at a coarser grid from finer block-mean features
    (block means compose: an 8x8 block is the mean of 2x2 adjacent 16x16
    blocks). ``feats``: (..., block_from*block_from*3)."""
    if block_from % block_to:
        raise ValueError("block_to must divide block_from")
    r = block_from // block_to
    lead = feats.shape[:-1]
    x = feats.reshape(*lead, block_to, r, block_to, r, 3)
    return x.mean(axis=(-4, -2)).reshape(*lead, block_to * block_to * 3)


def render_features(mats, angle_sets: dict, resolution: int = 128,
                    block: int = 16, radius: float = 2.6, shape: str = "box",
                    lighting: str = "flat") -> dict:
    """{set name: array (views, scenes, block*block*3)} in [0, 1]."""
    import mujoco

    from pseudomarble.data.generate_mujoco import build_mjcf, half_height

    feats = {k: [[] for _ in v] for k, v in angle_sets.items()}
    for ms in mats:
        model = mujoco.MjModel.from_xml_string(build_mjcf(shape, ms.material,
                                                          lighting=lighting))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        rend = mujoco.Renderer(model, resolution, resolution)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, half_height(shape)]
        for k, angles in angle_sets.items():
            for i, (az, el) in enumerate(angles):
                cam.azimuth, cam.elevation, cam.distance = az, el, radius
                rend.update_scene(data, camera=cam)
                img = rend.render().astype(np.float64) / 255.0
                feats[k][i].append(block_features(img, block))
        rend.close()
    return {k: np.array(v) for k, v in feats.items()}


def targets(mats, log_density: bool = False) -> dict:
    from pseudomarble.config import PHYSICS_NORMALIZERS as N

    p = [m.material.physics for m in mats]
    dens = np.array([math.log(q.density) if log_density else q.density / N["density"]
                     for q in p])
    return {"log_density" if log_density else "density": dens,
            "friction": np.array([q.friction / N["friction"] for q in p]),
            "restitution": np.array([q.restitution / N["restitution"] for q in p])}


def read_set(feats: np.ndarray, ys: dict, **kw) -> dict:
    """Per-view, mean-of-views and concatenated readouts for one layout."""
    v, n, d = feats.shape
    out = {"perview": [], "mean": {}, "concat": {}}
    for i in range(v):
        out["perview"].append({t: cv_readout(feats[i], y, **kw) for t, y in ys.items()})
    xm = feats.mean(0)
    xc = feats.transpose(1, 0, 2).reshape(n, v * d)
    for t, y in ys.items():
        out["mean"][t] = cv_readout(xm, y, **kw)
        out["concat"][t] = cv_readout(xc, y, **kw)
    return out


def _brief(res: dict) -> str:
    return f"{res['nested']:+.3f} (best {res['best_of_grid']:+.3f}@{res['best_alpha']:g})"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["layouts", "sweep"])
    ap.add_argument("--layouts", nargs="+", default=["sphere", "above"])
    ap.add_argument("--elevations", nargs="+", type=float,
                    default=[-45, -25, -10, -2, 5, 15, 30, 45, 65])
    ap.add_argument("--azimuths", nargs="+", type=float, default=[0, 90, 180, 270])
    ap.add_argument("--scenes", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--coupling-alpha", type=float, default=1.0)
    ap.add_argument("--coupling-gain", type=float, default=8.0)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--log-density", action="store_true")
    ap.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    ap.add_argument("--fold-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--features", help="optional .npz cache (written if absent)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    mats = sample_materials(args.scenes, args.seed, args.coupling_alpha, args.coupling_gain)
    if args.mode == "layouts":
        sets = {name: layout_angles(name) for name in args.layouts}
    else:
        sets = {f"el{e:+g}": [(a, e) for a in args.azimuths] for e in args.elevations}
    if args.features and os.path.exists(args.features):
        cache = np.load(args.features)
        feats = {k: cache[k] for k in sets}
    else:
        feats = render_features(mats, sets, args.resolution, args.block)
        if args.features:
            np.savez_compressed(args.features, **feats)
    ys = targets(mats, args.log_density)
    kw = dict(alphas=args.alphas, fold_seeds=tuple(args.fold_seeds))

    report = {"args": vars(args), "angles": {k: [list(a) for a in v] for k, v in sets.items()}}
    if args.mode == "layouts":
        for name in sets:
            report[name] = res = read_set(feats[name], ys, **kw)
            for i, row in enumerate(res["perview"]):
                az, el = sets[name][i]
                print(f"{name:6s} v{i:02d} az {az:7.1f} height {-el:+6.1f}  "
                      + "  ".join(f"{t} {_brief(r)}" for t, r in row.items()), flush=True)
            for pool in ("mean", "concat"):
                print(f"{name:6s} {pool:6s}  " + "  ".join(
                    f"{t} {_brief(r)}" for t, r in res[pool].items()), flush=True)
    else:
        for name in sets:
            xm = feats[name].mean(0)
            report[name] = row = {t: cv_readout(xm, y, **kw) for t, y in ys.items()}
            print(f"{name:7s} " + "  ".join(f"{t} {_brief(r)}" for t, r in row.items()),
                  flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
