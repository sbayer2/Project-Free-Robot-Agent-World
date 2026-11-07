"""F35 R1 step (d): F_track, a classical tracker on the sensor-model frames
(docs/CROSS_INTERACTION.md §10.2a, "The graded bar").

Per frame, from the 12-bit sensor frame (f35_r1_frames.sensor):
  1. background subtraction against the empty-scene frame's expected value;
  2. a 5 sigma threshold, sigma the analytic per-pixel noise of the empty
     scene (shot noise at its expected signal, 3 e- read noise, ADC step);
  3. connected components (8-connected), components under MIN_AREA px dropped;
  4. per body, the signal-weighted centroid (the box has one albedo, so the
     weight is pixel coverage: MuJoCo renders with 4x multisampling, edge
     pixels at 1/4, 1/2, 3/4) and, for the box, its orientation from the
     normalized fourth-order complex moment m4 = sum w (z - zbar)^4 /
     sum w |z - zbar|^4. A square's second moments are isotropic, so they
     carry no angle; m4 has the square's 4-fold symmetry (arg(-m4) = 4 theta:
     an axis-aligned square has m4 < 0) and its magnitude drops when the
     outline stops being a square (out-of-plane rotation, a cut edge). The
     readout takes m4's real and imaginary parts, so the sign is immaterial.
Two bodies (the strike): the box is the largest "core" component (pixels
above CORE_FRAC of the box's interior level, at least CORE_MIN px) grown by
two pixels for its anti-aliased edge; the ball is the largest remaining
component. A body not found is out of view: zeros and visibility bit 0, as
rule 4.

Per frame per body the features are (u, v) in the plane coordinates of
f35_r1_floor, m4 (re, im) for the box, and the visibility bit; the readout
is f35_r1_floor's (best of ridge and kNN-5, split seed 0, corner split
co-reported, skill vs train variance).

    python scripts/f35_r1_track.py --battery <frozen D7 battery, seed 1234> \
        --frames data/r1_frames_s1234 --sets kick_W2 drop_W4 strike_W1 \
        dropramp_W0.5 dropramp_W1 --noise-seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import f35_outcomes as O  # noqa: E402
import f35_r1_floor as R  # noqa: E402
import f35_r1_frames as F  # noqa: E402
from oracle_ceiling import ESSENCE_AXES  # noqa: E402

K_SIGMA = 5.0
MIN_AREA = 2                    # px; a 5 sigma noise pixel is ~3e-7 per pixel
CORE_FRAC = 0.85                # of the box interior level, for the box core (two bodies)
CORE_MIN = 6                    # px; the ball's specular highlight stays below this
EIGHT = np.ones((3, 3), int)
STEP = F.FULL_WELL / F.ADC_MAX  # e- per ADC count
BOX_LEVEL = 128 / 255 * F.E_WHITE / STEP   # interior box pixel, ADC counts


def background_model(bg_pair: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expected ADC value and noise sd per pixel of the empty scene."""
    e = F.merge(bg_pair) / 255.0 * F.E_WHITE
    sd = np.sqrt(e + F.READ_E ** 2 + STEP ** 2 / 12) / STEP
    return e / STEP, sd


def _moments(d: np.ndarray, mask: np.ndarray) -> tuple[float, float, complex, float]:
    rows, cols = np.nonzero(mask)
    w = np.clip(d[rows, cols], 0.0, None)
    sw = w.sum()
    if sw <= 0:
        return np.nan, np.nan, 0j, 0.0
    c, r = (w * cols).sum() / sw, (w * rows).sum() / sw
    z = (cols - c) - 1j * (rows - r)                 # image up = +imaginary
    den = (w * np.abs(z) ** 4).sum()
    m4 = (w * z ** 4).sum() / den if den > 0 else 0j
    return c, r, m4, float(len(rows))


def track_frame(img: np.ndarray, bg_mean: np.ndarray, thr: np.ndarray,
                two_bodies: bool) -> dict:
    """{body: (col, row, m4, area)} for the bodies found in one frame."""
    d = img.astype(np.float64) - bg_mean
    lab, n = ndimage.label(d > thr, structure=EIGHT)
    if n == 0:
        return {}
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    areas[0] = 0
    if not two_bodies:
        k = int(areas.argmax())
        return {"box": _moments(d, lab == k)} if areas[k] >= MIN_AREA else {}
    out = {}
    core, nc = ndimage.label(d > CORE_FRAC * BOX_LEVEL, structure=EIGHT)
    box = np.zeros_like(lab, bool)
    if nc:
        ca = np.bincount(core.ravel(), minlength=nc + 1)
        ca[0] = 0
        k = int(ca.argmax())
        if ca[k] >= CORE_MIN:
            box = ndimage.binary_dilation(core == k, EIGHT, iterations=2) & (lab > 0)
            out["box"] = _moments(d, box)
    rest = (lab > 0) & ~box
    if rest.any():
        lab2, n2 = ndimage.label(rest, structure=EIGHT)
        a2 = np.bincount(lab2.ravel(), minlength=n2 + 1)
        a2[0] = 0
        k = int(a2.argmax())
        if a2[k] >= MIN_AREA:
            out["ball"] = _moments(d, lab2 == k)
    return out


def _px_to_plane(col: float, row: float, win: tuple, w: float) -> tuple[float, float]:
    u_lo, _, _, v_hi = win
    return u_lo + (col + 0.5) * w / R.PX, v_hi - (row + 0.5) * w / R.PX


def track_scene(job: tuple) -> dict:
    """One scene's recordings -> {rec: (F, k) features}, plus centroid
    errors against projected T0 for the tracker's own accuracy report."""
    fam, w, scene, pairs, bgs, projs, noise_seed = job
    win = R.window(R.FAMILIES[fam]["place"], w)
    two = len(R.FAMILIES[fam]["recs"]) == 2 and fam == "strike"
    out, err = {}, {}
    for rec, pair in pairs.items():
        rng = np.random.default_rng([noise_seed, scene, zlib.crc32(rec.encode())])
        frames = F.sensor(pair, rng)
        bg_mean, bg_sd = background_model(bgs[rec])
        thr = K_SIGMA * bg_sd
        bodies = ["box", "ball"] if two else ["box"]
        feats = {b: np.zeros((len(frames), 5 if b == "box" else 3)) for b in bodies}
        cents = {b: np.full((len(frames), 2), np.nan) for b in bodies}
        for t, img in enumerate(frames):
            for b, (c, r, m4, _a) in track_frame(img, bg_mean, thr, two).items():
                u, v = _px_to_plane(c, r, win, w)
                feats[b][t, :2] = (u, v)
                if b == "box":
                    feats[b][t, 2:4] = (m4.real, m4.imag)
                feats[b][t, -1] = 1.0
                cents[b][t] = (c, r)
        names = {"box": rec, "ball": "strike_ball"}
        for b in bodies:
            out[names[b]] = feats[b]
            err[names[b]] = cents[b] - projs[names[b]]
    return {"feats": out, "err": err}


def features(fam: str, w: float, frames_path: str, noise_seed: int, workers: int):
    """(S, D) flattened F_track features for an input family, and per-
    recording centroid errors (S, F, 2) against projected T0."""
    z = np.load(frames_path)
    recs = [r for r, _ in F._family_recs(fam)]
    rendered = [r for r in recs if f"frames_{r}" in z.files]
    arrs = {r: z[f"frames_{r}"] for r in rendered}
    bgs = {r: z[f"background_{r}"] for r in rendered}
    projs = {r: z[f"proj_{r}"] for r in recs}
    n = len(next(iter(arrs.values())))
    jobs = [(fam, w, s, {r: arrs[r][s] for r in rendered}, bgs,
             {r: projs[r][s] for r in recs}, noise_seed) for s in range(n)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(track_scene, jobs, chunksize=4))
    X = np.concatenate([np.stack([r["feats"][rec] for r in res]).reshape(n, -1)
                        for rec in recs], 1)
    err = {rec: np.stack([r["err"][rec] for r in res]) for rec in recs}
    return X, err, z


def tracker_accuracy(err: np.ndarray, cent: np.ndarray) -> dict:
    """Centroid error vs projected T0 on frames where the body is fully in
    view and not occluded (cent: f35_r1_frames' per-frame geometry)."""
    full = (cent[..., 7] == 0) & (cent[..., 5] >= cent[..., 6] - 1e-6) & (cent[..., 6] > 0)
    e = np.linalg.norm(err, axis=-1)
    found = full & np.isfinite(e)
    return {"rms_px_full_view": float(np.sqrt(np.nanmean(e[found] ** 2))) if found.any() else None,
            "p99_px_full_view": float(np.nanpercentile(e[found], 99)) if found.any() else None,
            "missed_full_view_frac": float(1 - found.sum() / max(1, full.sum()))}


def diagnose(sets: list[str], battery: str, frames: str, workers: int) -> dict:
    """Why F_track exceeds F_px (§10.2a, step (d)): the floor at smaller
    pixel noise, the floor without its up-vector columns, and F_track
    without m4 and/or with the floor's visibility rule (zero once the true
    centroid leaves the image). Noise seed 1, iid test split."""
    man = json.load(open(os.path.join(battery, "manifest.json")))
    z = np.load(os.path.join(battery, "battery.npz"))
    T = {p: z[f"traj_{p}"].astype(float) for p in man["probes"]}
    f, fam_of, _ = O.compute(battery)
    targets = {t: np.stack([f[n] for n in f if fam_of[n] == t], 1) for t in set(fam_of.values())}
    n = len(z["essence"])
    tr, te = R.split(n)
    out: dict = {}

    def cells(X, fam):
        return {t: R.held_out(X, targets[t], tr, te)["test"] for t in R.PAIRS[fam]}
    sigma0 = R.SIGMA_PX
    for s in sets:
        fam, w = s.split("_W")
        w = float(w)
        nrec = len(R.FAMILIES[fam]["recs"])
        rows = {}
        try:
            for sig in (0.5, 0.08):
                R.SIGMA_PX = sig
                X = R.features(T, fam, w, 1, man["dropramp_deg"])
                rows[f"F_px sigma {sig} px"] = cells(X, fam)
                Xr = X.reshape(n, nrec, -1, 6)
                rows[f"F_px sigma {sig} px, no up"] = cells(Xr[..., [0, 1, 5]].reshape(n, -1), fam)
        finally:
            R.SIGMA_PX = sigma0
        X, _, fz = features(fam, w, os.path.join(frames, f"frames_{fam}_W{w:g}.npz"), 1, workers)
        recs = [r for r, _ in F._family_recs(fam)]
        blocks, off = {}, 0
        for r in recs:
            k = 3 if r == "strike_ball" else 5
            blocks[r] = X[:, off:off + 250 * k].reshape(n, 250, k)
            off += 250 * k

        def build(no_m4: bool, centroid_rule: bool, recs=recs, blocks=blocks,
                  fz=fz) -> np.ndarray:
            parts = []
            for r in recs:
                b = blocks[r].copy()
                if centroid_rule:
                    p = fz[f"proj_{r}"]
                    b[~((p >= -0.5) & (p <= R.PX - 0.5)).all(-1)] = 0.0
                if no_m4 and b.shape[-1] == 5:
                    b = b[..., [0, 1, 4]]
                parts.append(b.reshape(n, -1))
            return np.concatenate(parts, 1)
        for name, kw in (("F_track", (False, False)), ("F_track, no m4", (True, False)),
                         ("F_track, centroid rule", (False, True)),
                         ("F_track, no m4, centroid rule", (True, True))):
            rows[name] = cells(build(*kw), fam)
        out[s] = rows
        print(f"[r1-track diagnose] {s}")
        for name, row in rows.items():
            print(f"  {name:<32}" + "  ".join(f"{t} {v:+.3f}" for t, v in row.items()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--sets", nargs="+", required=True, help="e.g. kick_W2 dropramp_W0.5")
    ap.add_argument("--noise-seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    ap.add_argument("--diagnose", action="store_true",
                    help="only the F_px / F_track ablations (noise seed 1)")
    args = ap.parse_args()
    if args.diagnose:
        res = diagnose(args.sets, args.battery, args.frames, args.workers)
        out = args.out or os.path.join(args.frames, "r1_track_diagnose.json")
        json.dump(res, open(out, "w"), indent=2)
        print(f"-> {out}")
        return

    man = json.load(open(os.path.join(args.battery, "manifest.json")))
    E = np.load(os.path.join(args.battery, "battery.npz"))["essence"].astype(float)
    f, fam_of, _ = O.compute(args.battery)
    targets = {t: np.stack([f[n] for n in f if fam_of[n] == t], 1) for t in set(fam_of.values())}
    tr, te = R.split(len(E))
    report = {"battery": args.battery, "frames": args.frames,
              "battery_provenance": man.get("provenance"), "provenance": F.B.provenance(),
              "fields_version": O.FIELDS_VERSION, "noise_seeds": args.noise_seeds,
              "k_sigma": K_SIGMA, "min_area": MIN_AREA, "core_frac": CORE_FRAC,
              "core_min": CORE_MIN, "sets": {}}
    for s in args.sets:
        fam, w = s.split("_W")
        w = float(w)
        path = os.path.join(args.frames, f"frames_{fam}_W{w:g}.npz")
        t0 = time.time()
        per_seed, acc = [], {}
        for ns in args.noise_seeds:
            X, err, z = features(fam, w, path, ns, args.workers)
            row = {}
            for t, axis in R.PAIRS[fam].items():
                Y = targets[t]
                a = E[:, ESSENCE_AXES.index(axis)]
                ctr = np.where(a < np.quantile(a, 0.8))[0]
                cte = np.where(a >= np.quantile(a, 0.8))[0]
                row[t] = {"iid": R.held_out(X, Y, tr, te), "corner": R.held_out(X, Y, ctr, cte)}
            per_seed.append(row)
            if ns == args.noise_seeds[0]:
                acc = {rec: tracker_accuracy(e, z[f"cent_{rec}"]) for rec, e in err.items()}
        summ = {}
        for t in R.PAIRS[fam]:
            test = np.array([p[t]["iid"]["test"] for p in per_seed])
            cv = np.array([p[t]["iid"]["cv"] for p in per_seed])
            corner = np.array([p[t]["corner"]["skill_vs_train_var"] for p in per_seed])
            summ[t] = {"F_track_test": test.tolist(), "F_track_test_mean": float(test.mean()),
                       "F_track_test_sd": float(test.std(ddof=1)) if len(test) > 1 else 0.0,
                       "F_track_cv_mean": float(cv.mean()),
                       "corner_skill_vs_train_var": corner.tolist(),
                       "corner_skill_mean": float(corner.mean()),
                       "method_seed0": per_seed[0][t]["iid"]["method"],
                       "detail_seed0": per_seed[0][t]}
        report["sets"][s] = {"family": fam, "w": w, "tracker_accuracy": acc, "pairs": summ,
                             "seconds": round(time.time() - t0, 1)}
        print(f"[r1-track] {fam} W={w:g} m ({report['sets'][s]['seconds']} s, "
              f"{len(args.noise_seeds)} noise seed(s))")
        for rec, a in acc.items():
            print(f"  tracker {rec:<14} rms {a['rms_px_full_view']:.3f} px (p99 "
                  f"{a['p99_px_full_view']:.3f}) on fully visible frames; missed "
                  f"{a['missed_full_view_frac']:.4f}")
        for t, v in summ.items():
            print(f"  {fam}->{t:<5} F_track test {v['F_track_test_mean']:+.3f} "
                  f"(sd {v['F_track_test_sd']:.3f}, {v['method_seed0']}), CV "
                  f"{v['F_track_cv_mean']:+.3f}; corner skill {v['corner_skill_mean']:+.3f}")
    out = args.out or os.path.join(args.frames, "r1_track.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
