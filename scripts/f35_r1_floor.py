"""F35 R1 step (b): the T0 oracle floor at pixel precision, F_px, on the
camera window ladder, from T0 only, no frames (docs/CROSS_INTERACTION.md
§10.2a rules 1-7 and §10.5).

For each R1 input family, each admissible window W on the ladder gives
per-frame features per body: in-plane position (u, v), up vector and a
visibility bit. In view, position noise sigma = 0.5 px x W/96 and an angular
noise of the same pixel precision are added; out of view, the position is
clipped to the window edge, the up vector is zero, the bit is 0 and no noise
is added. F_px = mean over the target family's standardized outcome fields
of held-out R^2, best of ridge (alpha by inner CV) and kNN-5.

W* is chosen by 5-fold CV inside the train split (split seed 0, iid
410/102) and never on the test split; for a family feeding several pairs,
by the minimum across its pairs; rungs within TIE of the best are tied and
the largest tied W wins. The test-split F_px is reported at W* only, with
F_5mm (no framing, 5 mm) co-reported, and the corner split (top 20 % of the
pair's shared axis held out; per-field R^2 on the corner's own variance,
which is undefined where a target saturates there: see §10.2a, R1-W1
results). No noiseless reference is reported: with no noise, standardizing
near-constant columns lets sub-micron variation dominate the readout.

    python scripts/f35_r1_floor.py --data <frozen D7 battery, seed 1234>
    python scripts/f35_r1_floor.py --data <seed 1 battery> --w kick=2 drop=1 ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import f35_outcomes as O  # noqa: E402
from f35_code_table import add_noise, knn_predict, ridge_cv_predict  # noqa: E402
from oracle_ceiling import ESSENCE_AXES  # noqa: E402

PX = 96
LADDER = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)       # m, window width at the motion plane
MIN_BODY_PX = 2.0
SIGMA_PX = 0.5
HALF = 0.15                                     # m, the cube's half size
BODY_SIZE = {"box": 0.3, "ball": 0.1}           # m across
TIE = 0.01
TRAIN_FRAC = 0.8
CV_FOLDS = 5

# input family -> recordings as (recording, body), the camera plane and placement.
# plane "xz": u = x - x0, v = z (ground at 0); "ramp": u = -(y - y0) (downhill),
# v = z - z0. place: "trail" puts the box start 15 % of W from the trailing
# edge, "centre" centres it; the flat ground line sits 15 % of W from the bottom,
# "top" puts the box start 15 % of W from the top edge.
FAMILIES = {
    "kick": {"recs": (("kick1.0", "box"), ("kick3.0", "box")), "plane": "xz",
             "place": ("trail", "ground")},
    "drop": {"recs": (("drop", "box"),), "plane": "xz", "place": ("centre", "ground")},
    "strike": {"recs": (("strike", "box"), ("strike_ball", "ball")), "plane": "xz",
               "place": ("trail", "ground")},
    "dropramp": {"recs": (("dropramp{deg:04.1f}", "box"),), "plane": "ramp",
                 "place": ("trail", "top")},
}
# input family -> {target family: shared axis (corner split)}; the §10.4 pairs
# plus the drop onto the ramp's four (R1's pair list is decided at its freeze).
PAIRS = {
    "kick": {"ramp": "friction"},
    "drop": {"tap": "restitution"},
    "strike": {"scale": "density"},
    "dropramp": {"ramp": "friction", "kick": "friction", "drop": "restitution",
                 "tap": "restitution"},
}


def admissible(fam: str, w: float) -> bool:
    """Every tracked body spans >= MIN_BODY_PX at window width w."""
    return all(BODY_SIZE[b] * PX / w >= MIN_BODY_PX for _, b in FAMILIES[fam]["recs"])


def window(place: tuple[str, str], w: float) -> tuple[float, float, float, float]:
    """(u_lo, u_hi, v_lo, v_hi) in plane coordinates (box start at u = 0; v as
    in FAMILIES: ground at 0 for "ground", box start at 0 for "top")."""
    h, vert = place
    u_lo = -0.5 * w if h == "centre" else -0.15 * w
    v_lo = -0.15 * w if vert == "ground" else -0.85 * w
    return u_lo, u_lo + w, v_lo, v_lo + w


def plane_coords(r: np.ndarray, plane: str, origin: np.ndarray) -> np.ndarray:
    """(S, F, 7) [t, x, y, z, up] -> (S, F, 2) [u, v]; origin = the box's
    frame-0 position (S, 3)."""
    if plane == "xz":
        return np.stack([r[..., 1] - origin[:, None, 0], r[..., 3]], -1)
    return np.stack([-(r[..., 2] - origin[:, None, 1]), r[..., 3] - origin[:, None, 2]], -1)


def rotate(up: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate each up vector by an angle ~ N(0, sigma) about a random axis."""
    axis = rng.normal(size=up.shape)
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    ang = rng.normal(0.0, sigma, size=up.shape[:-1])[..., None]
    c, s = np.cos(ang), np.sin(ang)
    return (up * c + np.cross(axis, up) * s
            + axis * (axis * up).sum(-1, keepdims=True) * (1.0 - c))


def framed(uv: np.ndarray, up: np.ndarray, win: tuple, w: float,
           rng: np.random.Generator | None) -> np.ndarray:
    """(S, F, 2) + (S, F, 3) -> (S, F, 6) [u, v, up, visible] under rule 4.
    rng None: no noise (the noiseless framing)."""
    u_lo, u_hi, v_lo, v_hi = win
    vis = ((uv[..., 0] >= u_lo) & (uv[..., 0] <= u_hi)
           & (uv[..., 1] >= v_lo) & (uv[..., 1] <= v_hi))
    pos, ori = uv.copy(), up.copy()
    if rng is not None:
        m_px = w / PX
        pos = pos + rng.normal(0.0, SIGMA_PX * m_px, size=pos.shape)
        ori = rotate(ori, SIGMA_PX * m_px / HALF, rng)
        pos = np.where(vis[..., None], pos, uv)          # no noise out of view
        ori = np.where(vis[..., None], ori, up)
    pos = np.stack([np.clip(pos[..., 0], u_lo, u_hi), np.clip(pos[..., 1], v_lo, v_hi)], -1)
    ori = np.where(vis[..., None], ori, 0.0)
    return np.concatenate([pos, ori, vis[..., None].astype(float)], -1)


def features(T: dict[str, np.ndarray], fam: str, w: float | None, seed: int,
             dropramp_deg: float) -> np.ndarray:
    """Flattened per-scene features of an input family. w = 0: F_5mm, T0
    (pos - pos0, up) with the §5 gate noise and no framing. w None: the same
    without noise (diagnostics only)."""
    spec = FAMILIES[fam]
    box = T[spec["recs"][0][0].format(deg=dropramp_deg)]
    origin = box[:, 0, 1:4]
    win = None if not w else window(spec["place"], w)
    cols = []
    for i, (rec, _) in enumerate(spec["recs"]):
        r = T[rec.format(deg=dropramp_deg)]
        if not w:
            X = np.concatenate([r[..., 1:4] - r[:, :1, 1:4], r[..., 4:7]], -1)
            cols.append(X if w is None else add_noise(X, 5.0, seed=seed + i))
            continue
        rng = np.random.default_rng([seed, i, int(round(w * 1000))])
        cols.append(framed(plane_coords(r, spec["plane"], origin), r[..., 4:7], win, w, rng))
    return np.concatenate([c.reshape(len(c), -1) for c in cols], 1)


def r2_mean(Y: np.ndarray, P: np.ndarray) -> float:
    tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return float(np.mean(1.0 - ((Y - P) ** 2).sum(0) / np.where(tot > 0, tot, 1.0)))


def fit_predict(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, method: str,
                seed: int = 0) -> np.ndarray:
    """Columns standardized on train; targets standardized on train."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym, ys = Ytr.mean(0), np.where(Ytr.std(0) < 1e-12, 1.0, Ytr.std(0))
    Z = (Ytr - ym) / ys
    P = ridge_cv_predict(A, Z, B, seed=seed) if method == "ridge" else knn_predict(A, Z, B)
    return P * ys + ym


def cv_scores(X: np.ndarray, Y: np.ndarray, idx: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Out-of-fold mean R^2 inside idx, per method."""
    folds = np.array_split(np.random.default_rng(seed).permutation(idx), CV_FOLDS)
    out = {}
    for m in ("ridge", "knn"):
        P = np.zeros((len(X), Y.shape[1]))
        for k in range(CV_FOLDS):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(CV_FOLDS) if j != k])
            P[te] = fit_predict(X[tr], Y[tr], X[te], m, seed=seed + k)
        out[m] = r2_mean(Y[idx], P[idx])
    return out


def held_out(X: np.ndarray, Y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> dict:
    """Method chosen by CV inside tr, refit on all of tr, scored on te."""
    cv = cv_scores(X, Y, tr)
    m = max(cv, key=cv.get)
    return {"method": m, "cv": cv[m], "test": r2_mean(Y[te], fit_predict(X[tr], Y[tr], X[te], m))}


def choose_w(cv_min: dict[float, float], tie: float = TIE) -> float:
    """The largest rung within tie of the best (rule 6)."""
    best = max(cv_min.values())
    return max(w for w, v in cv_min.items() if v >= best - tie)


def split(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    p = np.random.default_rng(seed).permutation(n)
    k = int(round(TRAIN_FRAC * n))
    return np.sort(p[:k]), np.sort(p[k:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--w", nargs="*", default=[],
                    help="fam=W: skip the choice and evaluate at this W (other seeds)")
    ap.add_argument("--noise-seed", type=int, default=1)
    args = ap.parse_args()
    fixed = {k: float(v) for k, v in (a.split("=") for a in args.w)}

    z = np.load(os.path.join(args.data, "battery.npz"))
    man = json.load(open(os.path.join(args.data, "manifest.json")))
    T = {p: z[f"traj_{p}"].astype(float) for p in man["probes"]}
    E = z["essence"].astype(float)
    f, fam_of, _ = O.compute(args.data)
    targets = {t: np.stack([f[n] for n in f if fam_of[n] == t], 1) for t in set(fam_of.values())}
    tr, te = split(len(E))
    report = {"data": args.data, "provenance": man.get("provenance"),
              "fields_version": O.FIELDS_VERSION, "split": [len(tr), len(te)],
              "ladder": LADDER, "families": {}}
    print(f"[r1-floor] {args.data}: {len(E)} scenes, split {len(tr)}/{len(te)}, fields "
          f"{O.FIELDS_VERSION}, mujoco {(man.get('provenance') or {}).get('mujoco')}")
    for fam in args.families:
        pairs = PAIRS[fam]
        rungs = [fixed[fam]] if fam in fixed else [w for w in LADDER if admissible(fam, w)]
        ladder = {}
        for w in rungs:
            X = features(T, fam, w, args.noise_seed, man["dropramp_deg"])
            ladder[w] = {t: cv_scores(X, targets[t], tr) for t in pairs}
        cv_min = {w: min(max(v.values()) for v in row.values()) for w, row in ladder.items()}
        w_star = choose_w(cv_min)
        print(f"\n{fam} -> {', '.join(pairs)}  (CV inside train; min over pairs)")
        for w, row in ladder.items():
            cells = "  ".join(f"{t} {max(v.values()):+.3f}" for t, v in row.items())
            print(f"  W {w:5.1f} m  {PX * 0.3 / w:5.1f} px box  min {cv_min[w]:+.3f}  {cells}"
                  f"{'   <- W*' if w == w_star else ''}")
        X = features(T, fam, w_star, args.noise_seed, man["dropramp_deg"])
        X5 = features(T, fam, 0, args.noise_seed, man["dropramp_deg"])
        at = {}
        for t, axis in pairs.items():
            Y = targets[t]
            a = E[:, ESSENCE_AXES.index(axis)]
            ctr, cte = np.where(a < np.quantile(a, 0.8))[0], np.where(a >= np.quantile(a, 0.8))[0]
            at[t] = {"F_px": held_out(X, Y, tr, te), "F_5mm": held_out(X5, Y, tr, te),
                     "F_px_corner": held_out(X, Y, ctr, cte), "shared_axis": axis}
            print(f"  test at W* {w_star:g} m: {fam}->{t}  F_px {at[t]['F_px']['test']:+.3f} "
                  f"({at[t]['F_px']['method']})  F_5mm {at[t]['F_5mm']['test']:+.3f}  "
                  f"corner F_px "
                  f"{at[t]['F_px_corner']['test']:+.3f}")
        report["families"][fam] = {
            "w_star": w_star, "m_per_px": w_star / PX, "chosen": fam not in fixed,
            "body_px": {b: BODY_SIZE[b] * PX / w_star for _, b in FAMILIES[fam]["recs"]},
            "ladder_cv": {str(w): row for w, row in ladder.items()},
            "ladder_cv_min": {str(w): v for w, v in cv_min.items()}, "at_w_star": at}
    out = args.out or os.path.join(args.data, "r1_floor.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
