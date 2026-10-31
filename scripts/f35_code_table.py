"""F35 design pilot: which interaction encodes which material axis, and where
in the interaction the information lives. Zero training, state trajectories
(no pixels), on either a --keep-trajectory world (sample.json per scene) or a
battery written by scripts/f35_generate_battery.py (battery.npz + manifest)
(docs/CROSS_INTERACTION.md §4-5).

Two readouts:

1. CODE TABLE: inverse R^2 of each essence axis read back out of ONE probe's
   trajectory (pos relative to frame 0 + up vector), best of ridge (alpha
   chosen per axis by inner 4-fold CV; D5, was fixed at 1) and kNN-5, 5-fold,
   under Gaussian position noise of 0 / 1 / 5 / 20 mm. The
   simulator is noiseless, so without noise any sub-millimetre difference is
   perfectly readable; the noise level is the channel.
2. KEYFRAMES (drop probe): the same readout from frame windows -- pure free
   fall, the first frames of contact, everything after -- and from fixed
   random frame subsets shared by every scene (the erasure question: does
   any k of n frames recover the axis?).
3. GATE (§5): probe FAMILIES read whole (every ramp angle, every push
   force, every kick speed, a reference ball with the box it hits, the
   scale plate with its box). For each ordered pair of families, the axes both read at inverse
   R^2 >= --gate-r2 at --gate-noise; an axis passes when at least one pair
   shares it, i.e. it has a positive transfer cell. Bar and noise are not
   frozen; every battery measured before the freeze commit is DESIGN
   (iteration log in docs/CROSS_INTERACTION.md).

    python scripts/f35_code_table.py --data data/pm_f31fix_loud
    python scripts/f35_code_table.py --data data/pm_f35_battery
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from oracle_ceiling import ESSENCE_AXES  # noqa: E402

PROBES = ("drop", "tilt", "push")
NOISE_MM = (0.0, 1.0, 5.0, 20.0)
UP_NOISE = 0.01   # added to the up-vector channels whenever position noise is on
ALPHAS = tuple(float(a) for a in 10.0 ** np.arange(-2, 6))   # 1e-2 ... 1e5


def resample(traj: list[dict], n: int) -> tuple[np.ndarray, np.ndarray]:
    """((n, 6) [pos - pos0, up], (n,) times) at n evenly spaced frame indices."""
    idx = np.linspace(0, len(traj) - 1, n).round().astype(int)
    p0 = np.asarray(traj[0]["pos"], float)
    rows = [np.concatenate([np.asarray(traj[i]["pos"], float) - p0,
                            np.asarray(traj[i]["up"], float)]) for i in idx]
    return np.stack(rows), np.array([traj[i]["t"] for i in idx], float)


def add_noise(X: np.ndarray, noise_mm: float, seed: int = 1) -> np.ndarray:
    """Gaussian noise on (..., 6) trajectories: noise_mm on the 3 position
    channels, UP_NOISE on the 3 up-vector channels; identity at 0 mm."""
    if noise_mm <= 0:
        return X
    rng = np.random.default_rng(seed)
    scale = np.r_[[noise_mm / 1000.0] * 3, [UP_NOISE] * 3]
    return X + rng.normal(size=X.shape) * scale


def first_contact_frame(times: np.ndarray, t_contact: float) -> int:
    """Index of the first resampled frame at or after t_contact."""
    return int(np.searchsorted(times, t_contact - 1e-9))


def ridge_path(A: np.ndarray, Y: np.ndarray, B: np.ndarray,
               alphas: tuple[float, ...] = ALPHAS) -> np.ndarray:
    """Ridge predictions for every alpha at once, (len(alphas), len(B), Y cols).
    Dual form: exact for column-centred A (the intercept is Y's mean), and its
    cost is set by the number of scenes, not features, so whole families at
    125 fps (thousands of columns) stay cheap."""
    ym = Y.mean(0)
    w, V = np.linalg.eigh(A @ A.T)
    VtY = V.T @ (Y - ym)
    KB = B @ A.T
    return np.stack([KB @ (V @ (VtY / (w + a)[:, None])) + ym for a in alphas])


def ridge_cv_predict(A: np.ndarray, Y: np.ndarray, B: np.ndarray, seed: int = 0) -> np.ndarray:
    """Ridge with alpha chosen per output column by inner 4-fold CV on (A, Y).
    A fixed alpha overfits once a family's columns outnumber its scenes (D4:
    the strike family read friction at -1.0)."""
    inner = np.array_split(np.random.default_rng(seed).permutation(len(A)), 4)
    sse = np.zeros((len(ALPHAS), Y.shape[1]))
    for m in range(len(inner)):
        te = inner[m]
        tr = np.concatenate([inner[j] for j in range(len(inner)) if j != m])
        mu = A[tr].mean(0)
        sse += ((ridge_path(A[tr] - mu, Y[tr], A[te] - mu) - Y[te]) ** 2).sum(1)
    P = ridge_path(A, Y, B)
    best = sse.argmin(0)
    return P[best, :, np.arange(Y.shape[1])].T


def knn_predict(A: np.ndarray, Y: np.ndarray, B: np.ndarray, k: int = 5) -> np.ndarray:
    """kNN-k mean, with squared distances from Gram products (memory n_te x n_tr,
    not n_te x n_tr x features)."""
    d = (B ** 2).sum(1)[:, None] + (A ** 2).sum(1)[None, :] - 2.0 * B @ A.T
    idx = np.argsort(d, axis=1)[:, :min(k, len(A))]
    return Y[idx].mean(axis=1)


def inverse_r2(X: np.ndarray, E: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Per-axis out-of-fold R^2 of reading E from flattened X, best of ridge
    (alpha by inner CV) and kNN-5, columns standardized on each training fold."""
    X = X.reshape(len(X), -1)
    folds = np.array_split(np.random.default_rng(seed).permutation(len(X)), k)
    preds = {"ridge": np.zeros_like(E), "knn": np.zeros_like(E)}
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
        preds["ridge"][te] = ridge_cv_predict(A, E[tr], B, seed=seed + i)
        preds["knn"][te] = knn_predict(A, E[tr], B)
    tot = ((E - E.mean(0)) ** 2).sum(0)
    r2 = [1.0 - ((E - P) ** 2).sum(0) / tot for P in preds.values()]
    return np.maximum(*r2)


def load(data: str, n: int):
    """-> essence (S, 3), {probe: (S, n, 6)}, {probe: (n,) times}, drop height."""
    E, T, times, heights = [], {p: [] for p in PROBES}, {}, set()
    for f in sorted(glob.glob(os.path.join(data, "*", "sample.json"))):
        d = json.load(open(f))
        E.append([d["physics"]["normalized"][a] for a in ESSENCE_AXES])
        for p in d["behavior"]["probes"]:
            kind = p["spec"]["kind"]
            if "trajectory" not in p:
                raise SystemExit(f"{f}: no trajectory; regenerate with --keep-trajectory")
            X, t = resample(p["trajectory"], n)
            T[kind].append(X)
            times[kind] = t
            if kind == "drop":
                heights.add(p["spec"]["height"])
    if len(heights) != 1:
        raise SystemExit(f"drop heights vary across scenes ({heights}); keyframes need one")
    return np.array(E), {k: np.array(v) for k, v in T.items()}, times, heights.pop()


def load_npz(data: str, n: int | None):
    """The f35_generate_battery format -> the same tuple as load(). Recordings
    are (scenes, frames, [t, x, y, z, up_x, up_y, up_z]); positions are taken
    relative to frame 0, as resample() does. n = None keeps every recorded
    frame (the battery's native rate)."""
    man = json.load(open(os.path.join(data, "manifest.json")))
    if "drop_height" not in man:
        raise SystemExit(f"{data}/manifest.json has no drop_height; regenerate the battery")
    z = np.load(os.path.join(data, "battery.npz"))
    T, times = {}, {}
    for name in man["probes"]:
        a = z[f"traj_{name}"].astype(float)
        idx = (np.arange(a.shape[1]) if n is None
               else np.linspace(0, a.shape[1] - 1, n).round().astype(int))
        X = a[:, idx, 1:7].copy()
        X[:, :, :3] -= a[:, :1, 1:4]
        T[name], times[name] = X, a[0, idx, 0]
    return z["essence"].astype(float), T, times, float(man["drop_height"])


def family(name: str) -> str:
    """Probe family of a recording: ramp05.7 -> ramp, push0960 -> push,
    strike_ball -> strike, dropramp38.7 -> dropramp; single-recording probes
    are their own family."""
    for prefix in ("dropramp", "ramp", "push", "strike", "scale", "tap", "kick"):
        if name.startswith(prefix):
            return prefix
    return name


def families(T: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each family's recordings side by side along the feature axis."""
    out: dict[str, list] = {}
    for name, X in T.items():
        out.setdefault(family(name), []).append(X.reshape(len(X), -1))
    return {f: np.concatenate(v, 1) for f, v in out.items()}


def pair_matrix(table: dict[str, list[float]], bar: float) -> dict[str, list[str]]:
    """For each ordered pair "A->B" of distinct families, the axes BOTH read at
    >= bar: the axes on which watching A could predict B. Every other axis B
    reads is a non-shared cell, where transfer must sit at predict-the-mean."""
    out = {}
    for a, ra in table.items():
        for b, rb in table.items():
            if a != b:
                out[f"{a}->{b}"] = [x for j, x in enumerate(ESSENCE_AXES)
                                    if ra[j] >= bar and rb[j] >= bar]
    return out


def gate(table: dict[str, list[float]], bar: float) -> dict[str, dict]:
    """Per axis: the families reaching the bar, and PASS iff the axis is shared
    by at least one ordered pair in pair_matrix -- i.e. it has at least one
    positive transfer cell. (Equivalent to >= 2 families reading it; stated
    as pairs because the cross-prediction test is scored on pairs.)"""
    pairs = pair_matrix(table, bar)
    out = {}
    for j, a in enumerate(ESSENCE_AXES):
        ok = [f for f, r in table.items() if r[j] >= bar]
        cells = [p for p, axes in pairs.items() if a in axes]
        out[a] = {"families": ok, "pairs": cells, "pass": bool(cells)}
    return out


def strike_windows(data: str, T: dict[str, np.ndarray], E: np.ndarray,
                   noise_mm: float) -> dict[str, list[float]]:
    """Where the strike's signal lives: ball + box before the ball first
    touches the ground (the box-ball impact alone) vs the whole recording.
    D4's strike read restitution mostly from the ball's ground bounces,
    which replayed the box's restitution (fixed in D5)."""
    man = json.load(open(os.path.join(data, "manifest.json")))
    zb = np.load(os.path.join(data, "battery.npz"))["traj_strike_ball"][:, :, 3]
    if T["strike_ball"].shape[1] != zb.shape[1]:
        print("\nSTRIKE WINDOWS skipped: needs the battery's native frames (no --frames)")
        return {}
    on_ground = zb <= man["strike"]["radius"] + 0.003
    first = np.where(on_ground.any(1), on_ground.argmax(1), zb.shape[1])
    cut = int(first.min())
    rows = {}
    for name, hi in ((f"ball + box, before ground contact (frames 0-{cut - 1})", cut),
                     ("ball + box, whole recording", zb.shape[1])):
        X = np.concatenate([add_noise(T[p][:, :hi], noise_mm, seed=s).reshape(len(E), -1)
                            for s, p in enumerate(("strike_ball", "strike"), start=1)], 1)
        rows[name] = inverse_r2(X, E).tolist()
    print(f"\nSTRIKE WINDOWS at {noise_mm:g} mm (ball first touches the ground at frame "
          f"{cut}-{int(first.max())})")
    for name, r in rows.items():
        print(f"  {name:50s} {fmt(np.array(r))}")
    return rows


def fmt(r: np.ndarray) -> str:
    return " ".join(f"{a[:4]} {v:+.2f}" for a, v in zip(ESSENCE_AXES, r, strict=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--frames", type=int, default=None,
                    help="resample to this many frames (default: a battery's native "
                         "rate; 64 for sample.json worlds)")
    ap.add_argument("--gate-noise", type=float, default=5.0, help="mm; not frozen")
    ap.add_argument("--gate-r2", type=float, default=0.8, help="not frozen")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    npz = os.path.exists(os.path.join(args.data, "battery.npz"))
    E, T, times, height = (load_npz(args.data, args.frames) if npz
                           else load(args.data, args.frames or 64))
    args.frames = len(times["drop"])
    report: dict = {"data": args.data, "frames": args.frames, "code_table": {},
                    "per_recording": {}, "keyframes": {}}
    print(f"[f35] {args.data}: {len(E)} scenes, {len(T)} recordings, {args.frames} frames "
          f"each over {times['drop'][-1]:.2f} s")

    fam = families(T)
    print("\nCODE TABLE: axis read from ONE probe family's trajectories (inverse R^2)")
    for noise in NOISE_MM:
        row = {f: inverse_r2(add_noise(X.reshape(len(X), -1, 6), noise), E).tolist()
               for f, X in fam.items()}
        report["code_table"][f"{noise:g}mm"] = row
        print(f"  {noise:4g} mm  " + " | ".join(f"{f}: {fmt(np.array(r))}" for f, r in row.items()))
    if len(fam) < len(T):
        print(f"\nPER RECORDING at {args.gate_noise:g} mm")
        for name, X in T.items():
            r = inverse_r2(add_noise(X, args.gate_noise), E)
            report["per_recording"][name] = r.tolist()
            print(f"  {name:12s} {fmt(r)}")

    table = {f: inverse_r2(add_noise(X.reshape(len(X), -1, 6), args.gate_noise), E).tolist()
             for f, X in fam.items()}
    report["gate"] = {"noise_mm": args.gate_noise, "bar": args.gate_r2,
                      "axes": gate(table, args.gate_r2)}
    report["gate"]["pairs"] = pair_matrix(table, args.gate_r2)
    shared = {p: ax for p, ax in report["gate"]["pairs"].items() if ax}
    print(f"\nPAIRS sharing an axis at R^2 >= {args.gate_r2:g}, {args.gate_noise:g} mm "
          f"({len(shared)} of {len(report['gate']['pairs'])} ordered pairs):")
    for p, ax in shared.items():
        print(f"  {p:16s} {', '.join(ax)}")
    print("\nGATE (every axis shared by >= 1 pair; bar and noise NOT frozen)")
    for a, g in report["gate"]["axes"].items():
        print(f"  {a:12s} {'PASS' if g['pass'] else 'FAIL'}  families: {g['families'] or 'none'}")

    if npz and "strike_ball" in T:
        report["strike_windows"] = strike_windows(args.data, T, E, args.gate_noise)

    # Free fall from rest covers `height` (the gap to the ground) in sqrt(2h/g).
    t_contact = float(np.sqrt(2.0 * height / 9.81))
    c = first_contact_frame(times["drop"], t_contact)
    D = T["drop"]
    windows = {f"free fall (frames 0-{c - 1})": np.arange(0, c),
               f"first 4 frames of contact ({c}-{c + 3})": np.arange(c, c + 4),
               f"after contact ({c + 4}-{args.frames - 1})": np.arange(c + 4, args.frames)}
    print(f"\nKEYFRAMES (drop from {height} m; contact at t = {t_contact:.3f} s, frame {c})")
    for noise in (0.0, 1.0, 5.0):
        rows = {}
        for name, idx in windows.items():
            rows[name] = inverse_r2(add_noise(D[:, idx], noise), E).tolist()
        for k in (4, 8, 16):
            draws = [np.sort(np.random.default_rng(100 + j).choice(args.frames, k, replace=False))
                     for j in range(5)]
            rr = np.array([inverse_r2(add_noise(D[:, d], noise, seed=j), E)
                           for j, d in enumerate(draws)])
            rows[f"same random {k} frames, every scene (mean of 5 draws)"] = rr.mean(0).tolist()
        report["keyframes"][f"{noise:g}mm"] = rows
        print(f"  noise {noise:g} mm")
        for name, r in rows.items():
            print(f"    {name:50s} {fmt(np.array(r))}")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
