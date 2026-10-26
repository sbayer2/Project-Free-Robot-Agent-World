"""F35 design pilot: which interaction encodes which material axis, and where
in the interaction the information lives. Zero training, state trajectories
(no pixels), on a --keep-trajectory world (docs/CROSS_INTERACTION.md §4).

Two readouts:

1. CODE TABLE: inverse R^2 of each essence axis read back out of ONE probe's
   trajectory (pos relative to frame 0 + up vector, resampled), best of ridge
   and kNN, 5-fold, under Gaussian position noise of 0 / 1 / 5 / 20 mm. The
   simulator is noiseless, so without noise any sub-millimetre difference is
   perfectly readable; the noise level is the channel.
2. KEYFRAMES (drop probe): the same readout from frame windows -- pure free
   fall, the first frames of contact, everything after -- and from fixed
   random frame subsets shared by every scene (the erasure question: does
   any k of n frames recover the axis?).

    python scripts/f35_code_table.py --data data/pm_f31fix_loud
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
from oracle_ceiling import ESSENCE_AXES, knn_fit_predict, ridge_fit_predict  # noqa: E402

PROBES = ("drop", "tilt", "push")
NOISE_MM = (0.0, 1.0, 5.0, 20.0)
UP_NOISE = 0.01   # added to the up-vector channels whenever position noise is on


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


def inverse_r2(X: np.ndarray, E: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Per-axis out-of-fold R^2 of reading E from flattened X, best of ridge
    (alpha 1) and kNN-5, columns standardized on each training fold."""
    X = X.reshape(len(X), -1)
    folds = np.array_split(np.random.default_rng(seed).permutation(len(X)), k)
    best = None
    for fit in (lambda a, b, c: ridge_fit_predict(a, b, c, alpha=1.0), knn_fit_predict):
        P = np.zeros_like(E)
        for i in range(k):
            te = folds[i]
            tr = np.concatenate([folds[j] for j in range(k) if j != i])
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            P[te] = fit((X[tr] - mu) / sd, E[tr], (X[te] - mu) / sd)
        r2 = 1.0 - ((E - P) ** 2).sum(0) / ((E - E.mean(0)) ** 2).sum(0)
        best = r2 if best is None else np.maximum(best, r2)
    return best


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


def fmt(r: np.ndarray) -> str:
    return " ".join(f"{a[:4]} {v:+.2f}" for a, v in zip(ESSENCE_AXES, r, strict=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    E, T, times, height = load(args.data, args.frames)
    report: dict = {"data": args.data, "frames": args.frames, "code_table": {}, "keyframes": {}}
    print(f"[f35] {args.data}: {len(E)} scenes, {args.frames} frames per probe over "
          f"{times['drop'][-1]:.2f} s")

    print("\nCODE TABLE: axis read from ONE probe's trajectory (inverse R^2)")
    for noise in NOISE_MM:
        row = {p: inverse_r2(add_noise(X, noise), E).tolist() for p, X in T.items()}
        report["code_table"][f"{noise:g}mm"] = row
        print(f"  {noise:4g} mm  " + " | ".join(f"{p}: {fmt(np.array(r))}" for p, r in row.items()))

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
