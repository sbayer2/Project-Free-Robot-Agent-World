#!/usr/bin/env python
"""Is the F22a coupling->Delta curve non-monotonic? (stage-1.5 confirmation)

The 3-seed sweep suggested Delta PEAKS at intermediate coupling (r~0.64,
Delta +0.925) and falls away toward maximum coupling (r~0.99, +0.577) --
Welch t ~ 2.2-2.9, non-overlapping seed ranges, but only n=3. If it holds it
is the most useful result of the arc: Option B should target r ~ 0.6 rather
than maximising legibility. If it dissolves, the curve is a plateau above
r ~ 0.6 and the guidance is simply "get past 0.6".

DECISION RULE, fixed before the confirmation seeds were read:

  NON-MONOTONIC CONFIRMED  if BOTH
      (a) Welch t(peak vs r=0.99) >= 2.5 with the peak higher, AND
      (b) peak mean exceeds r=0.99 mean by >= 0.20 (an effect worth acting on)
  PLATEAU (peak refuted)   if |peak - r=0.99| < 0.15
  INCONCLUSIVE             otherwise -> report as provisional, do not claim

Welch (unequal variance) rather than Student, because the arms' spreads differ
by ~2x. One-sided, since the hypothesis is directional (peak is HIGHER).

Usage:  .venv/bin/python scripts/f22_peak_test.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from oracle_ceiling import best_oracle  # noqa: E402
from oracle_ceiling import gain as gain_ratio  # noqa: E402
from probe_appearance import encode_z  # noqa: E402
from render_fidelity_eval import behavior_from_z, load_arrays, shape_onehot  # noqa: E402

# (label, r, dataset, checkpoint tag)
POINTS = [
    ("r=0.636 (g1.5) PEAK", 0.636, "data/pm_f22_g15", "g15_cont"),
    ("r=0.809 (g2)", 0.809, "data/pm_f22_g2", "g2_cont"),
    ("r=0.992 (g8) MAX", 0.992, "data/pm_f22_loud", "loud_cont"),
]


def welch_t(x, y):
    """One-sided Welch t for mean(x) > mean(y)."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vx, vy = st.variance(x), st.variance(y)
    se = math.sqrt(vx / len(x) + vy / len(y))
    return (st.mean(x) - st.mean(y)) / se if se > 0 else float("nan")


def main() -> None:
    from pseudomarble.config import ModelConfig
    from pseudomarble.models.mlx_net import build_model

    out: dict = {"points": {}}
    print(f"{'point':24s} {'n':>2s} {'shape_or':>9s} {'gain':>16s} {'DELTA':>9s}")
    deltas: dict = {}

    for label, r, data, tag in POINTS:
        cks = sorted(glob.glob(f"runs/f22/{tag}_s*/model.safetensors"))
        if not cks:
            print(f"{label:24s}  -- no checkpoints")
            continue
        imgs, Yb, Ya, tr, te, ds = load_arrays(data)
        S = shape_onehot(ds)
        shape_gain = best_oracle(S[tr], Yb[tr], S[te], Yb[te], quad=False)[0]

        ds_ = []
        for ck in cks:
            m = build_model(ModelConfig(image_size=128))
            m.load_weights(ck)
            beh = behavior_from_z(m, encode_z(m, imgs))
            ds_.append(gain_ratio(Yb[tr], Yb[te], beh[te]) - shape_gain)
        deltas[label] = ds_
        out["points"][label] = {"r": r, "n": len(ds_), "shape_oracle": float(shape_gain),
                                "delta_mean": float(np.mean(ds_)),
                                "delta_sd": float(st.stdev(ds_)) if len(ds_) > 1 else 0.0,
                                "delta_per_seed": [float(x) for x in ds_]}
        print(f"{label:24s} {len(ds_):2d} {shape_gain:9.3f} "
              f"{np.mean(ds_) + shape_gain:10.3f}+/-{np.std(ds_):.3f} "
              f"{np.mean(ds_):+9.3f}")

    peak = deltas.get("r=0.636 (g1.5) PEAK")
    mx = deltas.get("r=0.992 (g8) MAX")
    mid = deltas.get("r=0.809 (g2)")
    if peak and mx:
        t_max = welch_t(peak, mx)
        d_max = st.mean(peak) - st.mean(mx)
        t_mid = welch_t(peak, mid) if mid else float("nan")
        print(f"\npeak vs MAX : diff {d_max:+.3f}  Welch t {t_max:.2f}  "
              f"(n={len(peak)} vs {len(mx)})")
        if mid:
            print(f"peak vs r=0.81: diff {st.mean(peak) - st.mean(mid):+.3f}  "
                  f"Welch t {t_mid:.2f}")

        if t_max >= 2.5 and d_max >= 0.20:
            v = ("NON-MONOTONIC CONFIRMED -- Delta peaks at intermediate coupling. "
                 "Option B should TARGET r ~ 0.6, not maximise legibility.")
        elif abs(d_max) < 0.15:
            v = ("PLATEAU -- the peak does not survive more seeds. Above r ~ 0.6 the "
                 "curve is flat; guidance is simply 'get past 0.6'.")
        else:
            v = ("INCONCLUSIVE -- suggestive but below the prefixed bar. Report as "
                 "provisional; do NOT claim non-monotonicity.")
        print(f"\n[peak test] {v}")
        out["test"] = {"welch_t_vs_max": float(t_max), "diff_vs_max": float(d_max),
                       "verdict": v}

    json.dump(out, open("runs/f22/f22_peak_test.json", "w"), indent=2)
    print("\n[peak] wrote runs/f22/f22_peak_test.json")


if __name__ == "__main__":
    main()
