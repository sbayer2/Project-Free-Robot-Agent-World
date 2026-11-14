"""When did each seed leave the predict-the-mean plateau, and did it matter?

Reads each run's metrics.json (train_loss per epoch). A seed "escapes" at the
first epoch after epoch 1 whose train loss falls below the midpoint between its
epoch-1 loss and its final loss; a seed whose loss fell by less than
--min-drop never escaped. Optionally joins a probe_shape.py report to correlate
the escape epoch with B2 (gain over shape-only) and B3 (material use beyond
shape) across seeds.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def escape_epoch(train_loss: list[float], min_drop: float) -> int | None:
    tl = np.asarray(train_loss, dtype=float)
    if tl[1] - tl[-1] < min_drop:
        return None
    thr = 0.5 * (tl[1] + tl[-1])
    return next(e for e in range(1, len(tl)) if tl[e] < thr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs holding metrics.json")
    ap.add_argument("--shape-report", default=None, help="probe_shape.py JSON to join")
    ap.add_argument("--min-drop", type=float, default=0.05)
    args = ap.parse_args()

    B = {}
    if args.shape_report:
        for r in json.load(open(args.shape_report))["B_per_seed"]:
            B[os.path.basename(os.path.dirname(r["checkpoint"]))] = r

    rows = []
    for run in args.runs:
        h = json.load(open(os.path.join(run, "metrics.json")))["history"]
        tl = [e["train_loss"] for e in h]
        name = os.path.basename(os.path.normpath(run))
        esc = escape_epoch(tl, args.min_drop)
        r = B.get(name, {})
        rows.append((name, esc, tl[-1], r.get("B2_gain_over_shape_only"),
                     r.get("B3_within_shape_pearson")))
        print(f"{name:10s} escape {str(esc) if esc is not None else 'never':>5s}  "
              f"final loss {tl[-1]:.3f}  B2 {r.get('B2_gain_over_shape_only', float('nan')):.3f}  "
              f"B3 {r.get('B3_within_shape_pearson', float('nan')):+.3f}")

    esc = [r for r in rows if r[1] is not None]
    e = np.array([r[1] for r in esc], dtype=float)
    print(f"\nescaped {len(esc)}/{len(rows)}; escape epoch mean {e.mean():.1f} "
          f"median {np.median(e):.0f} range {int(e.min())}-{int(e.max())}")
    if B and len(esc) > 2:
        b2 = np.array([r[3] for r in esc], dtype=float)
        b3 = np.array([r[4] for r in esc], dtype=float)
        print(f"over escaped seeds: B2 {b2.mean():.3f} +- {b2.std():.3f}, "
              f"B3 {b3.mean():.3f} +- {b3.std():.3f}; corr(escape, B2) "
              f"{np.corrcoef(e, b2)[0, 1]:+.2f}, corr(escape, B3) {np.corrcoef(e, b3)[0, 1]:+.2f}")


if __name__ == "__main__":
    main()
