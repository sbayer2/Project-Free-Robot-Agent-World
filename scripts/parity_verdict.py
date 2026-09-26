"""Torch<->MLX parity gate verdict (docs/CLOUD_COMPUTE.md §3, frozen 2026-09-26).

Committed before any torch-trained number on the f27b loud world existed.
Reads the regenerated world (apparatus check) and the torch runs'
metrics.json, applies the frozen rule, prints and optionally writes a verdict.

    python scripts/parity_verdict.py --data data/pm_f31_loud \
        --runs runs/parity/loud_s0 runs/parity/loud_s1 runs/parity/loud_s2
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

# --- frozen constants (CLOUD_COMPUTE.md §3) ---------------------------------
MLX_LOUD_GAINS = (5.040, 5.910, 6.067)
MLX_LOUD_MEAN = 5.672
BAND = (4.82, 6.52)            # +/-15 % of MLX_LOUD_MEAN
MIN_SEED_GAIN = 3.0
MIN_PR = 8.0
REF_R = 0.992
R_TOL = 0.02
EXPECTED_SPLIT = {"train": 492, "test": 20}   # seed-1234 corner, verified 2026-09-26


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def apparatus(data_dir: str) -> dict:
    """Measured roughness<->friction r (F27's convention) and split sizes."""
    rough, fric, split = [], [], {"train": 0, "test": 0}
    for f in sorted(glob.glob(os.path.join(data_dir, "*", "sample.json"))):
        s = json.load(open(f))
        rough.append(s["material_truth"]["appearance_params"]["roughness"])
        fric.append(s["physics"]["raw"]["friction"])
        split[s["split"]] = split.get(s["split"], 0) + 1
    r = pearson(rough, fric)
    ok = abs(abs(r) - REF_R) <= R_TOL and split == EXPECTED_SPLIT
    return {"r": r, "split": split, "pass": ok}


def verdict(gains: list[float], prs: list[float]) -> dict:
    mean = sum(gains) / len(gains)
    checks = {
        "mean_in_band": BAND[0] <= mean <= BAND[1],
        "no_seed_below_min": min(gains) >= MIN_SEED_GAIN,
        "all_pr_healthy": min(prs) >= MIN_PR,
    }
    return {"mean_gain": mean, "gains": gains, "prs": prs, "checks": checks,
            "verdict": "PASS" if all(checks.values()) else "FAIL"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs=3, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    app = apparatus(args.data)
    print(f"[parity] apparatus: r={app['r']:+.4f} (ref {REF_R}±{R_TOL}), "
          f"split={app['split']} -> {'PASS' if app['pass'] else 'FAIL'}")
    report = {"apparatus": app, "reference": {"mlx_gains": MLX_LOUD_GAINS,
                                              "mlx_mean": MLX_LOUD_MEAN, "band": BAND}}
    if not app["pass"]:
        report["verdict"] = "WITHHELD (apparatus)"
        print("[parity] verdict WITHHELD: apparatus check failed; gains not read")
    else:
        finals = [json.load(open(os.path.join(r, "metrics.json")))["final"] for r in args.runs]
        v = verdict([f["heldout_gain"] for f in finals], [f["latent_pr"] for f in finals])
        report.update(v)
        print(f"[parity] torch gains {[f'{g:.3f}' for g in v['gains']]} "
              f"mean {v['mean_gain']:.3f} (band {BAND}); PR {[f'{p:.1f}' for p in v['prs']]}")
        for k, ok in v["checks"].items():
            print(f"[parity]   {k:20s} {'ok' if ok else 'FAIL'}")
        print(f"[parity] VERDICT: {v['verdict']}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
