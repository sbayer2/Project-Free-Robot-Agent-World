#!/usr/bin/env python
"""F22 stage-1 pilot: does a STRONGER coupling get learned? (docs/ABO_COUPLING.md)

Reads the four preregistered arms and reports the primary metric

    Delta = model_gain - shape_oracle_gain

never raw gain (F18: raw gain is mostly shape). Every gain here uses the same
helpers F21 used, so the numbers are directly comparable to F18/F21.

Arms:
  ctrl  a=0 g=1  continuous  -- zero-coupling CONTROL; P1 gates on |Delta| < 0.05
  base  a=1 g=1  continuous  -- historical coupling (the pm_big baseline)
  loud  a=1 g=8  continuous  -- bigger letters: the axis F21 never touched
  loud  a=1 g=8  FSQ k=1     -- the DECISIVE CELL (F17's forced unity on top)

Usage:  .venv/bin/python scripts/f22_pilot_eval.py [--root runs/f22]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from oracle_ceiling import best_oracle  # noqa: E402
from oracle_ceiling import gain as gain_ratio  # noqa: E402
from probe_appearance import encode_z  # noqa: E402
from render_fidelity_eval import behavior_from_z, load_arrays, shape_onehot  # noqa: E402

# (label, dataset, checkpoint tag, latent_trits)
ARMS = [
    ("ctrl  a=0 g=1", "data/pm_f22_ctrl", "ctrl_cont", 0),
    ("base  a=1 g=1", "data/pm_f22_base", "base_cont", 0),
    ("loud  a=1 g=8", "data/pm_f22_loud", "loud_cont", 0),
    ("loud  a=1 g=8 FSQ", "data/pm_f22_loud", "loud_fsq1", 1),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/f22")
    ap.add_argument("--image-size", type=int, default=128)
    args = ap.parse_args()

    from pseudomarble.config import ModelConfig
    from pseudomarble.models.mlx_net import build_model

    report: dict = {"arms": {}}
    print(f"{'arm':22s} {'n':>2s} {'shape_oracle':>13s} {'model_gain':>18s} "
          f"{'DELTA':>16s} {'PR':>7s}")

    for label, data, tag, trits in ARMS:
        cks = sorted(glob.glob(f"{args.root}/{tag}_s*/model.safetensors"))
        if not os.path.isdir(data) or not cks:
            print(f"{label:22s}  -- skipped (missing data or checkpoints)")
            continue

        imgs, Yb, Ya, tr, te, ds = load_arrays(data)

        # Shape-only oracle on the SAME corner split (F18's control).
        S = shape_onehot(ds)
        shape_gain = best_oracle(S[tr], Yb[tr], S[te], Yb[te], quad=False)[0]

        gains, prs = [], []
        for ck in cks:
            m = build_model(ModelConfig(image_size=args.image_size, latent_trits=trits))
            m.load_weights(ck)
            z = encode_z(m, imgs)
            beh = behavior_from_z(m, z)
            gains.append(gain_ratio(Yb[tr], Yb[te], beh[te]))
            # Participation ratio guards the F10 trap: a COLLAPSED encoder scores
            # HIGHER on coupling metrics, so a Delta claim from a collapsed z is
            # void. Must use train.py:94's definition -- (sum var)^2 / sum(var^2)
            # over per-dimension variance across scenes -- because F10/F12's
            # healthy band (PR ~ 8-84) is stated in those units. The eigenvalue
            # spectrum of the covariance is a different quantity and reads far
            # lower on a correlated latent, which would look like collapse.
            var = z.var(axis=0)
            s1, s2 = float(var.sum()), float((var ** 2).sum())
            prs.append(s1 * s1 / s2 if s2 > 0 else 0.0)

        gm, gs = float(np.mean(gains)), float(np.std(gains))
        delta = gm - shape_gain
        rec = {
            "n_seeds": len(cks), "data": data,
            "shape_oracle_gain": float(shape_gain),
            "model_gain_mean": gm, "model_gain_std": gs,
            "model_gain_per_seed": [float(g) for g in gains],
            "delta": float(delta),
            "delta_per_seed": [float(g - shape_gain) for g in gains],
            "latent_pr_mean": float(np.mean(prs)),
        }
        report["arms"][label] = rec
        print(f"{label:22s} {len(cks):2d} {shape_gain:13.3f} "
              f"{gm:12.3f}+/-{gs:.3f} {delta:+16.3f} {np.mean(prs):7.1f}")

    # ---- preregistered gates ------------------------------------------- #
    arms = report["arms"]
    ctrl = arms.get("ctrl  a=0 g=1")
    verdicts = []
    if ctrl:
        d = abs(ctrl["delta"])
        p1 = "PASS" if d < 0.05 else ("MARGINAL" if d <= 0.10 else "FAIL")
        verdicts.append(f"P1 control |delta|={d:.3f} -> {p1}"
                        + ("" if p1 == "PASS" else
                           "  (>0.10 = leak; NOTHING downstream is interpretable)"))
        report["P1"] = {"abs_delta": d, "verdict": p1}

    dec = arms.get("loud  a=1 g=8 FSQ")
    base = arms.get("base  a=1 g=1")
    loud = arms.get("loud  a=1 g=8")
    # P1 is a HARD gate (docs/ABO_COUPLING.md section 5): if the control does not
    # pass, no downstream arm is interpretable and no verdict may be printed.
    # Printing one anyway is how a gated design quietly becomes an ungated one.
    if ctrl and report["P1"]["verdict"] != "PASS":
        verdicts.append(
            "decisive cell SUPPRESSED -- P1 did not pass, so no downstream arm "
            "is interpretable. Resolve the control before reading any verdict.")
        report["decisive"] = {"suppressed_by": "P1 " + report["P1"]["verdict"]}
    elif dec and base:
        dd = dec["delta"]
        # sign-consistency across seeds is required by the anchor in section 5.
        pos = sum(1 for x in dec["delta_per_seed"] if x > 0.15)
        if dd > 0.15 and pos >= 2:
            v = ("WORLD limit: stronger coupling IS learned -> Option B is worth "
                 "building, with a calibrated target. Stage 2 (ABO) authorized.")
        elif loud and loud["delta"] > 0.15:
            v = ("WORLD limit (continuous only): gain appears without the "
                 "bottleneck -> re-check the FSQ arm before authorizing stage 2.")
        else:
            v = ("ARCHITECTURE limit: every lever at maximum -- full coupling, "
                 "legible channel, forced unity -- and delta is flat. F18-F21 "
                 "re-reads as an architecture result. Stage 2 NOT authorized.")
        verdicts.append(f"decisive cell delta={dd:+.3f} "
                        f"({pos}/{len(dec['delta_per_seed'])} seeds >0.15) -> {v}")
        report["decisive"] = {"delta": dd, "seeds_over_threshold": pos, "verdict": v}

    print()
    for v in verdicts:
        print(f"[F22 gate] {v}")

    os.makedirs(args.root, exist_ok=True)
    out = f"{args.root}/f22_pilot_report.json"
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n[F22] wrote {out}")


if __name__ == "__main__":
    main()
