"""F21 Arm 1 eval: does lowering appearance_noise close the F20 gap?

For each noise level (its own data/pm_n<NN> dataset + runs/render_fidelity/n<NN>_s*
checkpoints, trained at appearance_weight=0.3), reports on the corner split:
  - oracle ceiling  = shape+appearance ridge/knn gain (P1 manipulation check; rises as noise falls)
  - untrained-z reachability R^2 of roughness/metallic/transmission (P2: should be noise-invariant)
  - fresh-head-on-z ceiling = ridge/knn z->behavior gain (P3 crux + GATE)
  - trained behavior gain (the model's own head)

Design + registered predictions + the gate rule: docs/RENDER_FIDELITY.md.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from oracle_ceiling import best_oracle, knn_fit_predict, ridge_fit_predict  # noqa: E402
from oracle_ceiling import gain as gain_ratio  # noqa: E402
from probe_appearance import APPEARANCE_CHANNELS, encode_z, kfold_r2  # noqa: E402

NOISE = [("0.07", "n07"), ("0.03", "n03"), ("0.0", "n00")]
MAT = ("roughness", "metallic", "transmission")


def load_arrays(data: str):
    from pseudomarble.data.dataset import PseudoMarbleDataset
    ds = PseudoMarbleDataset(data, split=None)
    b = next(ds.iter_batches(batch_size=len(ds), shuffle=False, with_images=True,
                             max_views=16, as_mlx=True))
    split = np.array([s.record.get("split") for s in ds.scenes])
    return (b["images"], np.array(b["behavior"], np.float32),
            np.array(b["appearance"], np.float32), split == "train", split == "test", ds)


def shape_onehot(ds) -> np.ndarray:
    names = sorted({s.shape for s in ds.scenes})
    oh = np.zeros((len(ds.scenes), len(names)), np.float32)
    for i, s in enumerate(ds.scenes):
        oh[i, names.index(s.shape)] = 1.0
    return oh


def main() -> None:
    from pseudomarble.config import ModelConfig
    from pseudomarble.models.mlx_net import build_model

    root = sys.argv[1] if len(sys.argv) > 1 else "runs/render_fidelity"
    mi = {c: APPEARANCE_CHANNELS.index(c) for c in MAT}
    report = {"by_noise": {}}
    print(f"{'noise':>6s} {'oracle_ceil':>11s} {'reach r/m/t':>18s} "
          f"{'freshhead_ceil':>14s} {'trained_gain':>12s}")

    for noise, tag in NOISE:
        data = f"data/pm_{tag}"
        cks = sorted(glob.glob(f"{root}/{tag}_s*/model.safetensors"))
        if not os.path.exists(f"{data}/manifest.json") or not cks:
            print(f"{noise:>6s}  (missing data or checkpoints; skipped)")
            continue
        imgs, Yb, Ya, tr, te, ds = load_arrays(data)
        S = shape_onehot(ds)

        # P1: oracle ceiling shape+appearance (uses the CLEAN recorded params).
        SA = np.concatenate([S, Ya], 1)
        oracle_ceil = best_oracle(SA[tr], Yb[tr], SA[te], Yb[te])[0]

        fresh, tgain, reach, aret = [], [], [], []
        for ck in cks:
            m = build_model(ModelConfig(appearance_weight=0.3))
            m.load_weights(ck)
            z = encode_z(m, imgs)
            beh = np.array(m(imgs)["behavior"].tolist(), np.float32)
            tgain.append(gain_ratio(Yb[tr], Yb[te], beh[te]))
            # fresh head ceiling: best of ridge/knn z->behavior on corner split
            gr = gain_ratio(Yb[tr], Yb[te], ridge_fit_predict(z[tr], Yb[tr], z[te]))
            gk = gain_ratio(Yb[tr], Yb[te], knn_fit_predict(z[tr], Yb[tr], z[te]))
            fresh.append(max(gr, gk))
            aret.append(kfold_r2(z, Ya))
        # P2: untrained-z reachability of the material channels (arch-only ceiling)
        for _ in range(3):
            um = build_model(ModelConfig(appearance_weight=0.3))
            reach.append(kfold_r2(encode_z(um, imgs), Ya))
        reach_m = np.stack(reach).mean(0)
        aret_m = np.stack(aret).mean(0)

        rec = {
            "n": len(cks), "oracle_ceiling": float(oracle_ceil),
            "fresh_head_ceiling_mean": float(np.mean(fresh)),
            "fresh_head_ceiling_std": float(np.std(fresh)),
            "trained_gain_mean": float(np.mean(tgain)),
            "reach_material_r2": {c: float(reach_m[mi[c]]) for c in MAT},
            "retained_material_r2": {c: float(aret_m[mi[c]]) for c in MAT},
        }
        report["by_noise"][noise] = rec
        rmt = "/".join(f"{reach_m[mi[c]]:.2f}" for c in MAT)
        print(f"{noise:>6s} {oracle_ceil:11.3f} {rmt:>18s} "
              f"{np.mean(fresh):8.3f}+/-{np.std(fresh):.3f} {np.mean(tgain):12.3f}")

    # P3 gate on the noise=0.0 fresh-head ceiling.
    n0 = report["by_noise"].get("0.0")
    if n0:
        c = n0["fresh_head_ceiling_mean"]
        verdict = ("Link 1 (decoupling) DOMINATED -> cheap; STOP the ladder" if c >= 1.90
                   else "Link 2 (legibility) BINDS -> run Arm 2 (256px+lighting)" if c <= 1.70
                   else "BOTH bind (1.70-1.90 band) -> run Arm 2 to apportion")
        report["gate"] = {"noise0_fresh_head_ceiling": c, "verdict": verdict}
        print(f"\n[F21 gate] noise=0 fresh-head ceiling {c:.3f} (F20 was 1.57): {verdict}")

    os.makedirs(root, exist_ok=True)
    json.dump(report, open(f"{root}/f21_arm1_report.json", "w"), indent=2)
    print(f"[F21] wrote {root}/f21_arm1_report.json")


if __name__ == "__main__":
    main()
