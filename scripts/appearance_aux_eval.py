"""F20 eval: does the appearance-reconstruction aux move behavior off shape-only?

Scans runs/appearance_aux/aw<W>_s<S>/ checkpoints (trained by the sweep in
docs/APPEARANCE_AUX.md), and per appearance_weight reports, on the corner test
split, mean +/- std over seeds of:
  - behavior gain = MSE(predict train-mean) / MSE(model's behavior head)   [the headline]
  - z->appearance R^2 per channel (5-fold CV, all scenes)                   [did the aux reshape z?]
  - render MSE and essence gain                                            [capacity guards]

The question (P2): does behavior gain climb from ~1.41 (appearance_weight 0)
toward the 2.26 appearance ceiling? Design: docs/APPEARANCE_AUX.md.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from oracle_ceiling import gain as gain_ratio  # noqa: E402
from probe_appearance import APPEARANCE_CHANNELS, encode_z, kfold_r2  # noqa: E402

RUN_RE = re.compile(r"aw([0-9p.]+)_s(\d+)")


def main() -> None:
    import mlx.core as mx  # type: ignore

    from pseudomarble.config import ModelConfig
    from pseudomarble.data.dataset import PseudoMarbleDataset
    from pseudomarble.models.mlx_net import build_model

    data = sys.argv[1] if len(sys.argv) > 1 else "data/pm_big"
    run_root = sys.argv[2] if len(sys.argv) > 2 else "runs/appearance_aux"

    ds = PseudoMarbleDataset(data, split=None)
    b = next(ds.iter_batches(batch_size=len(ds), shuffle=False, with_images=True,
                             max_views=16, as_mlx=True))
    imgs = b["images"]
    Yb = np.array(b["behavior"], dtype=np.float32)
    Ye = np.array(b["essence"], dtype=np.float32)
    Ya = np.array(b["appearance"], dtype=np.float32)
    split = np.array([s.record.get("split") for s in ds.scenes])
    tr, te = split == "train", split == "test"
    render_target = np.array(mx.mean(imgs, axis=1).tolist(), dtype=np.float32)

    by_w: dict[str, list[dict]] = defaultdict(list)
    for ck in sorted(glob.glob(os.path.join(run_root, "aw*_s*", "model.safetensors"))):
        m = RUN_RE.search(ck)
        if not m:
            continue
        weight = float(m.group(1).replace("p", "."))
        cfg = ModelConfig(appearance_weight=weight)
        model = build_model(cfg)
        model.load_weights(ck)
        out = model(imgs)
        beh = np.array(out["behavior"].tolist(), dtype=np.float32)
        ess = np.array(out["essence"].tolist(), dtype=np.float32)
        rnd = np.array(out["render"].tolist(), dtype=np.float32)
        z = encode_z(model, imgs)
        rec = {
            "seed": int(m.group(2)),
            "behavior_gain": gain_ratio(Yb[tr], Yb[te], beh[te]),
            "essence_gain": gain_ratio(Ye[tr], Ye[te], ess[te]),
            "render_mse": float(((rnd[te] - render_target[te]) ** 2).mean()),
            "appearance_r2": kfold_r2(z, Ya).tolist(),  # per channel, all scenes
        }
        by_w[m.group(1)].append(rec)
        print(f"[f20] aw={m.group(1)} s{rec['seed']}: behavior_gain={rec['behavior_gain']:.3f}",
              flush=True)

    print(f"\n{'weight':>7s} {'n':>2s} {'behavior_gain':>16s} {'essence_gain':>13s} "
          f"{'render_mse':>11s}   roughness/metallic/transmission z-R^2")
    report = {"data": data, "shape_only": 1.33, "appearance_ceiling": 2.26, "by_weight": {}}
    idx = {c: APPEARANCE_CHANNELS.index(c) for c in ("roughness", "metallic", "transmission")}
    for w in sorted(by_w, key=lambda x: float(x.replace("p", "."))):
        recs = by_w[w]
        bg = np.array([r["behavior_gain"] for r in recs])
        eg = np.array([r["essence_gain"] for r in recs])
        rm = np.array([r["render_mse"] for r in recs])
        ar = np.array([r["appearance_r2"] for r in recs]).mean(0)
        report["by_weight"][w] = {
            "n": len(recs), "behavior_gain_mean": float(bg.mean()),
            "behavior_gain_std": float(bg.std()), "essence_gain_mean": float(eg.mean()),
            "render_mse_mean": float(rm.mean()), "appearance_r2_mean": ar.tolist(),
            "channels": list(APPEARANCE_CHANNELS),
        }
        rmt = " / ".join(f"{ar[idx[c]]:.2f}" for c in ("roughness", "metallic", "transmission"))
        print(f"{w:>7s} {len(recs):2d} {bg.mean():8.3f} +/-{bg.std():5.3f} "
              f"{eg.mean():13.3f} {rm.mean():11.5f}   {rmt}")

    os.makedirs(run_root, exist_ok=True)
    out_path = os.path.join(run_root, "f20_report.json")
    json.dump(report, open(out_path, "w"), indent=2)
    print(f"\n[f20] shape-only baseline 1.33, appearance ceiling 2.26; wrote {out_path}")


if __name__ == "__main__":
    main()
