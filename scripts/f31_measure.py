"""F31a measurement: what the latent holds, for torch checkpoints on one world.

Per world: the PIXEL CEILING (F19's pixel_features -> each essence axis,
5-fold ridge R^2) -- what is linearly in the pixels at all. Per checkpoint:
the same 5-fold ridge R^2 from the trained latent z (F19's probe), plus the
run's final held-out behavior gain, latent PR, and escape epoch (first epoch
with held-out gain >= ESCAPE_GAIN) read from its metrics.json.

    python scripts/f31_measure.py --data data/pm_f31_loud \
        --runs runs/parity/loud_s0 runs/parity/loud_s1 --out runs/f31/loud_joint.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from probe_appearance import kfold_r2, pixel_features  # noqa: E402

AXES = ("density", "friction", "restitution")
ESCAPE_GAIN = 2.0


def load_world(data: str, max_views: int = 16):
    from pseudomarble.data.dataset import PseudoMarbleDataset
    ds = PseudoMarbleDataset(data)
    imgs, ess = [], []
    for b in ds.iter_batches(64, shuffle=False, with_images=True, max_views=max_views):
        imgs.append(np.asarray(b["images"], dtype=np.float32))
        ess.extend(b["essence"])
    return np.concatenate(imgs), np.asarray(ess, dtype=float)


def escape_epoch(history: list[dict], threshold: float = ESCAPE_GAIN) -> int | None:
    """First epoch whose held-out gain reaches ``threshold`` (None if never)."""
    for row in history:
        if row.get("heldout_gain", 0.0) >= threshold:
            return int(row["epoch"])
    return None


def encode(run_dir: str, imgs: np.ndarray, chunk: int = 32) -> tuple[np.ndarray, dict]:
    import torch

    from pseudomarble.config import ModelConfig
    from pseudomarble.models.torch_net import TorchModel

    meta = json.load(open(os.path.join(run_dir, "metrics.json")))
    known = ModelConfig.__dataclass_fields__.keys()
    c = {k: v for k, v in meta["config"].items() if k in known}
    for k in ("conv_channels", "trajectory_probes"):
        if k in c:
            c[k] = tuple(c[k])
    model = TorchModel(ModelConfig(**c))
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), weights_only=True))
    model.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, len(imgs), chunk):
            zs.append(model.encode(torch.from_numpy(imgs[i:i + chunk])).numpy())
    return np.concatenate(zs), meta


def r2_dict(X: np.ndarray, E: np.ndarray) -> dict[str, float]:
    return {a: float(v) for a, v in zip(AXES, kfold_r2(X, E), strict=True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    imgs, E = load_world(args.data)
    report = {"data": args.data, "pixel_ceiling_r2": r2_dict(pixel_features(imgs), E),
              "runs": {}}
    print(f"[f31-measure] {args.data}: {len(E)} scenes")
    print("  pixel ceiling R2  " + "  ".join(
        f"{a} {v:+.3f}" for a, v in report["pixel_ceiling_r2"].items()))
    for r in args.runs:
        z, meta = encode(r, imgs)
        final = meta["final"]
        row = {"z_r2": r2_dict(z, E), "heldout_gain": final.get("heldout_gain"),
               "heldout_trajectory_gain": final.get("heldout_trajectory_gain"),
               "latent_pr": final.get("latent_pr"),
               "escape_epoch": escape_epoch(meta["history"])}
        report["runs"][r] = row
        print(f"  {r}: z R2 " + "  ".join(f"{a} {v:+.3f}" for a, v in row["z_r2"].items())
              + f" | gain {row['heldout_gain']:.3f} PR {row['latent_pr']:.1f}"
              f" escape@{row['escape_epoch']}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
