"""Per-probe difficulty figure — which behavior outputs are hard to predict.

IMPORTANT: this is a METHODOLOGY demonstration on a synthetic toy-physics
stand-in, because real probe outcomes require MuJoCo (not available in a plain
sandbox). It trains the *real* torch model + behavior head and measures
per-field test error, so it shows the apparatus and the *expected* difficulty
ordering — threshold-like outcomes (topple) hard, smooth ones (settle/bounce)
easy. The real numbers come from the Mac run on MuJoCo data; this is a
placeholder with the right shape.

Run: python scripts/figure_probe_difficulty.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pseudomarble import probes as P  # noqa: E402
from pseudomarble.config import ModelConfig  # noqa: E402
from pseudomarble.materials import MaterialSampler  # noqa: E402

IMPULSE = 1.0
SIZE = 16


def make_image(factors, aspect, rng):
    """A toy appearance tile. The essence factors that drive physics are made
    VISIBLE (as colour channels) so appearance genuinely carries the signal — the
    point of the figure is *which outcomes are hard given visible inputs*, not to
    test whether the inputs are visible at all. Aspect ratio is drawn as a
    rectangle (the cue for toppling)."""
    col = np.array([factors["heaviness"], factors["grip"], factors["hardness"]], "float32")
    img = np.ones((SIZE, SIZE, 3), "float32") * col
    h = int(np.clip(2 + 12 * (aspect / 3.0), 2, SIZE))
    w = int(np.clip(2 + 12 * (1.0 / aspect / 3.0), 2, SIZE))
    y0, x0 = (SIZE - h) // 2, (SIZE - w) // 2
    img[y0:y0 + h, x0:x0 + w, :] *= 0.5  # darker block encodes the shape
    img += rng.normal(0, 0.02, img.shape).astype("float32")
    return np.clip(img, 0, 1)


def toy_outcomes(phys_norm, aspect, rng):
    """Synthetic stand-in for probe outcomes. Threshold-like fields (topple) are
    deliberately harder than smooth ones."""
    f, e = phys_norm["friction"], phys_norm["restitution"]
    tip = aspect * IMPULSE - 2.2 * f                      # tall + low friction -> tips
    toppled = 1.0 if tip + rng.normal(0, 0.05) > 0 else 0.0
    return {
        "toppled": toppled,
        "settle_time": 0.5 + 1.5 * e + rng.normal(0, 0.03),
        "slid_distance": (1.0 - toppled) * 0.4 / (f + 0.25) + rng.normal(0, 0.02),
        "n_bounces": max(0.0, round(4 * e + rng.normal(0, 0.3))),
        "max_height": 0.2 + 1.0 * e + rng.normal(0, 0.02),
        "path_length": (1.0 - toppled) * 0.4 / (f + 0.25) + 0.1 * round(4 * e) + 0.05,
        "final_tilt_deg": 88.0 * toppled + rng.normal(0, 4.0),
    }


def build_dataset(n, seed=0):
    sampler = MaterialSampler(seed=seed)
    rng = np.random.default_rng(seed)
    imgs, targets = [], []
    for _ in range(n):
        ms = sampler.sample()
        aspect = float(rng.uniform(0.5, 3.0))
        imgs.append(make_image(ms.factors, aspect, rng)[None])  # (1,H,W,3) one view
        pn = {"friction": ms.material.physics.friction / 1.5,
              "restitution": ms.material.physics.restitution}
        out = toy_outcomes(pn, aspect, rng)
        # The model predicts the full 21-dim (drop/tilt/push); for this toy we
        # tile the same outcome across the three probe blocks.
        targets.append(P.outcome_vector_from_dict(out, normalize=True) * len(P.PROBE_ORDER))
    return np.asarray(imgs, "float32"), np.asarray(targets, "float32")


def main():
    import torch

    cfg = replace(ModelConfig(), conv_channels=(8, 16), latent_dim=64,
                  behavior_head_width=64, essence_head_width=16, image_size=SIZE)
    from pseudomarble.models.torch_net import TorchModel

    Xtr, Ytr = build_dataset(400, seed=1)
    Xte, Yte = build_dataset(120, seed=99)
    xtr, ytr = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    xte, yte = torch.from_numpy(Xte), torch.from_numpy(Yte)

    torch.manual_seed(0)
    model = TorchModel(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n, bs = xtr.shape[0], 32
    for epoch in range(80):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = model(xtr[idx])["behavior"]
            loss = torch.mean((pred - ytr[idx]) ** 2)
            loss.backward()
            opt.step()

    with torch.no_grad():
        pred = model(xte)["behavior"].numpy()
    # Per-field test MSE on the normalized targets.
    per_field = ((pred - Yte) ** 2).mean(axis=0)
    # Baseline: predicting the train mean (so we see what the model beats).
    base = ((Yte - Ytr.mean(axis=0)) ** 2).mean(axis=0)

    # Show just the seven fields for ONE probe block per field type by averaging
    # across the three probe blocks (same field names repeat per probe).
    fields = P.OUTCOME_FIELDS
    nf = len(fields)
    pf = per_field.reshape(len(P.PROBE_ORDER), nf).mean(axis=0)
    bf = base.reshape(len(P.PROBE_ORDER), nf).mean(axis=0)

    order = np.argsort(pf)
    fields_s = [fields[i] for i in order]
    pf_s, bf_s = pf[order], bf[order]
    threshold_like = {"toppled", "final_tilt_deg"}
    colors = ["#c0504d" if f in threshold_like else "#4f8a4f" for f in fields_s]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(nf)
    ax.barh(y, pf_s, color=colors, label="model test MSE")
    ax.plot(bf_s, y, "k.", ms=9, label="predict-the-mean baseline")
    ax.set_yticks(y)
    ax.set_yticklabels(fields_s)
    ax.set_xlabel("normalized mean squared error on held-out objects (lower = easier)")
    ax.set_title("Per-probe difficulty: which behaviors are hard to predict from appearance")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2, axis="x")
    fig.text(0.01, 0.01,
             "SYNTHETIC toy-physics stand-in (no MuJoCo in sandbox). Red = threshold-like "
             "outcomes (topple / final tilt) — bimodal, so hardest. Green = smooth outcomes.\n"
             "Shape is illustrative of the apparatus; real values come from the MuJoCo "
             "training run on the Mac.", fontsize=7.5, color="#444")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = ROOT / "docs" / "figures" / "probe_difficulty.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    for f, v, b in zip(fields_s, pf_s, bf_s):
        print(f"  {f:16s} mse {v:.4f}  (mean-baseline {b:.4f})")


if __name__ == "__main__":
    main()
