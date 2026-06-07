"""Generate figures from real in-sandbox runs (no mock data).

Produces docs/figures/sandbox_findings.png with four panels:
  1. Material coupling — appearance ("looks metallic") vs. physics (log density),
     showing the relationship is predictive but noisy.
  2. Training convergence — torch overfit loss curve (gradients flow end to end).
  3. CPU scale ceiling — step time vs. parameter count across model sizes.
  4. Coherence control — the architectural baseline (untrained shared models over
     several seeds), the independent baseline (~0), illustrating why the honest
     signal is learned_coherence = trained - untrained.

Everything here is computed live from the package, so the figure reflects the
actual sandbox behavior. Run: python scripts/make_figures.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pseudomarble.config import ModelConfig  # noqa: E402
from pseudomarble.materials import MaterialSampler  # noqa: E402
from pseudomarble.models import coherence_bench as cb  # noqa: E402
from pseudomarble.models.numpy_net import NumpyModel  # noqa: E402
from pseudomarble.models.torch_net import overfit_smoke  # noqa: E402


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.corrcoef(x, y)[0, 1])


# --------------------------------------------------------------------------- #
def panel_material_coupling(ax):
    s = MaterialSampler(seed=7)
    metallic, logd = [], []
    for _ in range(500):
        m = s.sample().material
        metallic.append(m.visual.metallic)
        logd.append(math.log10(m.physics.density))
    r = pearson(metallic, logd)
    ax.scatter(logd, metallic, s=8, alpha=0.35, color="#3b6ea5", edgecolors="none")
    ax.set_xlabel("log10 density  (physics)")
    ax.set_ylabel("metallic appearance")
    ax.set_title(f"1. Material coupling: predictive but noisy (r={r:.2f})")
    ax.grid(alpha=0.2)


def panel_training_curve(ax):
    import torch
    cfg = replace(ModelConfig(), conv_channels=(16, 32), latent_dim=64,
                  behavior_head_width=64, essence_head_width=32, image_size=32)
    torch.manual_seed(0)
    images = torch.rand(8, 4, 32, 32, 3)
    bt = torch.rand(8, cfg.behavior_dim)
    et = torch.rand(8, cfg.essence_dim)
    hist = overfit_smoke(cfg, images, bt, et, steps=150, lr=1e-3)
    ax.plot(hist, color="#c0504d", lw=1.8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title(f"2. Training converges (torch CPU): {hist[0]:.3f} -> {hist[-1]:.3f}")
    ax.grid(alpha=0.2)


def panel_scale(ax):
    from scripts.bench_torch import run_case
    cases = [
        ("tiny", 8, 4, 32, (8, 16), 32),
        ("small", 8, 8, 64, (32, 64), 128),
        ("medium", 8, 12, 128, (32, 64, 128), 256),
        ("large", 8, 16, 128, (64, 128, 256), 512),
        ("xl", 8, 8, 256, (64, 128, 256), 512),
    ]
    res = [run_case(*c, steps=3) for c in cases]
    params = [r["params_m"] for r in res]
    ms = [r["ms_step"] for r in res]
    ax.plot(params, ms, "o-", color="#4f8a4f", lw=1.6)
    for r in res:
        ax.annotate(f"{r['name']} ({r['img']}px)", (r["params_m"], r["ms_step"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.axhspan(0, 1000, color="#4f8a4f", alpha=0.06)
    ax.set_yscale("log")
    ax.set_xlabel("parameters (millions)")
    ax.set_ylabel("ms / training step (log)")
    ax.set_title("3. CPU scale ceiling (full model incl. render head)")
    ax.grid(alpha=0.2, which="both")


def panel_coherence(ax):
    cfg = replace(ModelConfig(), conv_channels=(8, 16), latent_dim=32,
                  behavior_head_width=32, essence_head_width=16, image_size=16)
    images = np.random.default_rng(0).random((6, 4, 16, 16, 3)).astype("float32")
    # Architectural baseline: untrained shared models across several seeds.
    arch = []
    for seed in range(8):
        m = NumpyModel(cfg, seed=seed)
        zs = [list(z) for z in np.asarray(m.encode(images))]
        fr, fb = cb.numpy_model_decoders(m)
        arch.append(cb.mean_coherence(zs, fr, fb, n_dirs=48, seed=0))
    # Independent baseline (two separate models, disjoint latent).
    r_only, b_only = NumpyModel(cfg, seed=1), NumpyModel(cfg, seed=2)
    zr = [list(z) for z in np.asarray(r_only.encode(images))]
    zb = [list(z) for z in np.asarray(b_only.encode(images))]
    indep = cb.independent_coherence(r_only, b_only, zr, zb, n_dirs=48, seed=0)

    ax.axhspan(min(arch), max(arch), color="#d08770", alpha=0.18,
               label=f"untrained-shared band [{min(arch):.2f}, {max(arch):.2f}]")
    ax.scatter(range(len(arch)), arch, color="#d08770", zorder=3,
               label="untrained shared (per seed)")
    ax.axhline(indep, color="#5e81ac", lw=2, label=f"independent baseline ({indep:.2f})")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_ylim(-0.1, 0.7)
    ax.set_xlabel("untrained seed")
    ax.set_ylabel("coherence")
    ax.set_title("4. Why the honest signal is learned = trained - untrained")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.2)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panel_material_coupling(axes[0, 0])
    panel_training_curve(axes[0, 1])
    panel_scale(axes[1, 0])
    panel_coherence(axes[1, 1])
    fig.suptitle("pseudo-marble — initial sandbox findings (live data, CPU)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sandbox_findings.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
