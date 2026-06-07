"""Scale benchmark for the PyTorch CPU backend (in-sandbox boundary-pushing).

Sweeps model/input sizes, and for each reports parameter count, training
step time, image throughput, peak RSS, and that the loss decreases. This is a
dev/diagnostic tool — it tells us how large a model we can realistically iterate
on in a CPU sandbox before the canonical MLX/Metal runs on the Mac.

    python scripts/bench_torch.py            # default sweep
    python scripts/bench_torch.py --steps 20

Findings (4-core / 15 GB Linux CPU sandbox; FULL model incl. the render head;
loss dropped every case, so gradients flow at all sizes — the limit is
wall-clock, not memory or crashes):

    case     img  views  params   ms/step   img/s   peakRSS
    tiny      32    4     0.07M      15.8    2031     328 MB
    small     64    8     0.19M      49.9    1284     379 MB
    medium   128   12     0.41M     268.5     358     668 MB
    large    128   16     1.01M     847.3     151     913 MB   <- 128px / ~1M target
    xl       256    8     1.02M    2357.2      27    1309 MB

Takeaway: the sandbox comfortably iterates the full model (encoder + behavior +
essence + render head) up to ~1 M params / 128 px / 16 views at sub-second to
~0.85 s/step — fine for correctness + convergence smoke tests. At 256 px step
time jumps to ~2.4 s; past that CPU becomes impractical, which is why MLX/Metal
on the Mac stays the canonical trainer. Memory never bound (peak 1.3 of 15 GB
here). image_size must be render_seed(4) * 2^k for the decoder (32/64/128/256).
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.config import ModelConfig  # noqa: E402

# (name, batch, n_views, image_size, conv_channels, latent_dim)
# image_size must be render_seed(4) * 2^k for the render decoder: 32/64/128/256.
SWEEP = [
    ("tiny", 8, 4, 32, (8, 16), 32),
    ("small", 8, 8, 64, (32, 64), 128),
    ("medium", 8, 12, 128, (32, 64, 128), 256),
    ("large", 8, 16, 128, (64, 128, 256), 512),   # the 128px ~1M target
    ("xl", 8, 8, 256, (64, 128, 256), 512),
]


def _peak_rss_mb() -> float:
    # ru_maxrss is KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_case(name, B, N, img, channels, latent, steps):
    import torch

    from pseudomarble.models.torch_net import TorchModel, loss_fn

    cfg = replace(ModelConfig(), conv_channels=channels, latent_dim=latent,
                  behavior_head_width=256, essence_head_width=128, image_size=img)
    torch.manual_seed(0)
    model = TorchModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    images = torch.rand(B, N, img, img, 3)
    behavior_t = torch.rand(B, cfg.behavior_dim)
    essence_t = torch.rand(B, cfg.essence_dim)
    render_t = images.mean(dim=1)  # mean-view recon target (incl. the render head)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # one warmup step (allocations, MKL planning) excluded from timing
    loss0 = loss_fn(model(images), behavior_t, essence_t, cfg, render_t)
    opt.zero_grad(); loss0.backward(); opt.step()

    t0 = time.perf_counter()
    last = float(loss0.detach())
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(images), behavior_t, essence_t, cfg, render_t)
        loss.backward()
        opt.step()
        last = float(loss.detach())
    dt = time.perf_counter() - t0

    per_step = dt / steps
    imgs_per_s = (B * N * steps) / dt
    print(f"{name:7s} B{B:<2d} N{N:<2d} {img}px {str(channels):16s} z{latent:<4d} "
          f"| {n_params/1e6:6.2f}M params "
          f"| {per_step*1000:7.1f} ms/step "
          f"| {imgs_per_s:7.1f} img/s "
          f"| peakRSS {_peak_rss_mb():6.0f} MB "
          f"| loss {float(loss0.detach()):.3f}->{last:.3f}", flush=True)
    return {
        "name": name, "img": img, "params_m": n_params / 1e6,
        "ms_step": per_step * 1000.0, "img_s": imgs_per_s,
        "peak_rss_mb": _peak_rss_mb(), "loss0": float(loss0.detach()), "last": last,
    }


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=15)
    args = p.parse_args(argv)
    print(f"torch CPU scale sweep ({args.steps} timed steps/case)\n" + "-" * 110, flush=True)
    for case in SWEEP:
        try:
            run_case(*case, steps=args.steps)
        except Exception as exc:  # surface a boundary (OOM, etc.) without aborting
            print(f"{case[0]:7s} | FAILED: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
