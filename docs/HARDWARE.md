# Hardware framing — the substrate this project is built on

Durable context for the **target machine** and *why it shapes the code*. Read this
before touching anything performance-shaped (data generation widths, the training
backend, where heavy compute runs). It exists so a future session doesn't have to
re-derive the trade-offs — or conflate two different pieces of "neural silicon".

Pairs with: [`ARCHITECTURE.md`](ARCHITECTURE.md) (design decisions),
[`PREDICTIVE_CODING.md`](PREDICTIVE_CODING.md) (the older-idea-meets-new-hardware
framing). The per-session "is the Mac actually here?" caveat lives in
[`../CLAUDE.md`](../CLAUDE.md).

## The target: MacBook Pro, Apple M5 Pro

| | |
|---|---|
| **CPU** | 18 cores (6 super + 12 performance) |
| **GPU** | 20 cores — **each with a Neural Accelerator** (per-core matmul/AI units, new in the M5 generation) |
| **Unified memory** | 64 GB @ **307 GB/s**, ONE shared pool (CPU + GPU) |
| **Neural Engine (ANE)** | 16-core, separate fixed-function NPU |

Sources: [Apple M5 (Wikipedia)](https://en.wikipedia.org/wiki/Apple_M5),
[Apple debuts M5 Pro and M5 Max](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/),
[Apple unleashes M5](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/).

## Unified memory is two-sided — and both sides are in the code

The upside everyone quotes (e.g. *Predict*, "iswarya writes", May 2026): on a
discrete-GPU PC, moving data CPU↔GPU means a **copy** across PCIe (time + power).
On Apple silicon the CPU and GPU share **one** pool — "the data is just *there*."
For an ML model whose weights, activations and image batches ping between
projections, that zero-copy locality is a real structural win, and a chunk of why
**MLX** can be fast on a laptop.

The **flip side**, which most write-ups skip and which we had to engineer around:
one pool means **one shared memory bus (307 GB/s) and one GPU**. So:

- **Render** (MuJoCo's Metal off-screen renderer) is **GPU-bound** and there is
  exactly **one** GPU. Spawning a worker per CPU core just makes N processes
  **queue** on that single GPU and **contend for the shared bandwidth** — more
  processes, not more throughput.
- **Simulate** (`mj_step`, the drop/tilt/push battery) is **pure CPU**; the GPU is
  idle during it, so it genuinely scales across the performance cores.

Those have **opposite** optimal worker counts, which is exactly why
`data/generate_mujoco.py` runs render and sim as **separate phases at independent
widths** (`--render-workers` small, `--sim-workers` wide), and why the auto default
in `data/parallel.py` is **not** `os.cpu_count()`. See `default_render_workers`
(capped small) vs `default_cpu_workers` (most cores). On a discrete-GPU box the
naive "one process per core" would have been fine; on a unified-memory SoC it
oversubscribes the very resource the GPU needs. **The architecture celebrates
unified memory; the scheduler respects its cost.**

## Two different "neural silicon" — do NOT conflate them

Popular coverage collapses everything AI on the chip into "the Neural Engine."
For *this* project the distinction is load-bearing:

| | **Neural Engine (ANE)** | **GPU Neural Accelerators** |
|---|---|---|
| What | 16-core fixed-function NPU | matmul/AI units **inside each of the 20 GPU cores** (new in M5) |
| Reached via | **CoreML** (and a constrained op set) | **MLX / Metal** (general GPU compute) |
| Our use | **not targeted** | **this is where our model runs** |

Our encoder + behavior/essence/render heads, the training loop (`models/train.py`),
and the coherence measurement run as **MLX on the GPU** — and *that's* what taps
the per-core Neural Accelerators. MLX does **not**, in general, dispatch to the
ANE; CoreML is the ANE's path and we don't use it. So when tuning the Mac run, the
relevant new M5 hardware is the **GPU Neural Accelerators**, not the headline
16-core ANE. (Same industry trend the article points at — everyone spending
transistors on neural units — but for our workload it's the GPU side that does the
lifting.)

## Why MLX, and why we don't retarget

The "vertical stack" argument (one team owns chip → memory architecture → OS →
frameworks) is, concretely, the case for **MLX**: it exists *because* the same
vendor controls Metal and the unified-memory scheduler. That's the lever the
project leans on, and the reason CLAUDE.md says **do not retarget away from MLX**.

Practical consequence already baked into the repo (three mirrored backends from one
`ModelConfig`):

- **`models/mlx_net.py`** — canonical trainer, MLX/Metal, **Mac only** (this is the
  real substrate above).
- **`models/numpy_net.py`** — forward-only, validates shapes in any Linux session
  (the pip `mlx` wheel is non-functional on Linux x86 — no `libmlx.so`).
- **`models/torch_net.py`** — PyTorch **CPU** stand-in, trainable in-sandbox to
  confirm the loop converges before spending Mac time. NOT canonical.

So: author MLX/Mac code with **pure-Python cores that test anywhere**, and **defer
Mac-only execution** (MLX training, Metal/MuJoCo rendering at speed) to when the
laptop is in hand.

## Quick rules of thumb for a future session

- **Heavy model compute → MLX on the GPU.** Don't reach for the ANE; don't add a
  CoreML path expecting a free lunch.
- **Data-gen parallelism is phase-aware.** Keep the render pool small (one GPU +
  shared bus); let the sim pool go wide. Don't "simplify" it back to a single
  `--workers = cpu_count()` — that regression is the whole point of PR #16.
- **Memory is rarely the bound in-sandbox; the GPU/bus is the bound on the Mac.**
  In the Linux sandbox the limit is CPU step-time (see `scripts/bench_torch.py`);
  on the Mac the unified-memory bandwidth and the single GPU set the ceiling.
- **The per-phase optima are still unmeasured.** Defaults are deliberately
  conservative; a short sweep on the Mac (workers × resolution × views) pins them.

*Personal research; not affiliated with World Labs. Hardware figures per Apple's
published M5 Pro specs (linked above).*
