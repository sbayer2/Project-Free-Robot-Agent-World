# pseudo-marble

> A small, runnable experiment about giving machines a piece of the **physical
> common sense** humans build over a lifetime: understanding an object well
> enough to predict *what it will do when you act on it* — not just what it looks
> like.

**Personal research.** Not affiliated with World Labs, and not an attempt to copy
their [Marble](https://www.worldlabs.ai/) product. It's a thinking instrument you
can actually run on a MacBook Pro.

---

## The big picture

**Rendering** is going from an *idea* of an object to a *picture* of it. Machines
are very good at this now — type "a ceramic cup" and you get a convincing image.

The hard part is the **inverse**. Go from the description or appearance of an
object to *what it will do when you act on it*: how heavy it is, whether it
slides or grips, how it tips, whether it shatters when dropped, whether it can
hold water. That bundle of "what it will do" is the **essence of the thing** — and
it's *physical*, not linguistic.

Here's the gap. When a machine reads "ceramic cup," it treats those words as text
to continue — the next likely token. When *you* read it, you instantly know it's
hard, brittle, roughly the weight of a small apple, that it'll smash if it hits
tile, that it tips if you push its rim. You don't reason this out; you just know
it. You know because you spent a **childhood physically exploring the world** —
dropping, stacking, spilling, breaking, catching things — as a baby, toddler,
child, teenager, adult. That long, embodied curriculum is the advantage we have
over machines, and it's exactly what these systems are missing.

**Marble and Marble-like systems are early attempts to put that physical
understanding inside a model.** pseudo-marble is a tiny version of the same idea,
small enough to run and study on commodity hardware.

## A simple way to picture it

Imagine a machine that, given a soda can, hands you both a **photo** of the can
*and* its **weight**.

- It could be **two separate machines in one box** — a camera and a scale that
  know nothing about each other.
- Or it could be **one machine that genuinely understands "full aluminum can,"**
  and produces both the photo and the weight from that single understanding.

From the outside they look identical. The whole question behind world models —
and behind this project — is whether you can build (and recognize) the *second*
kind: one understanding from which both appearance and behavior come.

## Where this sits in Li's taxonomy

Fei-Fei Li's [*A Functional Taxonomy of World
Models*](https://www.worldlabs.ai/blog/taxonomy-of-world-models) is the clearest
map of this landscape. It sorts these systems by what they output:

- **Renderer** — outputs **pixels** (how it looks).
- **Simulator** — outputs **state** (how it behaves: shape, mass, friction).
- **Planner** — outputs **actions** (what to do). Li calls the planner *"the
  inverse of the renderer"* — which is exactly the inversion described above.

Her bet is that these are three views of one underlying understanding.
**pseudo-marble focuses on the renderer↔simulator link**: can a single model
understand a thing well enough to produce both how it looks *and* how it behaves —
and is that genuinely *one* understanding, or two outputs glued together?

## What pseudo-marble actually does

1. Takes a simple scene — a **shape** (cube, sphere, …) made of a **material**
   (ceramic, rubber, steel, glass, …).
2. Tries to produce two things from one shared internal representation: **how it
   looks** (rendered views) and **how it behaves** (mass, friction, bounciness).
3. **Measures whether those two really come from one understanding** or are just
   bolted together — by nudging the model's internal state and checking whether
   the look *and* the physics change together (they should, if it's truly one
   thing).

It runs on Apple-silicon (MLX/Metal), with training data anyone can generate
using free tools (MuJoCo, optionally Blender) — no industrial GPU cluster.

## Why it has value

- It's a **runnable, honest, small-scale probe** of the central question behind
  world models: *can a machine hold the physical essence of an object the way a
  person does?*
- It produces a **measurement** for whether a system's looks and physics actually
  share one understanding — something even large systems leave fuzzy.
- It's **reproducible on a laptop**, for everyday coders, not just industry labs.

## Being honest about the limits

- **We can't measure real friction.** We hand the model plausible textbook
  values, so it's learning a simplified world, not the real one. (`materials.py`)
- **A small model might just memorize.** So we test it on shape+material
  combinations it *never saw together* — if it gets those right, it learned the
  rule, not the examples. (`splits.py`)
- **Maybe "glued together" works just as well** as one shared understanding. If
  so, that's a real (if deflating) finding, and we're willing to report it.

The deeper background argument — Li's taxonomy, what Marble is and isn't, and how
this project relates to it — lives in
[`docs/TAXONOMY_NOTES.md`](docs/TAXONOMY_NOTES.md).

## Quick start

```bash
# core logic + tests — no MuJoCo/Blender/MLX needed
python -m pytest                      # or: python tests/test_materials.py

# generate a tiny dataset (primary path: MuJoCo, native on Apple silicon)
pip install -e ".[mujoco]"
python -m pseudomarble.data.generate_mujoco \
    --output data/pseudo_marble --num-scenes 16 --views 16 --resolution 256
```

Full setup (including the optional Blender path and Apple-silicon/MLX) is in
[`docs/HOWTO.md`](docs/HOWTO.md).

## Status

| Part | State |
|---|---|
| Materials: the look↔behavior pairings | ✅ done + tested |
| Train/test split that checks *understanding*, not memorization | ✅ done + tested |
| The "do look and physics move together?" measurement | ✅ done + tested |
| Data generation — MuJoCo (primary), Blender (optional) | ✅ done; runs on your Mac |
| The model itself (encoder + the two outputs, in MLX) | 🔜 next |
| Full experiment: one-understanding vs. glued-together | 🔜 next |
| Using real scanned objects instead of textbook values | 🅿️ planned ([GSO_EXPERIMENT.md](docs/GSO_EXPERIMENT.md)) |

## Repository layout

```
src/pseudomarble/
  materials.py            # the look↔behavior pairings (ceramic, rubber, glass, …)
  splits.py               # held-out shape+material combos (tests understanding)
  config.py               # settings, sized for a MacBook Pro
  data/
    samples.py            # the shared data format every generator writes
    generate_mujoco.py    # primary: MuJoCo renders + physics
    generate_blender.py   # optional: photorealistic renders, same format
    mesh_validate.py      # checks a 3D mesh is solid enough to have a real mass
    collision.py          # keeps an object's real shape (e.g. a cup's cavity)
  models/
    coherence.py          # the "do look and physics move together?" measurement
docs/
  TAXONOMY_NOTES.md       # background: Li's taxonomy, what Marble is and isn't
  ARCHITECTURE.md         # design decisions + honest limitations
  HOWTO.md                # setup & usage
  GSO_EXPERIMENT.md       # planned: using real scanned objects
tests/                    # runnable anywhere, no special hardware
```

## License

[MIT](LICENSE) © 2026 sbayer2. Independent research; not affiliated with or
endorsed by World Labs.
