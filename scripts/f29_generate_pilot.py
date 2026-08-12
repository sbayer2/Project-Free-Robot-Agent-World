#!/usr/bin/env python
"""F29 pilot dataset generator (docs/HELD_OUT_TRANSFER.md sections 3-4).

Generates data/pm_f29_pilot_a{5,20,40}_{ctrl,base,g2,loud}: per aggression
alpha, four matched arms of N_SCENES_PER_ARM held-out scenes whose materials
satisfy the FROZEN section-3 filter:

  * top-alpha tail: density x restitution strictly above the (1 - alpha)
    quantile of that product over the f27 arm's TRAINING scenes, and
  * strictly outside the training convex hull in normalized
    (density, friction, restitution) space (scipy Delaunay).

Every pilot scene is split "test" (held-out sets are evaluation-only; the
train_mean baseline comes from the f27 training data, not from here).

The matched-arms property (F24/F27 apparatus): all four arms of one alpha
share ONE material seed, and the filter reads PHYSICS ONLY, so the
acceptance sequence -- and therefore the physics answer key -- is identical
across arms; only appearance differs via the F22 dials. The generator
asserts this before spending any render time.

Generation settings mirror runs/f27/launch.sh byte-for-byte where they
matter (box, 16 views, 128 px, flat lighting, appearance_noise 0.07);
the pilot uses its OWN seed family (2900 + alpha-percent) so its scenes
are new draws, not replays of the training stream.

Usage:
    .venv/bin/python scripts/f29_generate_pilot.py --dry-run   # filter stats only
    .venv/bin/python scripts/f29_generate_pilot.py             # full generation
    .venv/bin/python scripts/f29_generate_pilot.py --alphas 0.20   # subset
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from f29_pilot import (  # noqa: E402
    N_SCENES_PER_ARM,
    PILOT_ALPHAS,
    PILOT_DATA,
    alpha_tag,
)

from pseudomarble.config import PHYSICS_NORMALIZERS  # noqa: E402
from pseudomarble.materials import MaterialSampler  # noqa: E402

# Arm dials, pinned from runs/f27/launch.sh (the f27 world definitions).
ARM_DIALS: dict = {"ctrl": (0.0, 1.0), "base": (1.0, 1.0),
                   "g2": (1.0, 2.0), "loud": (1.0, 8.0)}
TRAIN_DATA = "data/pm_f27_{arm}"          # where training physics comes from
PILOT_SEED_BASE = 2900                    # seed = 2900 + alpha-percent
APPEARANCE_NOISE = 0.07                   # historical value (launch.sh default)
MAX_CANDIDATES = 200_000                  # loud failure if the tail is too thin


def pilot_seed(alpha: float) -> int:
    return PILOT_SEED_BASE + int(round(alpha * 100))


def normalize_phys(row: tuple) -> tuple:
    d, f, r = row
    return (d / PHYSICS_NORMALIZERS["density"],
            f / PHYSICS_NORMALIZERS["friction"],
            r / PHYSICS_NORMALIZERS["restitution"])


def load_train_physics(arm: str = "ctrl") -> list:
    """Raw (density, friction, restitution) rows from the f27 arm's TRAIN
    split. Physics is matched across arms (F27 apparatus), so any arm works;
    ctrl is the convention and main() asserts the match against loud."""
    rows = []
    for p in sorted(glob.glob(TRAIN_DATA.format(arm=arm) + "/train_*/sample.json")):
        raw = json.load(open(p))["physics"]["raw"]
        rows.append((raw["density"], raw["friction"], raw["restitution"]))
    if not rows:
        raise SystemExit(f"no training scenes under {TRAIN_DATA.format(arm=arm)}")
    return rows


def product_threshold(train_phys: list, alpha: float) -> float:
    """The (1 - alpha) quantile of density x restitution over training
    scenes. Acceptance is STRICTLY above this value (prereg section 3).
    method="linear" is pinned explicitly: the threshold is part of the
    frozen filter, so a numpy default change must not move it."""
    import numpy as np
    products = np.array([d * r for d, _f, r in train_phys])
    return float(np.quantile(products, 1.0 - alpha, method="linear"))


def build_hull(train_phys: list):
    """Delaunay triangulation of the normalized training physics cloud.
    A query point q is outside the hull iff find_simplex(q) < 0."""
    import numpy as np
    from scipy.spatial import Delaunay
    return Delaunay(np.array([normalize_phys(p) for p in train_phys]))


def passes_filter(phys_raw: tuple, threshold: float, hull) -> bool:
    """The frozen section-3 filter: top-alpha product tail AND strictly
    outside the training hull. Returns a plain Python bool."""
    d, _f, r = phys_raw
    if d * r <= threshold:
        return False
    return bool(hull.find_simplex([normalize_phys(phys_raw)])[0] < 0)


def pilot_assignments(arm: str, alpha: float, train_phys: list,
                      n_scenes: int = N_SCENES_PER_ARM) -> tuple:
    """(assignments, stats) for one (arm, alpha) cell.

    Candidates are drawn from a MaterialSampler seeded by alpha ONLY (not
    arm), and the filter reads physics only, so the acceptance sequence is
    identical across arms -- the matched-arms property. Assignments carry
    split "test" and sequential test_NNNNNN ids."""
    ca, cg = ARM_DIALS[arm]
    sampler = MaterialSampler(seed=pilot_seed(alpha),
                              appearance_noise=APPEARANCE_NOISE,
                              coupling_alpha=ca, coupling_gain=cg)
    threshold = product_threshold(train_phys, alpha)
    hull = build_hull(train_phys)
    out, n_drawn = [], 0
    while len(out) < n_scenes:
        if n_drawn >= MAX_CANDIDATES:
            raise RuntimeError(
                f"{arm} alpha={alpha}: only {len(out)}/{n_scenes} accepted "
                f"after {n_drawn} draws -- the alpha-tail outside the hull is "
                f"too thin; the prereg's section-3 band-miss path applies")
        ms = sampler.sample(material_id=f"pilot_{n_drawn:06d}")
        n_drawn += 1
        p = ms.material.physics
        if not passes_filter((p.density, p.friction, p.restitution),
                             threshold, hull):
            continue
        out.append({"scene_id": f"test_{len(out):06d}", "shape": "box",
                    "sample": ms, "split": "test"})
    stats = {"alpha": alpha, "arm": arm, "seed": pilot_seed(alpha),
             "threshold": threshold, "n_drawn": n_drawn,
             "acceptance_rate": n_scenes / n_drawn}
    return out, stats


def physics_rows(assignments: list) -> list:
    return [(a["sample"].material.physics.density,
             a["sample"].material.physics.friction,
             a["sample"].material.physics.restitution) for a in assignments]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default=None,
                    help="comma-separated subset of the frozen alphas")
    ap.add_argument("--out-template", default=PILOT_DATA,
                    help="single source of truth: f29_pilot.PILOT_DATA")
    ap.add_argument("--dry-run", action="store_true",
                    help="filter stats + matched-arms assert only; no MuJoCo")
    ap.add_argument("--render-workers", type=int, default=0)
    ap.add_argument("--sim-workers", type=int, default=0)
    args = ap.parse_args()
    alphas = ([float(a) for a in args.alphas.split(",")] if args.alphas
              else list(PILOT_ALPHAS))
    for a in alphas:
        if a not in PILOT_ALPHAS:
            raise SystemExit(f"alpha {a} is not in the frozen set {PILOT_ALPHAS}")

    train_phys = load_train_physics("ctrl")
    # F27 apparatus re-check: training physics matched across ALL arms.
    for other in ("base", "g2", "loud"):
        if load_train_physics(other) != train_phys:
            raise SystemExit(f"ctrl/{other} training physics differ -- apparatus broken")
    print(f"[f29 pilot gen] {len(train_phys)} training physics rows loaded "
          f"(matched across all 4 arms)")

    for alpha in alphas:
        per_arm, per_arm_stats = {}, {}
        for arm in ARM_DIALS:
            per_arm[arm], per_arm_stats[arm] = pilot_assignments(arm, alpha, train_phys)
            stats = per_arm_stats[arm]
            print(f"  a={alpha:.2f} {arm:5s} accepted {len(per_arm[arm])} of "
                  f"{stats['n_drawn']} draws ({stats['acceptance_rate']:.1%}), "
                  f"threshold {stats['threshold']:.1f}")
        # Matched-arms assert BEFORE any render time is spent.
        ref = physics_rows(per_arm["ctrl"])
        for arm in ("base", "g2", "loud"):
            if physics_rows(per_arm[arm]) != ref:
                raise SystemExit(f"a={alpha}: {arm} physics diverged from ctrl "
                                 "-- matched-arms property violated")
        print(f"  a={alpha:.2f} matched-arms check PASSED (physics identical x4)")
        if args.dry_run:
            continue

        # Full generation: reuse generate_mujoco's phases verbatim.
        from pseudomarble.data import samples
        from pseudomarble.data.generate_mujoco import (
            PhysicsConfig,
            RenderConfig,
            _render_task,
            _sim_task,
            assemble_scene,
        )
        from pseudomarble.data.parallel import (
            default_cpu_workers,
            default_render_workers,
            ordered_parallel_map,
            resolve_workers,
        )
        render_cfg = RenderConfig(resolution=128, num_views=16, lighting="flat")
        physics_cfg = PhysicsConfig()
        for arm, assignments in per_arm.items():
            out_dir = args.out_template.format(tag=alpha_tag(alpha), arm=arm)
            os.makedirs(out_dir, exist_ok=True)
            n = len(assignments)
            rw = resolve_workers(args.render_workers, n, default=default_render_workers())
            sw = resolve_workers(args.sim_workers, n, default=default_cpu_workers())
            print(f"  a={alpha:.2f} {arm}: generating {n} scenes -> {out_dir} "
                  f"(render {rw} / sim {sw})")
            frames_list = ordered_parallel_map(
                _render_task, [(rec, out_dir, render_cfg) for rec in assignments], rw)
            behavior_list = ordered_parallel_map(
                _sim_task, [(rec, physics_cfg, False) for rec in assignments], sw)
            scenes = [assemble_scene(rec["scene_id"], rec["shape"], rec["sample"],
                                     rec["split"], out_dir, render_cfg, physics_cfg,
                                     frames_list[i], behavior_list[i])
                      for i, rec in enumerate(assignments)]
            manifest = samples.build_manifest(
                "mujoco", {"n_train": 0, "n_test": n, "n_scenes": n}, [], scenes)
            manifest["f29_pilot"] = {
                **per_arm_stats[arm],  # alpha, arm, seed, threshold, n_drawn, acceptance_rate
                "filter": "product>threshold AND outside training hull "
                          "(docs/HELD_OUT_TRANSFER.md section 3)",
                "train_source": TRAIN_DATA.format(arm="ctrl"),
                "dials": {"coupling_alpha": ARM_DIALS[arm][0],
                          "coupling_gain": ARM_DIALS[arm][1]},
            }
            with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
                json.dump(manifest, fh, indent=1)
            print(f"  a={alpha:.2f} {arm}: wrote {out_dir}/manifest.json")
    print("[f29 pilot gen] done")


if __name__ == "__main__":
    main()
