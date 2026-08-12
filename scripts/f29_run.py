#!/usr/bin/env python
"""F29 main run driver: held-out-transfer coherence
(docs/HELD_OUT_TRANSFER.md sections 2, 5-7; Amendment 1 pins alpha* = 0.40
and main seed 2941).

Two modes:

    --generate  Generate data/pm_f29_holdout_{ctrl,base,g2,loud}: 128
                scenes per arm at MAIN_ALPHA through the SAME frozen filter
                and matched-arms machinery as the pilot datasets
                (f29_generate_pilot.pilot_assignments with the Amendment-1
                seed), asserted before any render time. MuJoCo only.

    (default)   The main measurement: per arm, held-out gains for the 3
                joint seeds and 3 disjoint behav pairs on the holdout
                (baseline = that arm's f27 TRAIN-split mean, section 2's
                formula), every section-7 gate, and the report in the
                exact schema the FROZEN scripts/f29_verdicts.py reads --
                with meta.amendment_1 embedding the pilot summary from
                runs/f29/pilot_report.json (the frozen pilot gate). The
                driver asserts pilot alpha_star == MAIN_ALPHA before
                measuring. NO verdicts here.

MAIN_SEED is 2941 and must never equal the pilot's alpha-0.40 seed
(2940): reusing it would duplicate pm_f29_pilot_a40_* scene-for-scene
and import Amendment 1's disclosed pilot leak into the verdict-bearing
measurement. tests/test_f29_run.py pins this.

Usage (Mac only for the default mode):
    .venv/bin/python scripts/f29_run.py --generate   # holdout datasets
    .venv/bin/python scripts/f29_run.py --dry-run    # list work, check inputs
    .venv/bin/python scripts/f29_run.py              # measure + write report
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from f29_pilot import (  # noqa: E402
    DISJOINT_BEHAV,
    JOINT_CHECKPOINTS,
    r_from_records,
)

# ---- Amendment-1 constants (docs/HELD_OUT_TRANSFER.md Amendment 1) ---------- #
MAIN_ALPHA = 0.40                       # alpha*, pinned by the pilot's frozen rule
MAIN_SEED = 2941                        # fresh seed; 2940 would replay the pilot a40 sets
MAIN_DATA = "data/pm_f29_holdout_{arm}"  # per-arm suffix per Amendment 1 section 6
ARMS = ("ctrl", "base", "g2", "loud")


def arm_record(r: float, gates: dict, joint_gains: list[float],
               disjoint_gains: list[float]) -> dict:
    """The per-arm schema scripts/f29_verdicts.py reads (pinned by test)."""
    return {"r": float(r), "gates": gates,
            "joint": {"per_seed_gain": [float(g) for g in joint_gains]},
            "disjoint": {"per_pair_gain": [float(g) for g in disjoint_gains]}}


def generate_main() -> None:
    """Holdout datasets at MAIN_ALPHA / MAIN_SEED through the pilot's frozen
    machinery. Mirrors f29_generate_pilot.main()'s generation block."""
    from f29_generate_pilot import (
        ARM_DIALS,
        load_train_physics,
        physics_rows,
        pilot_assignments,
        product_threshold,
    )

    train_phys = load_train_physics("ctrl")
    for other in ("base", "g2", "loud"):
        if load_train_physics(other) != train_phys:
            raise SystemExit(f"ctrl/{other} training physics differ -- apparatus broken")

    per_arm, per_stats = {}, {}
    for arm in ARM_DIALS:
        per_arm[arm], per_stats[arm] = pilot_assignments(
            arm, MAIN_ALPHA, train_phys, seed=MAIN_SEED)
        s = per_stats[arm]
        print(f"  main {arm:5s} accepted {len(per_arm[arm])} of {s['n_drawn']} "
              f"draws ({s['acceptance_rate']:.1%}), seed {s['seed']}")
    ref = physics_rows(per_arm["ctrl"])
    for arm in ("base", "g2", "loud"):
        if physics_rows(per_arm[arm]) != ref:
            raise SystemExit(f"main: {arm} physics diverged -- matched-arms violated")
    print("  main matched-arms check PASSED (physics identical x4)")

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
        out_dir = MAIN_DATA.format(arm=arm)
        os.makedirs(out_dir, exist_ok=True)
        n = len(assignments)
        rw = resolve_workers(0, n, default=default_render_workers())
        sw = resolve_workers(0, n, default=default_cpu_workers())
        print(f"  main {arm}: generating {n} scenes -> {out_dir} "
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
        manifest["f29_main"] = {
            **per_stats[arm],
            "filter": "product>threshold AND outside training hull "
                      "(docs/HELD_OUT_TRANSFER.md section 3; Amendment 1 pins "
                      "alpha*=0.40, seed 2941)",
            "threshold_check": product_threshold(train_phys, MAIN_ALPHA),
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
        print(f"  main {arm}: wrote {out_dir}/manifest.json")
    print("[f29 main gen] done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true",
                    help="generate the main holdout datasets (MuJoCo only)")
    ap.add_argument("--pilot-report", default="runs/f29/pilot_report.json")
    ap.add_argument("--out", default="runs/f29/f29_report.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.generate:
        generate_main()
        return

    joint_cks = {arm: {f"s{k}": JOINT_CHECKPOINTS.format(arm=arm, seed=k)
                       for k in range(3)} for arm in ARMS}
    disjoint_cks = {arm: {f"pair{k}": DISJOINT_BEHAV.format(arm=arm, k=k)
                          for k in range(3)} for arm in ARMS}
    missing = [p for arm in ARMS
               for p in (list(joint_cks[arm].values())
                         + list(disjoint_cks[arm].values()))
               if not os.path.exists(p)]
    missing += [MAIN_DATA.format(arm=arm) for arm in ARMS
                if not os.path.exists(os.path.join(MAIN_DATA.format(arm=arm),
                                                   "manifest.json"))]
    if not os.path.exists(args.pilot_report):
        missing.append(args.pilot_report + " (run scripts/f29_pilot.py first)")
    if missing:
        listing = "\n  ".join(missing)
        raise SystemExit(f"missing inputs:\n  {listing}")
    if args.dry_run:
        print(f"would measure 6 models x {len(ARMS)} arms on the main holdout "
              f"+ disjoint train gains x {len(ARMS)} arms; report -> {args.out}")
        return

    import mlx.core as mx
    import numpy as np
    from f29_generate_pilot import build_hull, load_train_physics, normalize_phys
    from f29_pilot import _measure_models_on
    from render_fidelity_eval import load_arrays

    with open(args.pilot_report) as f:
        pilot = json.load(f)["summary"]
    if pilot["alpha_star"] != MAIN_ALPHA:
        raise SystemExit(f"pilot alpha_star {pilot['alpha_star']} != "
                         f"MAIN_ALPHA {MAIN_ALPHA} -- Amendment 1 mismatch")
    with open("runs/f27b/f27_report.json") as f:
        f27 = json.load(f)

    hull = build_hull(load_train_physics("ctrl"))
    report = {"meta": {"amendment_1": pilot, "main_alpha": MAIN_ALPHA,
                       "main_seed": MAIN_SEED},
              "arms": {}}

    for arm in ARMS:
        r_train = float(f27["apparatus"]["r"][arm])
        # Phase A: this arm's training world -- disjoint train gains (gate)
        # and the train-split behaviors that anchor the held-out baseline.
        imgs, Yb, _Ya, tr, te, _ds = load_arrays(f"data/pm_f27_{arm}")
        dis_train = _measure_models_on(imgs, Yb[tr], Yb[te],
                                       disjoint_cks[arm], eval_idx=te)
        Yb_train = Yb[tr]
        del imgs
        mx.clear_cache()

        # Phase B: the holdout -- both students, gains + PRs.
        imgs_H, Yb_H, _YaH, _trH, _teH, ds_H = load_arrays(MAIN_DATA.format(arm=arm))
        records = [s.record for s in ds_H.scenes]
        joint_held = _measure_models_on(imgs_H, Yb_train, Yb_H, joint_cks[arm])
        dis_held = _measure_models_on(imgs_H, Yb_train, Yb_H, disjoint_cks[arm])
        del imgs_H
        mx.clear_cache()

        phys = [(s["physics"]["raw"]["density"], s["physics"]["raw"]["friction"],
                 s["physics"]["raw"]["restitution"]) for s in records]
        ext = float(np.mean([hull.find_simplex([normalize_phys(p)])[0] < 0
                             for p in phys]))
        gates = {
            "r_held_out": r_from_records(records),
            "r_train": r_train,
            "extrapolation_frac_outside": ext,
            "disjoint_train_gain_min": min(v["gain"] for v in dis_train.values()),
            "joint_pr_min": min(v["pr"] for v in joint_held.values()),
            "disjoint_pr_min": min(v["pr"] for v in dis_held.values()),
        }
        jg = [joint_held[f"s{k}"]["gain"] for k in range(3)]
        dg = [dis_held[f"pair{k}"]["gain"] for k in range(3)]
        report["arms"][arm] = arm_record(r_train, gates, jg, dg)
        print(f"[{arm}] joint held " + "/".join(f"{g:.2f}" for g in jg)
              + "  disjoint held " + "/".join(f"{g:.2f}" for g in dg)
              + f"  adv {np.mean(jg) - np.mean(dg):+.3f}"
              + f"  r_held {gates['r_held_out']:+.3f}  outside {ext:.0%}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {args.out}\nnext: .venv/bin/python scripts/f29_verdicts.py "
          f"--report {args.out}")


if __name__ == "__main__":
    main()
