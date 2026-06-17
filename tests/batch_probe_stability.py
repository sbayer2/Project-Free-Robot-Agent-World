"""Probe-outcome stability study — de-risk the behavior labels before training.

`docs/BEHAVIOR_TASK.md` flags one honest risk: **chaos near tipping points**. The
PUSH probe records `toppled = final_tilt_deg > 50` (a hard threshold). An object
balanced near its tipping point can have its label flipped by an arbitrarily small
difference, so those labels would be noise — and any `learned_coherence` measured
against noisy targets is contaminated. Before spending Mac training time we want
to know *how bad* this is and *how much of the actual dataset* it touches.

This is a pure-simulation harness: it runs the drop/tilt/push battery directly
(`mj_step` on a single rigid body, milliseconds each) and never renders, so a few
thousand probe runs finish in seconds and never touch the GPU. It is therefore a
batch experiment, NOT a unit test — run it, read the numbers, don't collect it
with pytest. Results (raw arrays + summary) are written to `runs/stability/`
(gitignored) for later plotting.

Four layers, in order, telling one story:

  1. DETERMINISM  — same (shape, material) twice ⇒ identical outcomes. If this
     fails, the sim path has hidden nondeterminism and nothing below is trustworthy.
  2. BOUNDARY SWEEP — sweep heaviness→density per shape; locate the 50° topple
     crossover and measure how *sharp* it is (the discontinuity's width).
  3. SENSITIVITY  — jitter the push action by a tiny ε at a near-boundary material
     vs. a far one; the `toppled` flip-rate should spike ONLY near the boundary.
  4. POPULATION   — sample the real MaterialSampler distribution; report per-shape
     topple base rate and the fraction of scenes inside the ambiguous deadband
     around 50°. That fraction bounds the label contamination.

Run::

    python tests/batch_probe_stability.py            # defaults, ~seconds
    python tests/batch_probe_stability.py --population 400 --jitter-reps 48
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from typing import Dict, List, Optional

from pseudomarble import materials as M
from pseudomarble import probes as P
from pseudomarble.config import PHYSICS_NORMALIZERS, PhysicsConfig
from pseudomarble.data import generate_mujoco as G

# Shapes that span squat→tall, so PUSH yields both sliding and toppling.
SHAPES = ["box", "sphere", "cylinder", "capsule", "ellipsoid"]
# The canonical slide-vs-topple boundary shape (docs/BEHAVIOR_TASK.md's own example:
# "a tall cylinder topples where a squat box slides"). It has a stable upright pose
# AND a genuine interior crossover, so it is where near-boundary chaos can actually
# occur — unlike the capsule (rolls onto its side ⇒ always 'toppled', but stably) or
# the sphere (no upright pose ⇒ 'toppled' just measures how far it rolled).
FOCUS = "cylinder"
TOPPLE_DEG = P.TOPPLE_ANGLE_DEG   # the hard threshold whose neighbourhood is risky


def classify_label(topple_fraction: float, crossover_h: Optional[float]) -> str:
    """How well-posed is the binary `toppled` label for this shape's sweep?
      * never    — never crosses (squat: always slides)         → stable, always False
      * always   — always over threshold (rolls onto its side)  → stable, always True
      * boundary — interior crossover                           → the chaos-prone case
    'never'/'always' are *stable* even though degenerate; 'boundary' is where a tiny
    difference can flip the label."""
    if topple_fraction <= 0.02:
        return "never"
    if topple_fraction >= 0.98:
        return "always"
    return "boundary" if crossover_h is not None else "mixed_no_crossover"

# A neutral appearance — irrelevant to the sim, but Material requires a VisualProps.
_NEUTRAL_VISUAL = M.VisualProps(base_color=(0.6, 0.6, 0.6, 1.0), roughness=0.5, metallic=0.0)
_SAMPLER_FOR_PHYSICS = M.MaterialSampler(seed=0)   # only its (deterministic) factor→physics map


def physics_material(heaviness: float, grip: float, hardness: float) -> M.Material:
    """A Material whose physics follow the SAME factor→physics map the generator
    uses (so a heaviness sweep traces the real generative axis), with placeholder
    appearance (the sim ignores it)."""
    phys = _SAMPLER_FOR_PHYSICS._physics_from_factors(heaviness, grip, hardness)
    return M.Material(name="probe_study", visual=_NEUTRAL_VISUAL, physics=phys)


def push_final_tilt(shape: str, material: M.Material, cfg: PhysicsConfig,
                    spec: Optional[P.PushSpec] = None) -> float:
    """Run one PUSH and return final_tilt_deg (the quantity the topple label thresholds)."""
    traj = G.run_push(shape, material, spec or P.PushSpec(), cfg)
    return P.summarize(traj).final_tilt_deg


# --------------------------------------------------------------------------- #
# Layer 1 — determinism control.
# --------------------------------------------------------------------------- #
def layer_determinism(cfg: PhysicsConfig) -> Dict:
    mat = physics_material(0.5, 0.5, 0.4)
    runs = [G.run_probes(FOCUS, mat, cfg) for _ in range(3)]
    base = runs[0]
    max_abs_diff = 0.0
    for other in runs[1:]:
        for rb, ro in zip(base, other):
            for k in P.OUTCOME_FIELDS:
                max_abs_diff = max(max_abs_diff, abs(float(rb["outcome"][k]) - float(ro["outcome"][k])))
    return {"reps": len(runs), "shape": FOCUS, "max_abs_field_diff": max_abs_diff,
            "deterministic": max_abs_diff == 0.0}


# --------------------------------------------------------------------------- #
# Layer 2 — topple-boundary sweep (how sharp is the discontinuity?).
# --------------------------------------------------------------------------- #
def layer_boundary_sweep(cfg: PhysicsConfig, n_points: int, grip: float,
                         hardness: float) -> Dict:
    hs = [i / (n_points - 1) for i in range(n_points)]
    per_shape: Dict[str, Dict] = {}
    for shape in SHAPES:
        tilts, densities, toppled = [], [], []
        for h in hs:
            mat = physics_material(h, grip, hardness)
            tilt = push_final_tilt(shape, mat, cfg)
            tilts.append(tilt)
            densities.append(mat.physics.density)
            toppled.append(tilt > TOPPLE_DEG)
        # Crossover: where the topple label flips along the sweep (linear-interp in h).
        crossover_h = None
        max_slope = 0.0   # max |Δtilt / Δh| between adjacent points — sharpness proxy
        for i in range(1, n_points):
            slope = abs(tilts[i] - tilts[i - 1]) / max(1e-9, hs[i] - hs[i - 1])
            max_slope = max(max_slope, slope)
            if toppled[i] != toppled[i - 1] and crossover_h is None:
                # linear interpolate the h at which tilt == TOPPLE_DEG
                t0, t1 = tilts[i - 1], tilts[i]
                frac = (TOPPLE_DEG - t0) / (t1 - t0) if t1 != t0 else 0.5
                crossover_h = hs[i - 1] + frac * (hs[i] - hs[i - 1])
        # Ambiguous band: sweep points whose tilt sits within ±15° of the threshold.
        n_ambiguous = sum(1 for t in tilts if abs(t - TOPPLE_DEG) <= 15.0)
        topple_fraction = sum(toppled) / n_points
        per_shape[shape] = {
            "heaviness": hs,
            "density": densities,
            "final_tilt_deg": tilts,
            "toppled": toppled,
            "topple_fraction": topple_fraction,
            "crossover_heaviness": crossover_h,
            "max_slope_deg_per_h": max_slope,
            "n_ambiguous_pm15deg": n_ambiguous,
            "ambiguous_fraction": n_ambiguous / n_points,
            "label_class": classify_label(topple_fraction, crossover_h),
        }
    return {"grip": grip, "hardness": hardness, "n_points": n_points, "per_shape": per_shape}


# --------------------------------------------------------------------------- #
# Layer 3 — action-sensitivity: tiny push jitter near vs. far from the boundary.
# --------------------------------------------------------------------------- #
def _jittered_push_run(shape: str, mat: M.Material, cfg: PhysicsConfig, reps: int,
                       eps_impulse_rel: float, eps_az_deg: float, seed: int) -> Dict:
    rng = random.Random(seed)
    base = P.PushSpec()
    tilts, toppled = [], []
    for _ in range(reps):
        spec = P.PushSpec(
            impulse=base.impulse * (1.0 + rng.gauss(0.0, eps_impulse_rel)),
            height_frac=base.height_frac,
            azimuth_deg=base.azimuth_deg + rng.gauss(0.0, eps_az_deg),
        )
        tilt = push_final_tilt(shape, mat, cfg, spec)
        tilts.append(tilt)
        toppled.append(tilt > TOPPLE_DEG)
    p_topple = sum(toppled) / reps
    return {
        "reps": reps,
        "tilt_mean": statistics.mean(tilts),
        "tilt_std": statistics.pstdev(tilts) if reps > 1 else 0.0,
        "p_topple": p_topple,
        # flip-rate = how often the label disagrees with its own majority (0 stable, 0.5 max chaos)
        "flip_rate": min(p_topple, 1.0 - p_topple),
    }


def layer_sensitivity(cfg: PhysicsConfig, sweep: Dict, reps: int,
                      eps_impulse_rel: float, eps_az_deg: float) -> Dict:
    """Two complementary views of action-jitter sensitivity:

    (a) per_shape — jitter at each shape's own crossover heaviness (its most
        sensitive point; mid if it has none). A high flip_rate here means the
        binary label is chaotic for that shape near its boundary.
    (b) profile  — on the FOCUS shape (cylinder), jitter near the boundary vs.
        far on either side, to show the chaos is *localised* to the crossover.
    """
    grip, hardness = sweep["grip"], sweep["hardness"]

    per_shape: Dict[str, Dict] = {}
    for shape in SHAPES:
        d = sweep["per_shape"][shape]
        cross = d["crossover_heaviness"]
        h = cross if cross is not None else 0.5
        mat = physics_material(h, grip, hardness)
        res = _jittered_push_run(shape, mat, cfg, reps, eps_impulse_rel, eps_az_deg,
                                 seed=2000 + int(h * 1000))
        res.update({"heaviness": h, "at_crossover": cross is not None,
                    "label_class": d["label_class"]})
        per_shape[shape] = res

    cross = sweep["per_shape"][FOCUS]["crossover_heaviness"]
    if cross is None:
        cross = 0.5
    profile_points = {
        "near_boundary": min(1.0, max(0.0, cross)),
        "far_light": max(0.0, cross - 0.30),   # lighter than crossover → tends to topple
        "far_heavy": min(1.0, cross + 0.30),   # heavier than crossover → tends to resist
    }
    profile: Dict[str, Dict] = {}
    for label, h in profile_points.items():
        mat = physics_material(h, grip, hardness)
        res = _jittered_push_run(FOCUS, mat, cfg, reps, eps_impulse_rel, eps_az_deg,
                                 seed=1000 + int(h * 1000))
        res.update({"heaviness": h, "density": mat.physics.density})
        profile[label] = res
    return {"focus_shape": FOCUS, "crossover_heaviness": cross,
            "eps_impulse_rel": eps_impulse_rel, "eps_az_deg": eps_az_deg,
            "per_shape": per_shape, "profile": profile}


# --------------------------------------------------------------------------- #
# Layer 4 — population: how much of the ACTUAL sampled dataset is near-boundary?
# --------------------------------------------------------------------------- #
DEADBANDS = (5.0, 10.0, 15.0)   # report label-ambiguity at several deadband widths


def layer_population(cfg: PhysicsConfig, n: int, seed: int) -> Dict:
    # Mirror assign_scenes: one rng drives both shape choice and material sampling.
    sampler = M.MaterialSampler(seed=seed)
    rng = random.Random(seed)
    per_shape_topple: Dict[str, List[bool]] = {s: [] for s in SHAPES}
    tilts: List[float] = []
    for i in range(n):
        shape = rng.choice(SHAPES)
        ms = sampler.sample(material_id=f"scene_{i:06d}")
        tilt = push_final_tilt(shape, ms.material, cfg)
        tilts.append(tilt)
        per_shape_topple[shape].append(tilt > TOPPLE_DEG)
    per_shape = {
        s: {"n": len(v), "topple_rate": (sum(v) / len(v)) if v else None}
        for s, v in per_shape_topple.items()
    }
    deadband_fraction = {
        f"pm{int(db)}": sum(1 for t in tilts if abs(t - TOPPLE_DEG) <= db) / n
        for db in DEADBANDS
    }
    return {
        "n": n, "seed": seed,
        "overall_topple_rate": sum(1 for t in tilts if t > TOPPLE_DEG) / n,
        "deadband_fraction": deadband_fraction,
        "tilt_hist_edges": list(range(0, 181, 15)),
        "tilt_hist": _hist(tilts, list(range(0, 181, 15))),
        "per_shape": per_shape,
    }


def multi_population(cfg: PhysicsConfig, n: int, seeds: List[int]) -> Dict:
    """Run the population study over several seeds and aggregate (mean ± std), so
    the contamination bound is a multi-seed statistic, not a single draw."""
    runs = [layer_population(cfg, n, s) for s in seeds]

    def agg(getter) -> Dict:
        vals = [getter(r) for r in runs]
        return {"mean": statistics.mean(vals),
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "per_seed": vals}

    overall = agg(lambda r: r["overall_topple_rate"])
    deadbands = {f"pm{int(db)}": agg(lambda r, db=db: r["deadband_fraction"][f"pm{int(db)}"])
                 for db in DEADBANDS}
    per_shape = {}
    for s in SHAPES:
        rates = [r["per_shape"][s]["topple_rate"] for r in runs
                 if r["per_shape"][s]["topple_rate"] is not None]
        per_shape[s] = {"topple_rate_mean": statistics.mean(rates) if rates else None,
                        "topple_rate_std": (statistics.pstdev(rates) if len(rates) > 1 else 0.0)
                        if rates else None,
                        "n_per_seed": [r["per_shape"][s]["n"] for r in runs]}
    return {"n_per_seed": n, "seeds": seeds, "overall_topple_rate": overall,
            "deadband_fraction": deadbands, "per_shape": per_shape, "runs": runs}


def _hist(values: List[float], edges: List[int]) -> List[int]:
    counts = [0] * (len(edges) - 1)
    for v in values:
        for b in range(len(edges) - 1):
            if edges[b] <= v < edges[b + 1] or (b == len(edges) - 2 and v >= edges[-1]):
                counts[b] += 1
                break
    return counts


# --------------------------------------------------------------------------- #
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="probe-outcome stability study")
    p.add_argument("--out", default="runs/stability")
    p.add_argument("--sweep-points", type=int, default=25)
    p.add_argument("--grip", type=float, default=0.5, help="fixed friction factor for the sweep")
    p.add_argument("--hardness", type=float, default=0.3, help="fixed restitution factor for the sweep")
    p.add_argument("--jitter-reps", type=int, default=24)
    p.add_argument("--eps-impulse-rel", type=float, default=0.03, help="relative push-impulse jitter (std)")
    p.add_argument("--eps-az-deg", type=float, default=2.0, help="push azimuth jitter in degrees (std)")
    p.add_argument("--population", type=int, default=200, help="scenes per seed")
    p.add_argument("--pop-seeds", type=int, default=5,
                   help="number of seeds for the population study (multi-seed mean±std)")
    p.add_argument("--pop-seed0", type=int, default=1234,
                   help="first population seed (matches the generator default)")
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    if not G._HAVE_MUJOCO:
        raise SystemExit("this study needs the 'mujoco' package: pip install -e \".[mujoco]\"")
    args = parse_args(argv)
    cfg = PhysicsConfig()

    print("[stability] L1 determinism ...")
    det = layer_determinism(cfg)
    print(f"           max field diff over {det['reps']} reps = {det['max_abs_field_diff']:.3g} "
          f"({'DETERMINISTIC' if det['deterministic'] else 'NONDETERMINISTIC ***'})")

    print(f"[stability] L2 boundary sweep ({args.sweep_points} pts/shape) ...")
    sweep = layer_boundary_sweep(cfg, args.sweep_points, args.grip, args.hardness)
    for s, d in sweep["per_shape"].items():
        cx = d["crossover_heaviness"]
        cx_s = f"{cx:.3f}" if cx is not None else "none"
        print(f"           {s:9s} [{d['label_class']:8s}] topple_frac={d['topple_fraction']:.2f} "
              f"crossover_h={cx_s} max_slope={d['max_slope_deg_per_h']:6.0f}°/h "
              f"ambiguous={d['ambiguous_fraction']:.2f}")

    print(f"[stability] L3 action-sensitivity (eps_imp={args.eps_impulse_rel}, "
          f"reps={args.jitter_reps}) ...")
    sens = layer_sensitivity(cfg, sweep, args.jitter_reps, args.eps_impulse_rel, args.eps_az_deg)
    print("           per-shape @ own crossover:")
    for s, r in sens["per_shape"].items():
        print(f"             {s:9s} [{r['label_class']:8s}] h={r['heaviness']:.3f} "
              f"p_topple={r['p_topple']:.2f} flip_rate={r['flip_rate']:.2f} "
              f"tilt={r['tilt_mean']:6.1f}±{r['tilt_std']:4.1f}°")
    print(f"           {FOCUS} spatial profile:")
    for label, r in sens["profile"].items():
        print(f"             {label:14s} h={r['heaviness']:.3f} p_topple={r['p_topple']:.2f} "
              f"flip_rate={r['flip_rate']:.2f} tilt={r['tilt_mean']:6.1f}±{r['tilt_std']:4.1f}°")

    seeds = list(range(args.pop_seed0, args.pop_seed0 + args.pop_seeds))
    print(f"[stability] L4 population (n={args.population}/seed × {len(seeds)} seeds) ...")
    pop = multi_population(cfg, args.population, seeds)
    o = pop["overall_topple_rate"]
    print(f"           overall topple_rate = {o['mean']:.3f} ± {o['std']:.3f}")
    for db, agg in pop["deadband_fraction"].items():
        print(f"           deadband |tilt-50|<={db[2:]:>2}°  fraction = {agg['mean']:.3f} ± {agg['std']:.3f}")
    for s, d in pop["per_shape"].items():
        if d["topple_rate_mean"] is None:
            print(f"           {s:9s} topple_rate = n/a")
        else:
            print(f"           {s:9s} topple_rate = {d['topple_rate_mean']:.3f} ± {d['topple_rate_std']:.3f}")

    os.makedirs(args.out, exist_ok=True)
    result = {
        "physics_config": {"sim_seconds": cfg.sim_seconds, "fps": cfg.fps,
                           "gravity": cfg.gravity, "timestep": 0.002},
        "topple_threshold_deg": TOPPLE_DEG,
        "physics_normalizers": PHYSICS_NORMALIZERS,
        "determinism": det,
        "boundary_sweep": sweep,
        "sensitivity": sens,
        "population": pop,
    }
    path = os.path.join(args.out, "probe_stability.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[stability] wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
