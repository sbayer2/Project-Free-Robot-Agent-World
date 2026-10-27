"""F35 redesigned probe battery (docs/CROSS_INTERACTION.md §5), simulated on
an existing world's materials so its code table compares one-to-one.

Reads each scene's material (physics.raw + appearance_params) from a source
world, runs the battery on a box with the repo's own MJCF builder and contact
model, and stores state trajectories only (no renders: tier T0). Probes:

- drop      the existing drop (0.6 m), restitution's probe.
- rampXX.X  thermometer ramp: the box starts AT REST, flush on a plane
            tilted XX.X deg (no 5 cm landing skid). Slides iff mu < tan(theta),
            up to contact softness; angles tile the friction range below the
            cube's 45 deg tipping angle.
- pushNNNN  graded push at the box's centre (no torque): a constant force of
            NNNN N for 0.1 s. The box moves iff the force beats static
            friction mu*M*g, so the ladder is a thermometer code for mu*M.
- strike    a 17 kg reference sphere (the geometric middle of the box
            masses, 1.35-216 kg) launched at 4 m/s into the resting box; the
            sphere's rebound and the box's recoil encode the mass ratio and
            restitution. The sphere is stored as strike_ball.
- scale     (D2) "weigh it": the box rests on a 1 kg plate held by a vertical
            spring (slide joint, k = 5300 N/m, c = 1500 N.s/m). Static
            compression is (M + m_plate) g / k -- mass with no friction or
            gravity cancellation in it; the heaviest box sinks ~0.4 m, the
            lightest ~4 mm. The plate is stored as scale_plate.
            D4 fix: in D2/D3 the box-plate contact took the box's soft,
            restitution-set parameters and a 2 cm plate, so bouncy boxes
            (r >= 0.53) tunnelled through it in 24 of 512 scenes, the
            outcome depending on the MuJoCo build. The plate now has contact
            priority with stiff, critically damped parameters and is 10 cm
            thick: the scale reads mass, not bounce.
- tap       (D2) a 0.2 kg, 3 cm ball dropped 0.3 m onto the resting box's
            top face; its rebounds read restitution while the box, on the
            ground, barely moves. The ball is stored as tap_ball.

- kickV.V   (D3) "curling stone": the resting box on flat ground is given a
            horizontal velocity of V.V m/s at t = 0. Under Coulomb friction it
            decelerates at mu*g whatever its mass, so x(t) = v0 t - mu g t^2 / 2
            until it stops at v0^2 / (2 mu g): friction with no mass in it. A
            cube may tip forward instead when mu > half-width / COM height = 1.

Restitution convention (D1 strike, D2 tap): the reference ball's contact
parameters are set to the BOX's restitution, so each contact's restitution
is attributed to the box. Physically restitution belongs to the pair; this
is an authored modeling convention, not a simulated fact.
D5 fix: in D1-D4 that geom-level solref also applied to the ball's contacts
with the ground, whose friction is the box's own (build_mjcf), so the
strike ball's floor bounces replayed the box's restitution and friction
with the box not involved. An explicit ball-ground <pair> now gives those
contacts fixed, material-independent parameters; the convention is left
only on ball-box contacts. D5 also records at 125 fps (every 4 steps), the
R1 frame rate.

    python scripts/f35_generate_battery.py --source data/pm_f31fix_loud \
        --out data/pm_f35_battery
    python scripts/f35_generate_battery.py --sample 512 --seed 1234 \
        --out data/pm_f35_battery_sampled      # no source world: mujoco + numpy only

--sample draws fresh materials from MaterialSampler instead of reading a
rendered world, so the battery runs anywhere MuJoCo does (a CPU sandbox in
seconds). The physics is the same function of the materials either way.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

RAMP_DEG = (5.7, 11.3, 16.7, 24.2, 31.0, 38.7)   # tan = 0.10 0.20 0.30 0.45 0.60 0.80
PUSH_N = (15.0, 60.0, 240.0, 960.0, 3840.0)      # x 0.1 s = 1.5 ... 384 N.s
STRIKE_MASS, STRIKE_SPEED, STRIKE_R, STRIKE_GAP = 17.0, 4.0, 0.05, 0.25
SCALE_K, SCALE_C, SCALE_PLATE_MASS, SCALE_Z = 5300.0, 1500.0, 1.0, 0.5
SCALE_PLATE_HALF = 0.05          # m; D4 (was 0.01)
SCALE_PLATE_SOLREF = "0.01 1"    # timeconst 10 ms (>= 2 timesteps), critical damping; D4
TAP_MASS, TAP_R, TAP_HEIGHT = 0.2, 0.03, 0.3
KICK_MPS = (1.0, 3.0)
SHAPE = "box"
DROP_HEIGHT = 0.6                                 # m, gap below the box (as DropSpec)
SIM_SECONDS, FPS = 2.0, 125                       # D5: 125 fps (was 60, i.e. 62.5 Hz)
BALL_GROUND_FRICTION = 0.5                        # D5: fixed ball-ground contact,
BALL_GROUND_SOLREF = "0.01 1"                     # stiff and critically damped


def probe_names() -> list[str]:
    return (["drop"] + [f"ramp{a:04.1f}" for a in RAMP_DEG]
            + [f"push{int(f):04d}" for f in PUSH_N] + ["strike", "strike_ball"]
            + ["scale", "scale_plate", "tap", "tap_ball"]
            + [f"kick{v:.1f}" for v in KICK_MPS])


def plane_normal(theta_deg: float) -> np.ndarray:
    """Unit normal of the ground after MJCF euler (theta, 0, 0): R_x(theta) z."""
    t = math.radians(theta_deg)
    return np.array([0.0, -math.sin(t), math.cos(t)])


def replace_once(xml: str, old: str, new: str) -> str:
    if xml.count(old) != 1:
        raise RuntimeError(f"expected exactly one {old!r} in the MJCF")
    return xml.replace(old, new)


def material_from(record: dict):
    from pseudomarble import materials as M

    raw, ap = record["physics"]["raw"], record["material_truth"]["appearance_params"]
    visual = M.VisualProps(base_color=tuple(ap["base_color"]), roughness=ap["roughness"],
                           metallic=ap["metallic"], transmission=ap.get("transmission", 0.0),
                           ior=ap.get("ior", 1.45))
    physics = M.PhysicsProps(density=raw["density"], friction=raw["friction"],
                             restitution=raw["restitution"])
    return M.Material(name=record["scene_id"], visual=visual, physics=physics)


def _record(model, data, bodies: list[str], seconds: float = SIM_SECONDS,
            fps: int = FPS, force_fn=None) -> dict[str, list]:
    """Step and record (t, pos, up) for each named body, generate_mujoco's cadence."""
    import mujoco

    from pseudomarble.data.generate_mujoco import _quat_to_up

    ids = {b: model.body(b).id for b in bodies}
    n_steps = int(seconds / model.opt.timestep)
    every = max(1, round((1.0 / fps) / model.opt.timestep))
    out: dict[str, list] = {b: [] for b in bodies}
    for step in range(n_steps):
        t = step * model.opt.timestep
        if force_fn is not None:
            data.xfrc_applied[ids["object"]] = force_fn(t)
        mujoco.mj_step(model, data)
        if step % every == 0:
            for b, i in ids.items():
                out[b].append([t, *map(float, data.xpos[i]), *_quat_to_up(data.xquat[i])])
    return out


def run_ramp(mat, theta: float) -> list:
    import mujoco

    from pseudomarble.data.generate_mujoco import build_mjcf, half_height

    xml = build_mjcf(SHAPE, mat, ground_euler=(theta, 0.0, 0.0))
    pos = plane_normal(theta) * (half_height(SHAPE) + 0.0005)
    xml, n = re.subn(r'<body name="object" pos="[^"]*">',
                     f'<body name="object" euler="{theta} 0 0" '
                     f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}">', xml)
    if n != 1:
        raise RuntimeError("expected exactly one object body in the MJCF")
    model = mujoco.MjModel.from_xml_string(xml)
    return _record(model, mujoco.MjData(model), ["object"])["object"]


def run_push(mat, force: float) -> list:
    import mujoco

    from pseudomarble.data.generate_mujoco import build_mjcf

    model = mujoco.MjModel.from_xml_string(build_mjcf(SHAPE, mat))
    settle, window = 0.4, 0.1
    wrench = [force, 0.0, 0.0, 0.0, 0.0, 0.0]          # at the centre: no torque
    return _record(model, mujoco.MjData(model), ["object"],
                   force_fn=lambda t: wrench if settle <= t < settle + window else [0.0] * 6
                   )["object"]


def _ball_xml(name: str, pos: tuple[float, float, float], radius: float, mass: float,
              mat) -> str:
    """A reference sphere whose contacts carry the box's restitution (the
    authored convention in the module docstring)."""
    from pseudomarble.data.generate_mujoco import _restitution_to_solref

    vol = 4.0 / 3.0 * math.pi * radius ** 3
    return (f'<body name="{name}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"><freejoint/>'
            f'<geom name="{name}_geom" type="sphere" size="{radius}" density="{mass / vol:.1f}" '
            f'friction="0.01 0.005 0.0001" '
            f'solref="{_restitution_to_solref(mat.physics.restitution)}" '
            f'solimp="0.9 0.95 0.001" rgba="0.2 0.2 0.2 1"/></body>')


def with_ball(xml: str, ball: str, name: str = "ball") -> str:
    """Insert a reference ball into a build_mjcf scene, with an explicit
    ball-ground <pair> (D5): a pair overrides the per-geom parameter mixing,
    so the ball's ground contacts carry no material property of the box."""
    pair = (f'<contact><pair geom1="{name}_geom" geom2="ground" '
            f'friction="{BALL_GROUND_FRICTION} {BALL_GROUND_FRICTION} 0.005 0.0001 0.0001" '
            f'solref="{BALL_GROUND_SOLREF}" solimp="0.9 0.95 0.001"/></contact>')
    xml = replace_once(xml, "</worldbody>", ball + "\n  </worldbody>")
    return replace_once(xml, "</mujoco>", pair + "\n</mujoco>")


def run_strike(mat) -> tuple[list, list]:
    import mujoco

    from pseudomarble.data.generate_mujoco import build_mjcf, half_height

    h = half_height(SHAPE)
    ball = _ball_xml("ball", (-(h + STRIKE_GAP + STRIKE_R), 0.0, h), STRIKE_R, STRIKE_MASS, mat)
    xml = with_ball(build_mjcf(SHAPE, mat), ball)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    j = model.body("ball").jntadr[0]
    data.qvel[model.jnt_dofadr[j]] = STRIKE_SPEED
    rec = _record(model, data, ["object", "ball"])
    return rec["object"], rec["ball"]


def run_scale(mat) -> tuple[list, list]:
    """Box at rest on a spring-mounted plate; record box and plate."""
    import mujoco

    from pseudomarble.data.generate_mujoco import build_mjcf, half_height

    vol = 0.6 * 0.6 * 2 * SCALE_PLATE_HALF
    # priority="1": the box-plate contact uses the PLATE's solref/solimp, not
    # the box's restitution-derived (soft, barely damped) ones.
    plate = (f'<body name="plate" pos="0 0 {SCALE_Z}">'
             f'<joint name="spring" type="slide" axis="0 0 1" stiffness="{SCALE_K}" '
             f'damping="{SCALE_C}"/>'
             f'<geom type="box" size="0.3 0.3 {SCALE_PLATE_HALF}" priority="1" '
             f'solref="{SCALE_PLATE_SOLREF}" solimp="0.95 0.99 0.001" '
             f'density="{SCALE_PLATE_MASS / vol:.1f}" rgba="0.5 0.5 0.5 1"/></body>'
             '\n  </worldbody>')
    z = SCALE_Z + SCALE_PLATE_HALF + half_height(SHAPE) + 0.0005
    xml = replace_once(build_mjcf(SHAPE, mat, object_z=z), "</worldbody>", plate)
    model = mujoco.MjModel.from_xml_string(xml)
    rec = _record(model, mujoco.MjData(model), ["object", "plate"])
    return rec["object"], rec["plate"]


def run_tap(mat) -> tuple[list, list]:
    """Small ball dropped onto the resting box's top face; record box and ball."""
    import mujoco

    from pseudomarble.data.generate_mujoco import build_mjcf, half_height

    top = 2 * half_height(SHAPE)
    ball = _ball_xml("ball", (0.0, 0.0, top + TAP_HEIGHT + TAP_R), TAP_R, TAP_MASS, mat)
    xml = with_ball(build_mjcf(SHAPE, mat), ball)
    model = mujoco.MjModel.from_xml_string(xml)
    rec = _record(model, mujoco.MjData(model), ["object", "ball"])
    return rec["object"], rec["ball"]


def run_kick(mat, speed: float) -> list:
    """Resting box given a horizontal velocity at t = 0; record the box."""
    import mujoco

    from pseudomarble.data.generate_mujoco import build_mjcf

    model = mujoco.MjModel.from_xml_string(build_mjcf(SHAPE, mat))
    data = mujoco.MjData(model)
    data.qvel[model.jnt_dofadr[model.body("object").jntadr[0]]] = speed
    return _record(model, data, ["object"])["object"]


def run_drop(mat) -> list:
    import mujoco

    from pseudomarble.data.generate_mujoco import build_mjcf, half_height

    model = mujoco.MjModel.from_xml_string(
        build_mjcf(SHAPE, mat, object_z=half_height(SHAPE) + DROP_HEIGHT))
    return _record(model, mujoco.MjData(model), ["object"])["object"]


def essence(mat) -> tuple[list[float], list[float]]:
    """(normalized, raw) essence, as samples.py labels a scene."""
    from pseudomarble.data.samples import _physics_labels

    lab = _physics_labels(mat.physics, [])
    keys = ("density", "friction", "restitution")
    return [lab["normalized"][k] for k in keys], [lab["raw"][k] for k in keys]


def source_material(path: str):
    rec = json.load(open(path))
    return rec["scene_id"], material_from(rec)


def sampled_materials(seed: int, n: int) -> list:
    """n materials drawn in sequence from one seeded MaterialSampler."""
    from pseudomarble import materials as M

    sampler = M.MaterialSampler(seed=seed)
    ids = [f"sampled_{i:06d}" for i in range(n)]
    return [(i, sampler.sample(i).material) for i in ids]


def simulate_scene(item) -> tuple[str, list[float], list[float], dict[str, list]]:
    scene_id, mat = item
    out = {"drop": run_drop(mat)}
    for a in RAMP_DEG:
        out[f"ramp{a:04.1f}"] = run_ramp(mat, a)
    for f in PUSH_N:
        out[f"push{int(f):04d}"] = run_push(mat, f)
    out["strike"], out["strike_ball"] = run_strike(mat)
    out["scale"], out["scale_plate"] = run_scale(mat)
    out["tap"], out["tap_ball"] = run_tap(mat)
    for v in KICK_MPS:
        out[f"kick{v:.1f}"] = run_kick(mat, v)
    norm, raw = essence(mat)
    return scene_id, norm, raw, out


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="world whose materials to reuse")
    src.add_argument("--sample", type=int, help="draw this many fresh materials instead")
    ap.add_argument("--seed", type=int, default=1234, help="with --sample")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.source:
        files = sorted(glob.glob(os.path.join(args.source, "*", "sample.json")))[:args.limit]
        items = [source_material(f) for f in files]
    else:
        items = sampled_materials(args.seed, args.sample)[:args.limit]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(simulate_scene, items, chunksize=4))
    names = probe_names()
    traj = {p: np.array([r[3][p] for r in results], dtype=np.float32) for p in names}
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "battery.npz"),
                        scene_ids=np.array([r[0] for r in results]),
                        essence=np.array([r[1] for r in results]),
                        raw=np.array([r[2] for r in results]),
                        **{f"traj_{p}": v for p, v in traj.items()})
    json.dump({"source": args.source or f"MaterialSampler, seed {args.seed}",
               "shape": SHAPE, "n_scenes": len(results), "drop_height": DROP_HEIGHT,
               "probes": names, "ramp_deg": RAMP_DEG, "push_newtons": PUSH_N,
               "push_window_s": 0.1, "strike": {"mass": STRIKE_MASS, "speed": STRIKE_SPEED,
                                                "radius": STRIKE_R, "gap": STRIKE_GAP},
               "scale": {"k": SCALE_K, "c": SCALE_C, "plate_mass": SCALE_PLATE_MASS,
                         "plate_z": SCALE_Z, "plate_half": SCALE_PLATE_HALF,
                         "plate_solref": SCALE_PLATE_SOLREF},
               "tap": {"mass": TAP_MASS, "radius": TAP_R, "height": TAP_HEIGHT},
               "kick_mps": KICK_MPS,
               "restitution_convention": "reference balls carry the box's solref "
                                         "on ball-box contacts only",
               "ball_ground": {"friction": BALL_GROUND_FRICTION,
                               "solref": BALL_GROUND_SOLREF},
               "design_iteration": "D5",
               "sim_seconds": SIM_SECONDS, "fps": FPS,
               "columns": ["t", "x", "y", "z", "up_x", "up_y", "up_z"]},
              open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print(f"[f35-battery] {len(results)} scenes x {len(names)} recordings -> {args.out} "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
