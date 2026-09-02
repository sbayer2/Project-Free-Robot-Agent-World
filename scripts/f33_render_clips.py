#!/usr/bin/env python
"""F33 clip renderer: probe rollout videos for the video legs
(docs/VIDEO_BASELINE.md sections 4-5, Amendments A and B).

Per (test scene, probe): re-simulates the probe with the exact generator
physics path (same MJCF builder, same specs from the scene's own
sample.json, same timestep) while capturing frames from ONE fixed oblique
camera, then encodes TWO clips through the frozen filter chain:

    percep  full episode (outcome visible)      ~32 sampled frames
    pred    pre-outcome prefix only             drop 4 / tilt 6 / push 10
                                                (tilt freeze-padded even;
                                                 Amendment D pairwise fusion)

Frozen apparatus honored here:
  * -vf "setpts=4*PTS,fps=4" then -movflags +faststart (Amendment B: the
    server's fps_target is hardcoded 4.0 and RESAMPLES, so time-stretch is
    the only frames-per-real-second lever; faststart makes clips decode on
    both the patched build and clean master -- Amendment A).
  * Per-clip verification: moov atom precedes mdat (pure-python atom scan),
    no rotation side-data (ffprobe), decoded frame count within +/-1 of the
    Amendment-B arithmetic (real_s x 16).
  * OUTCOME-MATCH GATE: the re-simulated trajectory is summarized with
    probes.summarize and compared per-field against the scene's stored
    ground truth. A mismatch means the clip shows DIFFERENT physics than
    the answer key scores against (e.g. MuJoCo version drift since dataset
    generation) -- the scene is flagged and excluded, never silently kept.
    push.toppled is compared only when the stored value is a hard 0/1
    (soft-topple probabilities cannot match a single re-sim).

Frames are piped raw RGB to ffmpeg (no temp PNGs). Capture times follow a
next-capture accumulator so the mean rate is exactly ENCODE_FPS (a naive
int() step divisor would run 25 fps and silently stretch the frozen 4x to
~4.17x). The trajectory cadence mirrors generate_mujoco._simulate's int()
arithmetic EXACTLY so summarize sees the generator's own sampling grid.

Usage:
    .venv/bin/python scripts/f33_render_clips.py --data data/pm_big \
        --out data/pm_big_clips [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import struct
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

# ---- frozen constants (docs/VIDEO_BASELINE.md; pinned by tests) ------------- #
STRETCH = 4                  # Amendment B: uniform, all clips, never per-probe
SERVER_FPS = 4               # mtmd fps_target, hardcoded server-side
ENCODE_FPS = 24              # prereg section 4 clip spec
RESOLUTION = 256             # prereg section 4
EPISODE_S = 2.0              # PhysicsConfig.sim_seconds (test pins the match)
PREFIX_S = {"drop": 0.25, "tilt": 0.30, "push": 0.60}  # section 4, onset-anchored
CAMERA = {"azimuth": 45.0, "elevation": -25.0, "distance": 2.6}  # fixed oblique
OUTCOME_RTOL = 1e-3          # outcome-match gate tolerance (same-version re-sim
OUTCOME_ATOL = 1e-4          # is deterministic; drift shows up far above this)


def expected_frames(real_seconds: float) -> int:
    """Amendment B arithmetic: real x STRETCH x SERVER_FPS (~ +/-1 at decode)."""
    return round(real_seconds * STRETCH * SERVER_FPS)


def sampled_count(n_src: int) -> int:
    """Frames the server's fps=4 sampler emits for n_src source frames at
    ENCODE_FPS under STRETCH: floor(n_src / ENCODE_FPS * STRETCH * SERVER_FPS).
    Matches observed decode counts exactly (48->32, 6->4, 8->5, 15->10)."""
    return int(n_src * STRETCH * SERVER_FPS / ENCODE_FPS)


def pad_to_even(n_src: int, max_pad: int = 4) -> int:
    """Amendment D: source frames to APPEND (duplicates of the final frame)
    so the sampled count is EVEN. Qwen-VL fuses frames pairwise and the
    orphan last frame of an odd count is discarded -- for a pre-outcome
    clip that is the most informative frame. A freeze-pad adds zero new
    dynamics and cannot leak an outcome."""
    if sampled_count(n_src) % 2 == 0:
        return 0
    for k in range(1, max_pad + 1):
        if sampled_count(n_src + k) % 2 == 0:
            return k
    raise ValueError(f"no even padding within {max_pad} frames of {n_src}")


def ffmpeg_args(out_path: str) -> list[str]:
    """The frozen encode chain, raw RGB frames on stdin."""
    return ["ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{RESOLUTION}x{RESOLUTION}", "-r", str(ENCODE_FPS), "-i", "-",
            "-vf", f"setpts={STRETCH}*PTS,fps={SERVER_FPS}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", out_path]


def capture_times(duration_s: float, fps: int = ENCODE_FPS) -> list[float]:
    """Frame-capture instants [0, duration): k/fps. Exact mean rate by
    construction (accumulator, not a step divisor)."""
    return [k / fps for k in range(math.ceil(duration_s * fps))]


def moov_before_mdat(path: str) -> bool:
    """Pure-python top-level MP4 atom scan: True iff a moov atom appears
    before the first mdat (the +faststart property; Amendment A). False if
    either atom is missing or the file is not parseable as atoms."""
    order = []
    with open(path, "rb") as fh:
        while True:
            head = fh.read(8)
            if len(head) < 8:
                break
            size = struct.unpack(">I", head[:4])[0]
            kind = head[4:8]
            if size == 1:  # 64-bit largesize
                big = fh.read(8)
                if len(big) < 8:
                    return False
                size = struct.unpack(">Q", big)[0]
                skip = size - 16
            elif size == 0:  # to end of file
                order.append(kind)
                break
            else:
                skip = size - 8
            order.append(kind)
            if skip < 0:
                return False
            fh.seek(skip, 1)
    if b"moov" not in order or b"mdat" not in order:
        return False
    return order.index(b"moov") < order.index(b"mdat")


def outcome_mismatches(resim: dict, stored: dict) -> list[str]:
    """Field names where the re-simulated outcome disagrees with the stored
    ground truth beyond tolerance. 'toppled' is checked only when stored is
    a hard 0/1 (soft-topple probabilities cannot match one re-sim)."""
    bad = []
    for k, sv in stored.items():
        if k not in resim:
            continue
        rv = float(resim[k])
        s = float(sv)
        if k == "toppled":
            if s in (0.0, 1.0) and rv != s:
                bad.append(k)
            continue
        if abs(rv - s) > OUTCOME_ATOL + OUTCOME_RTOL * abs(s):
            bad.append(k)
    return bad


def manifest_row(scene_id: str, probe: str, kind: str, path: str,
                 n_frames_expected: int, n_frames_decoded: int,
                 checks: dict) -> dict:
    return {"scene_id": scene_id, "probe": probe, "kind": kind, "path": path,
            "n_frames_expected": n_frames_expected,
            "n_frames_decoded": n_frames_decoded, "checks": checks}


# ---- runtime layers (ffprobe + MuJoCo; nothing here is unit-tested) --------- #

def ffprobe_frames(path: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out) if out.isdigit() else -1


def ffprobe_rotation_absent(path: str) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream_side_data=rotation", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return out == ""


def encode(frames: list, out_path: str) -> None:
    proc = subprocess.Popen(ffmpeg_args(out_path), stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")


def render_probe(record: dict, probe_rec: dict):
    """(frames_at_capture_times, resim_outcome_dict) for one (scene, probe).

    Model construction mirrors generate_mujoco.run_{drop,tilt,push}
    verbatim (same builder, same spec fields, same settle/window/wrench
    math for push); the trajectory cadence mirrors _simulate's int()
    arithmetic exactly so probes.summarize sees the generator's grid. The
    outcome-match gate is the runtime verifier of that mirroring."""
    import mujoco
    import numpy as np

    from pseudomarble import probes as P
    from pseudomarble.config import PhysicsConfig
    from pseudomarble.data.generate_mujoco import (
        _quat_to_up,
        build_mjcf,
        half_height,
    )
    from pseudomarble.materials import MaterialSampler  # noqa: F401 (doc link)

    shape = record["input"]["shape"]
    mat = record["input"]["material"]  # a material-id string in schema v2
    # Rebuild the Material the generator used from the sample's own record.
    from pseudomarble.materials import Material, PhysicsProps, VisualProps
    material = Material(
        name=mat if isinstance(mat, str) else mat.get("name", "sample"),
        visual=VisualProps(**record["material_truth"]["appearance_params"]),
        physics=PhysicsProps(**record["physics"]["raw"]),
    )
    cfg = PhysicsConfig()
    kind = probe_rec["probe"]
    spec = probe_rec.get("spec", {})

    force_fn = None
    if kind == "drop":
        z = half_height(shape) + spec.get("height", 0.6)
        model = mujoco.MjModel.from_xml_string(build_mjcf(shape, material, object_z=z))
    elif kind == "tilt":
        # MJCF `euler` is DEGREES (F34 instrument flag) -- this mirror of
        # run_tilt duplicated the unit bug and was the third call site,
        # caught by the outcome-match gate refusing to render mismatched
        # clips after the answer-key regeneration.
        angle = spec.get("angle_deg", 20.0)
        z = half_height(shape) + 0.05
        model = mujoco.MjModel.from_xml_string(
            build_mjcf(shape, material, object_z=z, ground_euler=(angle, 0.0, 0.0)))
    elif kind == "push":
        model = mujoco.MjModel.from_xml_string(build_mjcf(shape, material))
        settle, window = 0.4, 0.1
        force = spec.get("impulse", 1.5) / window
        rz = (spec.get("height_frac", 0.8) - 0.5) * 2.0 * half_height(shape)
        az = math.radians(spec.get("azimuth_deg", 0.0))
        fx, fy = force * math.cos(az), force * math.sin(az)
        wrench = [fx, fy, 0.0, -rz * fy, rz * fx, 0.0]

        def force_fn(t):  # noqa: E306
            return wrench if settle <= t < settle + window else [0.0] * 6
    else:
        raise ValueError(f"unknown probe {kind!r}")

    data = mujoco.MjData(model)
    body = model.body("object").id
    renderer = mujoco.Renderer(model, RESOLUTION, RESOLUTION)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, half_height(shape)]
    cam.azimuth, cam.elevation = CAMERA["azimuth"], CAMERA["elevation"]
    cam.distance = CAMERA["distance"]

    dt = model.opt.timestep
    n_steps = int(cfg.sim_seconds / dt)
    traj_every = max(1, int((1.0 / cfg.fps) / dt))     # generator's exact int()
    caps = capture_times(cfg.sim_seconds)
    frames, traj, next_cap = [], [], 0

    def grab():
        renderer.update_scene(data, camera=cam)
        return np.asarray(renderer.render(), dtype=np.uint8)

    for step in range(n_steps):
        t = step * dt
        if next_cap < len(caps) and t >= caps[next_cap] - 1e-9:
            frames.append(grab())
            next_cap += 1
        if force_fn is not None:
            data.xfrc_applied[body] = force_fn(t)
        mujoco.mj_step(model, data)
        if step % traj_every == 0:
            traj.append({"t": round(t, 4),
                         "pos": [float(c) for c in data.xpos[body]],
                         "up": [round(c, 4) for c in _quat_to_up(data.xquat[body])]})
    renderer.close()
    outcome = P.summarize(traj).to_dict()
    return frames, outcome, caps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pm_big")
    ap.add_argument("--out", default=None, help="default <data>_clips")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out_root = args.out or (args.data.rstrip("/") + "_clips")

    paths = sorted(glob.glob(f"{args.data}/{args.split}_*/sample.json"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"no {args.split} scenes under {args.data}")
    if args.dry_run:
        n = len(paths)
        print(f"{n} scenes x 3 probes -> {n*3*2} clips under {out_root}/")
        for k, s in PREFIX_S.items():
            src = sum(1 for t in capture_times(EPISODE_S) if t < s)
            src += pad_to_even(src)
            print(f"  {k}: pred {s}s real -> {sampled_count(src)} sampled "
                  f"(even); percep -> {sampled_count(48 + pad_to_even(48))}")
        return

    rows, excluded = [], []
    for p in paths:
        record = json.load(open(p))
        sid = record["scene_id"]
        scene_dir = os.path.join(out_root, sid)
        os.makedirs(scene_dir, exist_ok=True)
        for probe_rec in record["behavior"]["probes"]:
            kind = probe_rec["probe"]
            if kind not in PREFIX_S:
                continue
            frames, resim, caps = render_probe(record, probe_rec)
            bad = outcome_mismatches(resim, probe_rec["outcome"])
            if bad:
                excluded.append({"scene_id": sid, "probe": kind, "fields": bad,
                                 "resim": {k: resim[k] for k in bad},
                                 "stored": {k: probe_rec["outcome"][k] for k in bad}})
                print(f"  !! {sid}/{kind}: outcome mismatch on {bad} -- EXCLUDED")
                continue
            n_prefix = sum(1 for t in caps if t < PREFIX_S[kind])
            for clip_kind, clip_frames in (
                    ("percep", frames),
                    ("pred", frames[:n_prefix])):
                # Amendment D: freeze-pad so the sampled count is even
                # (pairwise fusion discards an orphan last frame).
                pad = pad_to_even(len(clip_frames))
                clip_frames = clip_frames + [clip_frames[-1]] * pad
                path = os.path.join(scene_dir, f"{kind}_{clip_kind}.mp4")
                encode(clip_frames, path)
                nf = ffprobe_frames(path)
                exp = sampled_count(len(clip_frames))
                checks = {
                    "faststart_moov_first": moov_before_mdat(path),
                    "rotation_absent": ffprobe_rotation_absent(path),
                    "frame_count_ok": abs(nf - exp) <= 1,
                    "even_sampled": exp % 2 == 0,
                }
                if not all(checks.values()):
                    print(f"  !! {sid}/{kind}/{clip_kind}: checks {checks}")
                rows.append(manifest_row(sid, kind, clip_kind, path, exp, nf, checks))
            print(f"  {sid}/{kind}: percep+pred ok "
                  f"({len(frames)} src frames, prefix {n_prefix})")

    manifest = {"stretch": STRETCH, "server_fps": SERVER_FPS,
                "encode_fps": ENCODE_FPS, "resolution": RESOLUTION,
                "camera": CAMERA, "prefixes_s": PREFIX_S,
                "clips": rows, "excluded": excluded,
                "all_checks_pass": all(all(r["checks"].values()) for r in rows)}
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "clips_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {out_root}/clips_manifest.json: {len(rows)} clips, "
          f"{len(excluded)} excluded, all_checks_pass={manifest['all_checks_pass']}")


if __name__ == "__main__":
    main()
