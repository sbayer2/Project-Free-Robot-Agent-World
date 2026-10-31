"""F35 outcome fields: the targets a model predicts, computed from a battery's
T0 recordings (docs/CROSS_INTERACTION.md §10.3a). Never a raw MSE over
trajectories: each family is summarized by a few published fields, each a
function of the T0 trajectory, standardized on train statistics at training.

Rules (cloud proposal 2026-10-27, with the three fixes agreed that day):
- distances spanning millimetres to metres enter as log10(d + 1 mm);
- no binary fields; a field that a scene cannot define (a box that tipped
  over in the kick, tumbled down the ramp) is NaN there: excluded from the
  loss and the scoring, and counted in the report;
- bounces are SPEED ratios per contact, e_k = v_out / v_in, from apex
  heights (v = sqrt(2 g h)). No-bounce rule: a rebound under BOUNCE_MM is no
  bounce; the next contact is then unobservable and e_{k+1} takes e_k's
  value (one material, one ratio);
- the strike's velocities are taken AT the impact time t_c, from fits to the
  frames after separation and before any other event (ball touching the
  ground, a second ball-box contact, the box stopping): constant velocity for
  the ball (horizontal flight), constant deceleration for the box (Coulomb
  sliding), so friction does not leak into the box's recoil speed.

Prints each field's zero-training readout both ways: the field from the
essence (is it a function of the material?) and the essence from the
family's fields (which axis does the family carry?).

    python scripts/f35_outcomes.py --data data/pm_f35_d6_sampled
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from f35_code_table import fmt, inverse_r2  # noqa: E402
from f35_generate_battery import plane_normal  # noqa: E402

G = 9.81
HALF = 0.15                  # m, the cube's half size (SHAPE "box")
EPS_M = 0.001                # log10(d + 1 mm)
BOUNCE_MM = 2.0              # a rebound under this is no bounce
STOP_MPS = 0.0125            # 0.1 mm per 8 ms frame: the box has stopped
TIP_UP = 0.9                 # up_z (or up . plane normal) below this: tipped


def log_dist(d: np.ndarray) -> np.ndarray:
    return np.log10(np.asarray(d) + EPS_M)


def apexes(h: np.ndarray) -> np.ndarray:
    """Heights of the rebound apexes in one height trace h(t) above the
    contact level: the local maxima after the first local minimum (the first
    contact), in order, including sub-threshold ones."""
    d = np.diff(h)
    down = np.where(d < 0)[0]
    if len(down) == 0:
        return np.zeros(0)
    c = down[0] + int(np.argmax(d[down[0]:] >= 0))       # first local minimum
    peaks = [i for i in range(c + 1, len(h) - 1) if h[i] >= h[i - 1] and h[i] > h[i + 1]]
    return h[peaks] if peaks else np.zeros(0)


def rebound_ratios(h: np.ndarray, n: int = 2) -> np.ndarray:
    """(e_1 .. e_n) speed ratios per contact for one height trace, h[0] the
    release height above contact, with the no-bounce rule."""
    a = np.clip(apexes(h), 0.0, None)
    heights = [h[0]] + list(a)
    e = []
    for k in range(1, n + 1):
        if k == 1 or heights[k - 1] * 1000.0 >= BOUNCE_MM:
            hk = heights[k] if k < len(heights) else 0.0
            e.append(math.sqrt(max(hk, 0.0) / heights[k - 1]))
        else:
            e.append(e[-1])                                   # no bounce: unobservable
    return np.array(e)


def n_bounces(h: np.ndarray) -> int:
    return int((apexes(h) * 1000.0 >= BOUNCE_MM).sum())


def stop_index(pos: np.ndarray, dt: float, start: int = 0) -> int:
    """First frame >= start from which the body stays below STOP_MPS."""
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    moving = np.where(v[start:] >= STOP_MPS)[0]
    return start + (int(moving[-1]) + 1 if len(moving) else 0)


def settle_time(pos: np.ndarray, t: np.ndarray) -> float:
    """Time from which the body stays within 1 mm of its final position."""
    far = np.where(np.linalg.norm(pos - pos[-1], axis=1) > 0.001)[0]
    return float(t[far[-1] + 1] if len(far) else t[0])


def tipped(up: np.ndarray, normal: np.ndarray) -> bool:
    return bool((up @ normal).min() < TIP_UP)


def strike_fields(box: np.ndarray, ball: np.ndarray, radius: float,
                  speed: float) -> tuple[float, float, dict]:
    """(v_ball(t_c) / v0, v_box(t_c) / v0, diagnostics) for one strike."""
    t = box[:, 0]
    dt = t[1] - t[0]
    gap = (box[:, 1] - HALF) - (ball[:, 1] + radius)
    k = int(np.argmax(gap <= 0))                              # first frame in contact
    rate = (gap[k - 1] - gap[k - 2]) / dt if k >= 2 else -speed
    t_c = t[k - 1] - gap[k - 1] / rate
    s = k + int(np.argmax(gap[k:] > 0.001))                   # separated
    events = [len(t)]
    again = np.where(gap[s:] <= 0)[0]
    if len(again):
        events.append(s + int(again[0]))                      # second ball-box contact
    ground = np.where(ball[s:, 3] <= radius + 0.003)[0]
    ball_end = min(events + ([s + int(ground[0])] if len(ground) else []))
    box_end = min(events + [stop_index(box[:, 1:4], dt, s) + 1])
    tb = t[s:ball_end] - t_c
    v_ball = (np.polyfit(tb, ball[s:ball_end, 1], 1)[0] if len(tb) >= 2
              else (ball[s, 1] - ball[s - 1, 1]) / dt)
    tx = t[s:box_end] - t_c
    x = box[s:box_end, 1]
    if len(tx) >= 3:
        v_box = np.polyfit(tx, x, 2)[1]                       # slope at t_c
    elif len(tx) == 2:
        v_box = (x[1] - x[0]) / dt
    else:                                                     # stopped within a frame
        v_box = 2.0 * (box[-1, 1] - box[k - 1, 1]) / max(t[s] - t_c, dt)
    return (float(v_ball / speed), float(v_box / speed),
            {"t_c": float(t_c), "ball_frames": len(tb), "box_frames": len(tx)})


def compute(data: str) -> tuple[dict[str, np.ndarray], dict[str, str], dict]:
    """-> {field: (S,) values, NaN where undefined}, {field: family}, diagnostics."""
    man = json.load(open(os.path.join(data, "manifest.json")))
    z = np.load(os.path.join(data, "battery.npz"))
    tr = {p: z[f"traj_{p}"].astype(float) for p in man["probes"]}
    S = len(z["essence"])
    up = np.array([0.0, 0.0, 1.0])
    f: dict[str, np.ndarray] = {}
    fam: dict[str, str] = {}
    diag: dict = {}

    def put(name: str, family: str, values) -> None:
        f[name], fam[name] = np.asarray(values, float), family

    d = tr["drop"]
    hd = d[:, :, 3] - np.median(d[:, -25:, 3], axis=1, keepdims=True)
    e = np.array([rebound_ratios(h) for h in hd])
    put("drop.e1", "drop", e[:, 0])
    put("drop.e2", "drop", e[:, 1])
    put("drop.bounces", "drop", [n_bounces(h) for h in hd])
    put("drop.settle_s", "drop", [settle_time(r[:, 1:4], r[:, 0]) for r in d])

    tilts = []
    for a in man["ramp_deg"]:
        r = tr[f"ramp{a:04.1f}"]
        slide = np.linalg.norm(r[:, -1, 1:4] - r[:, 0, 1:4], axis=1)
        put(f"ramp{a:04.1f}.log_slide", "ramp", log_dist(slide))
        tilts.append(slide < 0.01)
    still = np.stack(tilts, 1)                                # (S, angles): did not slide
    tans = np.tan(np.radians(man["ramp_deg"]))
    put("ramp.tan_steepest_still", "ramp",
        [tans[np.where(row)[0]].max() if row.any() else 0.0 for row in still])

    for F in man["push_newtons"]:
        r = tr[f"push{int(F):04d}"]
        put(f"push{int(F):04d}.log_disp", "push",
            log_dist(np.linalg.norm(r[:, -1, 1:4] - r[:, 0, 1:4], axis=1)))

    st = man["strike"]
    sf = [strike_fields(b, bl, st["radius"], st["speed"])
          for b, bl in zip(tr["strike"], tr["strike_ball"], strict=True)]
    put("strike.v_ball", "strike", [x[0] for x in sf])
    put("strike.v_box", "strike", [x[1] for x in sf])
    diag["strike_fit_frames"] = {
        "ball_min": min(x[2]["ball_frames"] for x in sf),
        "box_min": min(x[2]["box_frames"] for x in sf),
        "box_under_3": int(sum(x[2]["box_frames"] < 3 for x in sf))}

    p = tr["scale_plate"]
    put("scale.sag", "scale", p[:, 0, 3] - p[:, p[0, :, 0] >= p[0, -1, 0] - 0.5, 3].mean(1))

    ball, box = tr["tap_ball"], tr["tap"]
    ht = ball[:, :, 3] - (box[:, :, 3] + HALF + man["tap"]["radius"])
    e = np.array([rebound_ratios(h) for h in ht])
    put("tap.e1", "tap", e[:, 0])
    put("tap.e2", "tap", e[:, 1])

    for v in man["kick_mps"]:
        r = tr[f"kick{v:.1f}"]
        dt = r[0, 1, 0] - r[0, 0, 0]
        tip = np.array([tipped(x[:, 4:7], up) for x in r])
        stop = np.array([stop_index(x[:, 1:4], dt) for x in r])
        dist = np.linalg.norm(r[np.arange(S), stop, 1:4] - r[:, 0, 1:4], axis=1)
        put(f"kick{v:.1f}.log_stop", "kick", np.where(tip, np.nan, log_dist(dist)))
        put(f"kick{v:.1f}.stop_s", "kick", np.where(tip, np.nan, r[np.arange(S), stop, 0]))
        diag[f"kick{v:.1f}_tipped"] = int(tip.sum())

    if "dropramp_deg" in man:
        a = man["dropramp_deg"]
        r = tr[f"dropramp{a:04.1f}"]
        n = plane_normal(a)
        downhill = -np.array([0.0, math.cos(math.radians(a)), math.sin(math.radians(a))])
        rest = n * (HALF + 0.0005)
        hn = (r[:, :, 1:4] - rest) @ n
        tip = np.array([tipped(x[:, 4:7], n) for x in r])
        put("dropramp.e1", "dropramp", [rebound_ratios(h, 1)[0] for h in hn])
        slide = (r[:, -1, 1:4] - r[:, 0, 1:4]) @ downhill
        put("dropramp.log_slide", "dropramp",
            np.where(tip, np.nan, log_dist(np.clip(slide, 0.0, None))))
        diag["dropramp_tumbled"] = int(tip.sum())
    return f, fam, diag


def readouts(f: dict[str, np.ndarray], fam: dict[str, str],
             E: np.ndarray) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Per field, R^2 of the field read from the 3 essence axes; per family,
    inverse R^2 of the essence read from its fields (scenes with any NaN
    field in the family dropped)."""
    per_field = {}
    for name, v in f.items():
        ok = np.isfinite(v)
        per_field[name] = float(inverse_r2(E[ok], v[ok, None])[0])
    per_family = {}
    for family in dict.fromkeys(fam.values()):
        X = np.stack([f[n] for n in f if fam[n] == family], 1)
        ok = np.isfinite(X).all(1)
        per_family[family] = inverse_r2(X[ok], E[ok]).tolist()
    return per_field, per_family


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    f, fam, diag = compute(args.data)
    E = np.load(os.path.join(args.data, "battery.npz"))["essence"].astype(float)
    per_field, per_family = readouts(f, fam, E)
    print(f"[f35-outcomes] {args.data}: {len(E)} scenes, {len(f)} fields")
    print("\nFIELD <- essence (R^2, kNN-5 / ridge best)     undefined  mean   sd")
    for name, v in f.items():
        ok = np.isfinite(v)
        print(f"  {name:26s} {per_field[name]:+.2f}   {int((~ok).sum()):5d}  "
              f"{np.nanmean(v):+.3f} {np.nanstd(v):.3f}")
    print("\nESSENCE <- family's fields (inverse R^2)")
    for family, r in per_family.items():
        print(f"  {family:10s} {fmt(np.array(r))}")
    print("\n" + json.dumps(diag))
    np.savez_compressed(os.path.join(args.data, "outcomes.npz"),
                        names=np.array(list(f)), families=np.array([fam[n] for n in f]),
                        values=np.stack(list(f.values()), 1))
    json.dump({"per_field": per_field, "per_family": per_family, "diagnostics": diag,
               "bounce_mm": BOUNCE_MM, "stop_mps": STOP_MPS, "tip_up": TIP_UP},
              open(os.path.join(args.data, "outcomes.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
