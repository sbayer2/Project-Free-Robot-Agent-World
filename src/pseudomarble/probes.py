"""Probes (actions) and behavior outcomes — the "what does it do when acted on?"

This is the heart of the behavior-based task. Instead of asking a model to
regress three static material constants, we *act* on each object and ask it to
predict the **outcome** — which is what "understanding the physical essence of a
thing" actually means (README: the essence is what it does when you act on it).

Three canonical probes:
  * DROP  — release from a height. Reveals restitution + mass (bounces, settling).
  * TILT  — place on a ramp. Reveals friction (does it slide, and how far).
  * PUSH  — apply a horizontal impulse at a height. Reveals friction + mass +
            **shape**: a tall object topples where a squat one slides. This is the
            probe that makes shape and material interact, so "glass is glass"
            can no longer transfer trivially across shapes.

Outcomes are *summary* statistics (the chosen granularity): interpretable,
robust, and tractable on a laptop. The summarization here is pure-Python and
operates on a recorded trajectory, so it is unit-tested with synthetic
trajectories — no MuJoCo runtime required. The MuJoCo generator produces the
trajectories; this module turns them into outcomes.

A trajectory is a list of frames::

    {"t": float, "pos": [x, y, z], "up": [ux, uy, uz]}

where ``up`` is the object's local +Z axis expressed in world coordinates (the
generator computes it from the body orientation), so we can detect toppling
without any quaternion math here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

# Toppled if the object's up-axis has tilted more than this from world-up.
TOPPLE_ANGLE_DEG = 50.0
# Speed (m/s) below which the object is considered at rest, for settling time.
REST_SPEED = 0.03


# --------------------------------------------------------------------------- #
# Probe specifications.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DropSpec:
    height: float = 0.6
    kind: str = field(default="drop", init=False)


@dataclass(frozen=True)
class TiltSpec:
    angle_deg: float = 20.0
    kind: str = field(default="tilt", init=False)


@dataclass(frozen=True)
class PushSpec:
    impulse: float = 1.5       # newton-seconds, horizontal
    height_frac: float = 0.8   # where on the object's height the push lands (0..1)
    azimuth_deg: float = 0.0   # horizontal push direction
    kind: str = field(default="push", init=False)


def default_probes() -> List[object]:
    """The standard drop+tilt+push battery applied to every object."""
    return [DropSpec(), TiltSpec(), PushSpec()]


def spec_to_dict(spec) -> Dict:
    d = asdict(spec)
    d["kind"] = spec.kind
    return d


# --------------------------------------------------------------------------- #
# Outcome schema.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProbeOutcome:
    toppled: bool
    settle_time: float          # seconds until motion drops below REST_SPEED
    slid_distance: float        # horizontal displacement, start -> rest (m)
    n_bounces: int              # vertical-velocity sign flips (floor contacts)
    max_height: float           # peak z reached (m)
    path_length: float          # total 3D distance travelled (m)
    final_tilt_deg: float       # final angle of up-axis from world-up

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Trajectory -> outcome (pure-Python; unit-tested with synthetic trajectories).
# --------------------------------------------------------------------------- #
def _angle_from_up(up: Sequence[float]) -> float:
    n = math.sqrt(sum(c * c for c in up)) or 1.0
    cos = max(-1.0, min(1.0, up[2] / n))
    return math.degrees(math.acos(cos))


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def summarize(trajectory: Sequence[Dict]) -> ProbeOutcome:
    """Reduce a recorded trajectory to a summary ProbeOutcome."""
    if len(trajectory) < 2:
        raise ValueError("trajectory needs at least two frames")

    times = [f["t"] for f in trajectory]
    pos = [f["pos"] for f in trajectory]
    ups = [f["up"] for f in trajectory]

    # Path length and peak height.
    path_length = sum(_dist(pos[i], pos[i - 1]) for i in range(1, len(pos)))
    max_height = max(p[2] for p in pos)

    # Bounces: count upward zero-crossings of vertical velocity.
    vz = [(pos[i][2] - pos[i - 1][2]) / max(1e-9, times[i] - times[i - 1])
          for i in range(1, len(pos))]
    n_bounces = 0
    for i in range(1, len(vz)):
        if vz[i - 1] < -REST_SPEED and vz[i] > REST_SPEED:
            n_bounces += 1

    # Settling time: last moment the 3D speed exceeded REST_SPEED.
    speeds = [_dist(pos[i], pos[i - 1]) / max(1e-9, times[i] - times[i - 1])
              for i in range(1, len(pos))]
    settle_idx = 0
    for i, s in enumerate(speeds):
        if s > REST_SPEED:
            settle_idx = i + 1
    settle_time = times[settle_idx] - times[0]

    # Horizontal slide: start -> final, in the ground plane.
    slid_distance = math.hypot(pos[-1][0] - pos[0][0], pos[-1][1] - pos[0][1])

    final_tilt = _angle_from_up(ups[-1])
    toppled = final_tilt > TOPPLE_ANGLE_DEG

    return ProbeOutcome(
        toppled=toppled,
        settle_time=settle_time,
        slid_distance=slid_distance,
        n_bounces=n_bounces,
        max_height=max_height,
        path_length=path_length,
        final_tilt_deg=final_tilt,
    )


def soft_topple_probability(final_tilts: Sequence[float]) -> float:
    """Fraction of (jittered) push outcomes whose final tilt exceeds the topple
    threshold — a smooth [0,1] target that replaces the binary ``toppled`` near the
    chaotic tipping point (docs/FINDINGS.md F8). The sim is deterministic, so the
    spread comes from jittering the *action*, not from re-running the same push.
    """
    if not final_tilts:
        raise ValueError("need at least one final tilt to estimate P(topple)")
    return sum(1 for t in final_tilts if t > TOPPLE_ANGLE_DEG) / len(final_tilts)


# Order of the numeric outcome fields when flattened to a model target vector.
OUTCOME_FIELDS: Tuple[str, ...] = (
    "toppled", "settle_time", "slid_distance",
    "n_bounces", "max_height", "path_length", "final_tilt_deg",
)

# Per-field scales that map raw outcomes into roughly [0, 1] so a regression
# target isn't dominated by the large-magnitude fields (tilt in degrees, etc.).
OUTCOME_NORMALIZERS: Dict[str, float] = {
    "toppled": 1.0,
    "settle_time": 2.0,
    "slid_distance": 1.0,
    "n_bounces": 5.0,
    "max_height": 1.5,
    "path_length": 3.0,
    "final_tilt_deg": 180.0,
}

# Canonical probe order for the flattened behavior target vector.
PROBE_ORDER: Tuple[str, ...] = ("drop", "tilt", "push")
# Length of the full behavior target: one OUTCOME_FIELDS block per probe.
BEHAVIOR_DIM: int = len(PROBE_ORDER) * len(OUTCOME_FIELDS)


def _flatten(d: Dict, normalize: bool) -> List[float]:
    if normalize:
        return [float(d[k]) / OUTCOME_NORMALIZERS[k] for k in OUTCOME_FIELDS]
    return [float(d[k]) for k in OUTCOME_FIELDS]


def outcome_vector(outcome: ProbeOutcome, normalize: bool = False) -> List[float]:
    """Flatten one outcome to floats (bool -> 0/1) in a fixed field order."""
    return _flatten(outcome.to_dict(), normalize)


def outcome_vector_from_dict(d: Dict, normalize: bool = False) -> List[float]:
    """Same as ``outcome_vector`` but from a plain dict (as stored in sample.json)."""
    return _flatten(d, normalize)


def behavior_vector(probe_records: Sequence[Dict], normalize: bool = True) -> List[float]:
    """Assemble the full behavior target from a scene's probe records.

    Probes are emitted in ``PROBE_ORDER`` regardless of their order on disk, so
    the target vector layout is stable. Missing probes are zero-filled.
    """
    by_kind = {r.get("probe"): r.get("outcome", {}) for r in probe_records}
    vec: List[float] = []
    for kind in PROBE_ORDER:
        outcome = by_kind.get(kind)
        if outcome:
            vec.extend(_flatten(outcome, normalize))
        else:
            vec.extend([0.0] * len(OUTCOME_FIELDS))
    return vec


def behavior_field_names() -> List[str]:
    """Human-readable name for each entry of the behavior vector (``probe.field``)."""
    return [f"{kind}.{field}" for kind in PROBE_ORDER for field in OUTCOME_FIELDS]
