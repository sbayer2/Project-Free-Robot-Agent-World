"""LLM world-model transfer test — score a language model against MuJoCo truth.

An external "language world model" (e.g. Qwen-AgentWorld, arXiv:2606.24597) is
prompted with the same state+action information our probes encode — one object's
shape and material, plus one action (drop / tilt / push) — and asked to predict
the summary outcome fields that MuJoCo actually measured. Scoring uses the same
per-field normalizers as the trained model's 21-dim behavior target
(``probes.OUTCOME_NORMALIZERS``), so the resulting MSE is directly comparable to
the behavior-head numbers in FINDINGS and to the predict-the-train-mean baseline.

Why "transfer": AgentWorld-style models are tuned on *digital* environments
(terminal / web / OS / SWE...); rigid-body physics is out-of-domain for that
tuning. This measures whether next-state-prediction skill transfers to a
physical substrate — against exact, resimulable ground truth, not an LLM judge.

Two state conditions:
  * ``essence``    — the model is told the true physical parameters (density,
    friction, restitution). Tests quantitative dynamics prediction alone.
  * ``appearance`` — the model is told only the *rendering* parameters (color,
    roughness, metallic, transmission, ior) and must infer the material first —
    the words->physics inverse, text-side. By construction (MaterialSampler
    noise) appearance is predictive but not invertible, so there is a ceiling.

Pure stdlib: prompt building, response parsing, and scoring are testable in any
session (``tests/test_llm_transfer.py``). The only I/O is ``chat_completion``, a
thin OpenAI-compatible HTTP client pointed at a local server (oMLX / mlx-lm
serve / llama.cpp); ``scripts/eval_llm_transfer.py`` is the runner.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pseudomarble.probes import (
    OUTCOME_FIELDS,
    OUTCOME_NORMALIZERS,
    PROBE_ORDER,
    TOPPLE_ANGLE_DEG,
)

CONDITIONS = ("essence", "appearance")

# Mirrors _simulate/build_mjcf in data/generate_mujoco.py — restated here so the
# prompt tells the model exactly what the simulator measured.
SIM_SECONDS = 2.0
REST_SPEED = 0.03  # probes.REST_SPEED; restated next to its prompt description

_FIELD_SPECS = f"""Report these outcome fields, measured over the {SIM_SECONDS:.0f}-second episode
(the object's position means its CENTER; sampling at 60 Hz):
- "toppled": true if the object's up-axis ends more than {TOPPLE_ANGLE_DEG:.0f} degrees from vertical.
- "settle_time": seconds until the object's speed last drops below {REST_SPEED} m/s (capped near {SIM_SECONDS:.2f}).
- "slid_distance": horizontal displacement of the center from start to rest, in meters.
- "n_bounces": integer count of upward reversals of vertical velocity (floor impacts).
- "max_height": peak height of the object's center above the ground, in meters.
- "path_length": total 3D distance traveled by the center, in meters.
- "final_tilt_deg": final angle of the object's up-axis from vertical, in degrees."""

SYSTEM_PROMPT = f"""You are a rigid-body physics world model: given an environment state and one
action, predict the next-state summary exactly as a physics simulator would measure it.

Simulation context (MuJoCo): timestep 0.002 s, gravity 9.81 m/s^2 downward, episode
length {SIM_SECONDS:.1f} s. The ground is a large flat plane whose sliding-friction
coefficient equals the object's. Contact bounciness follows the object's restitution
coefficient. The object is a rigid body, free to translate and rotate.

{_FIELD_SPECS}

Reason step by step first. Then output ONE fenced ```json code block containing exactly
the keys toppled, settle_time, slid_distance, n_bounces, max_height, path_length,
final_tilt_deg. The LAST fenced JSON block in your reply is taken as your answer."""


# --------------------------------------------------------------------------- #
# State + action -> prompt text.
# --------------------------------------------------------------------------- #
def _shape_text(shape: str) -> str:
    from pseudomarble.data.generate_mujoco import SHAPE_TO_GEOM

    geom = SHAPE_TO_GEOM[shape]
    return (
        f"a solid rigid {shape} (MuJoCo '{geom['type']}' geom, size attribute "
        f"\"{geom['size']}\" in meters — half-extents/radii per MuJoCo conventions; "
        f"half-height {geom['half_height']} m)"
    )


def state_text(sample: Dict, condition: str) -> str:
    """The environment-state block for one scene, under one information condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    shape = sample["input"]["shape"]
    lines = [f"Object: {_shape_text(shape)}, initially at rest."]
    if condition == "essence":
        raw = sample["physics"]["raw"]
        lines.append(
            "Material (ground truth): "
            f"density {raw['density']:.1f} kg/m^3, "
            f"sliding friction coefficient {raw['friction']:.3f}, "
            f"restitution coefficient {raw['restitution']:.3f}."
        )
    else:
        ap = sample["material_truth"]["appearance_params"]
        r, g, b, a = ap["base_color"]
        lines.append(
            "Material: unknown — you only see its rendered appearance: "
            f"base color RGBA ({r:.2f}, {g:.2f}, {b:.2f}, {a:.2f}), "
            f"roughness {ap['roughness']:.2f}, metallic {ap['metallic']:.2f}, "
            f"transmission {ap['transmission']:.2f}, ior {ap['ior']:.2f}. "
            "Infer plausible physical properties from this appearance first."
        )
    return "\n".join(lines)


def action_text(probe_record: Dict) -> str:
    """Describe one probe exactly as the generator executes it (generate_mujoco)."""
    kind = probe_record["probe"]
    spec = probe_record.get("spec", {})
    if kind == "drop":
        h = spec.get("height", 0.6)
        return (f"Action (drop): the object is released from rest with its center "
                f"{h} m above its resting height, and falls onto the flat ground.")
    if kind == "tilt":
        a = spec.get("angle_deg", 20.0)
        return (f"Action (tilt): the ground plane is inclined {a} degrees; the object "
                f"is released from rest just above the incline (0.05 m) and may slide, "
                f"roll, or stay put.")
    if kind == "push":
        imp = spec.get("impulse", 1.5)
        hf = spec.get("height_frac", 0.8)
        return (f"Action (push): after the object settles on flat ground for 0.4 s, a "
                f"horizontal impulse of {imp} N*s is delivered over 0.1 s at "
                f"{hf:.0%} of the object's height (above center height, so it also "
                f"applies a tipping torque).")
    raise ValueError(f"unknown probe kind {kind!r}")


def build_messages(sample: Dict, probe_record: Dict, condition: str) -> List[Dict]:
    """OpenAI-style chat messages for one (scene, probe, condition) query."""
    user = (
        f"{state_text(sample, condition)}\n\n{action_text(probe_record)}\n\n"
        "Predict the outcome fields."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# Response text -> outcome prediction.
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _candidate_jsons(text: str) -> List[str]:
    """JSON-object candidates, later-in-text last (fenced blocks preferred)."""
    fenced = _FENCE_RE.findall(text)
    if fenced:
        return fenced
    # Fall back to balanced top-level {...} spans anywhere in the reply.
    spans, depth, start = [], 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start:i + 1])
    return spans


def _coerce_field(name: str, value) -> Optional[float]:
    if name == "toppled":
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("true", "yes"):
                return 1.0
            if v in ("false", "no"):
                return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_prediction(text: str) -> Optional[Dict[str, float]]:
    """The LAST parseable outcome JSON in a model reply, fields coerced to float.

    Returns None when no candidate parses to an object containing at least one
    outcome field. Missing/uncoercible fields are simply absent from the dict —
    the scorer imputes them (counted, and scored as the train-mean baseline, so a
    model can't gain by omitting hard fields).
    """
    for cand in reversed(_candidate_jsons(text)):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        out = {}
        for f in OUTCOME_FIELDS:
            if f in obj:
                v = _coerce_field(f, obj[f])
                if v is not None:
                    out[f] = v
        if out:
            return out
    return None


# --------------------------------------------------------------------------- #
# Scoring — same normalization as the trained model's behavior target.
# --------------------------------------------------------------------------- #
def probe_outcomes(sample: Dict) -> Dict[str, Dict]:
    """{probe kind -> ground-truth outcome dict} for one scene (PROBE_ORDER kinds)."""
    return {
        r["probe"]: r["outcome"]
        for r in sample.get("behavior", {}).get("probes", [])
        if r.get("probe") in PROBE_ORDER
    }


def train_mean_outcomes(train_samples: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    """Per-(probe, field) mean over the train split — the predict-the-mean baseline."""
    sums: Dict[str, Dict[str, float]] = {k: {f: 0.0 for f in OUTCOME_FIELDS}
                                         for k in PROBE_ORDER}
    counts: Dict[str, int] = {k: 0 for k in PROBE_ORDER}
    for s in train_samples:
        for kind, outcome in probe_outcomes(s).items():
            counts[kind] += 1
            for f in OUTCOME_FIELDS:
                sums[kind][f] += float(outcome[f])
    return {
        kind: {f: (sums[kind][f] / counts[kind]) if counts[kind] else 0.0
               for f in OUTCOME_FIELDS}
        for kind in PROBE_ORDER
    }


def score_predictions(
    rows: Sequence[Tuple[str, str, Optional[Dict[str, float]], Dict]],
    train_mean: Dict[str, Dict[str, float]],
) -> Dict:
    """Score (scene_id, probe_kind, prediction-or-None, truth_outcome) rows.

    Per entry the error is ``((pred - truth) / OUTCOME_NORMALIZERS[field])**2`` —
    identical to the trained model's normalized behavior-MSE — and the baseline
    predicts the train-split mean. Missing predictions (parse failure or absent
    field) are imputed with the baseline value and counted in ``imputed``.
    """
    field_err: Dict[str, List[float]] = {}
    field_base: Dict[str, List[float]] = {}
    brier: List[float] = []
    n_parse_failures = 0
    n_imputed = 0
    for scene_id, kind, pred, truth in rows:
        if pred is None:
            n_parse_failures += 1
            pred = {}
        for f in OUTCOME_FIELDS:
            t = float(truth[f])
            base = train_mean[kind][f]
            p = pred.get(f)
            if p is None:
                n_imputed += 1
                p = base
            norm = OUTCOME_NORMALIZERS[f]
            key = f"{kind}.{f}"
            field_err.setdefault(key, []).append(((p - t) / norm) ** 2)
            field_base.setdefault(key, []).append(((base - t) / norm) ** 2)
            if f == "toppled" and kind == "push":
                brier.append((min(1.0, max(0.0, p)) - t) ** 2)

    def _mean(vals: Sequence[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    all_err = [e for v in field_err.values() for e in v]
    all_base = [e for v in field_base.values() for e in v]
    mse, base_mse = _mean(all_err), _mean(all_base)
    per_field = {
        k: {"mse": _mean(field_err[k]), "baseline_mse": _mean(field_base[k])}
        for k in sorted(field_err)
    }
    per_probe = {}
    for kind in PROBE_ORDER:
        errs = [e for k, v in field_err.items() if k.startswith(kind + ".") for e in v]
        bases = [e for k, v in field_base.items() if k.startswith(kind + ".") for e in v]
        if errs:
            per_probe[kind] = {"mse": _mean(errs), "baseline_mse": _mean(bases)}
    return {
        "n_rows": len(rows),
        "n_parse_failures": n_parse_failures,
        "n_imputed_fields": n_imputed,
        "mse": mse,
        "baseline_mse": base_mse,
        "gain_over_mean": (base_mse / mse) if mse > 0 else None,
        "push_toppled_brier": _mean(brier) if brier else None,
        "per_probe": per_probe,
        "per_field": per_field,
    }


# --------------------------------------------------------------------------- #
# OpenAI-compatible chat client (the only I/O in this module).
# --------------------------------------------------------------------------- #
def chat_completion(
    base_url: str,
    model: str,
    messages: List[Dict],
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: float = 600.0,
    transport: Optional[Callable[[str, bytes], bytes]] = None,
) -> str:
    """POST /chat/completions and return the reply text.

    ``transport(url, body) -> raw response bytes`` is injectable for tests; the
    default uses urllib against a local server (oMLX / mlx-lm serve).
    """
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    if transport is None:
        def transport(u: str, b: bytes) -> bytes:  # pragma: no cover - network
            req = urllib.request.Request(
                u, data=b, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
    raw = json.loads(transport(url, body))
    try:
        return raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"unexpected chat-completions response: {raw}") from e
