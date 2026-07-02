"""Tests for the LLM world-model transfer harness (``llm_transfer.py``).

Pure stdlib and fully offline: prompts, parsing, scoring, and the chat client
are exercised with synthetic samples and an injected fake transport — no
dataset, server, or model weights required.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble.llm_transfer import (  # noqa: E402
    SYSTEM_PROMPT,
    action_text,
    build_messages,
    chat_completion,
    extract_prediction,
    probe_outcomes,
    score_predictions,
    state_text,
    train_mean_outcomes,
)
from pseudomarble.probes import OUTCOME_FIELDS, OUTCOME_NORMALIZERS  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic scene fixtures (schema v2 shape, no disk access).
# --------------------------------------------------------------------------- #
def _outcome(**overrides):
    base = {"toppled": False, "settle_time": 1.0, "slid_distance": 0.1,
            "n_bounces": 2, "max_height": 0.5, "path_length": 0.8,
            "final_tilt_deg": 5.0}
    base.update(overrides)
    return base


def _sample(shape="sphere", density=2500.0, friction=0.6, restitution=0.4):
    return {
        "scene_id": "test_000000",
        "split": "test",
        "input": {"shape": shape, "material": "test_000000"},
        "physics": {"raw": {"density": density, "friction": friction,
                            "restitution": restitution}},
        "material_truth": {
            "appearance_params": {"base_color": [0.2, 0.4, 0.6, 1.0],
                                  "roughness": 0.3, "metallic": 0.9,
                                  "transmission": 0.05, "ior": 1.3},
        },
        "behavior": {"probes": [
            {"probe": "drop", "spec": {"kind": "drop", "height": 0.6},
             "outcome": _outcome(max_height=0.78, n_bounces=4)},
            {"probe": "tilt", "spec": {"kind": "tilt", "angle_deg": 20.0},
             "outcome": _outcome(slid_distance=0.09)},
            {"probe": "push", "spec": {"kind": "push", "impulse": 1.5,
                                       "height_frac": 0.8, "azimuth_deg": 0.0},
             "outcome": _outcome(toppled=True, final_tilt_deg=88.0)},
        ]},
    }


# --------------------------------------------------------------------------- #
# Prompt building.
# --------------------------------------------------------------------------- #
def test_state_text_essence_contains_physics_not_appearance():
    txt = state_text(_sample(), "essence")
    assert "2500.0" in txt and "0.600" in txt and "0.400" in txt
    assert "roughness" not in txt
    assert "sphere" in txt and "0.18" in txt  # shape + its MuJoCo size


def test_state_text_appearance_hides_physics():
    txt = state_text(_sample(), "appearance")
    assert "2500" not in txt and "density" not in txt
    assert "roughness 0.30" in txt and "metallic 0.90" in txt


def test_action_text_reflects_specs():
    s = _sample()
    drop, tilt, push = s["behavior"]["probes"]
    assert "0.6 m" in action_text(drop)
    assert "20.0 degrees" in action_text(tilt)
    assert "1.5 N*s" in action_text(push) and "80%" in action_text(push)


def test_build_messages_roles_and_fields():
    s = _sample()
    msgs = build_messages(s, s["behavior"]["probes"][0], "essence")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    for f in OUTCOME_FIELDS:
        assert f in msgs[0]["content"]


# --------------------------------------------------------------------------- #
# Response parsing.
# --------------------------------------------------------------------------- #
def test_extract_prediction_fenced_json():
    text = ("Let me reason... the sphere bounces.\n```json\n"
            '{"toppled": false, "settle_time": 1.5, "slid_distance": 0.0,\n'
            ' "n_bounces": 3, "max_height": 0.7, "path_length": 1.9,\n'
            ' "final_tilt_deg": 0.0}\n```\nDone.')
    pred = extract_prediction(text)
    assert pred["toppled"] == 0.0
    assert pred["n_bounces"] == 3.0
    assert set(pred) == set(OUTCOME_FIELDS)


def test_extract_prediction_takes_last_candidate():
    text = ('First guess: {"settle_time": 9.0, "toppled": true}\n'
            'Revised: {"settle_time": 1.0, "toppled": false}')
    pred = extract_prediction(text)
    assert pred["settle_time"] == 1.0 and pred["toppled"] == 0.0


def test_extract_prediction_coerces_strings_and_partial_fields():
    text = '```json\n{"toppled": "true", "n_bounces": "2", "unrelated": 1}\n```'
    pred = extract_prediction(text)
    assert pred == {"toppled": 1.0, "n_bounces": 2.0}


def test_extract_prediction_none_for_garbage():
    assert extract_prediction("no json here at all") is None
    assert extract_prediction('{"unrelated": 1}') is None


# --------------------------------------------------------------------------- #
# Scoring.
# --------------------------------------------------------------------------- #
def test_train_mean_outcomes_averages_per_probe_field():
    s1, s2 = _sample(), _sample()
    s2["behavior"]["probes"][0]["outcome"]["settle_time"] = 3.0  # drop: 1.0 vs 3.0
    mean = train_mean_outcomes([s1, s2])
    assert mean["drop"]["settle_time"] == 2.0
    assert mean["push"]["toppled"] == 1.0  # both toppled


def test_score_perfect_prediction_zero_mse():
    s = _sample()
    truths = probe_outcomes(s)
    mean = train_mean_outcomes([s])
    rows = [("s", k, {f: float(t[f]) for f in OUTCOME_FIELDS}, t)
            for k, t in truths.items()]
    rep = score_predictions(rows, mean)
    assert rep["mse"] == 0.0
    assert rep["n_parse_failures"] == 0 and rep["n_imputed_fields"] == 0
    assert rep["push_toppled_brier"] == 0.0


def test_score_error_uses_normalizers():
    s = _sample()
    truth = probe_outcomes(s)["drop"]
    mean = train_mean_outcomes([s])
    # Off by exactly one normalizer unit in every field -> per-field error 1.0.
    pred = {f: float(truth[f]) + OUTCOME_NORMALIZERS[f] for f in OUTCOME_FIELDS}
    rep = score_predictions([("s", "drop", pred, truth)], mean)
    assert abs(rep["mse"] - 1.0) < 1e-12
    assert abs(rep["per_field"]["drop.settle_time"]["mse"] - 1.0) < 1e-12


def test_score_imputes_baseline_for_missing_and_failed():
    s1, s2 = _sample(), _sample()
    s2["behavior"]["probes"][0]["outcome"]["settle_time"] = 3.0
    mean = train_mean_outcomes([s1, s2])
    truth = probe_outcomes(s1)["drop"]
    rows = [("s1", "drop", None, truth),                    # total parse failure
            ("s1", "drop", {"settle_time": 1.0}, truth)]    # partial answer
    rep = score_predictions(rows, mean)
    assert rep["n_parse_failures"] == 1
    # 7 fields imputed for the failure + 6 for the partial answer.
    assert rep["n_imputed_fields"] == 13
    # Imputed entries score exactly the baseline; the answered field is perfect,
    # so overall MSE must be <= baseline and > 0 (baseline is off on settle_time).
    assert 0.0 < rep["mse"] <= rep["baseline_mse"]
    assert rep["per_field"]["drop.settle_time"]["mse"] < \
        rep["per_field"]["drop.settle_time"]["baseline_mse"]


# --------------------------------------------------------------------------- #
# Chat client with injected transport (no network).
# --------------------------------------------------------------------------- #
def test_chat_completion_fake_transport_roundtrip():
    seen = {}

    def transport(url, body, headers):
        seen["url"] = url
        seen["payload"] = json.loads(body)
        seen["headers"] = headers
        return json.dumps(
            {"choices": [{"message": {"content": "reply-text"}}]}).encode()

    s = _sample()
    msgs = build_messages(s, s["behavior"]["probes"][0], "essence")
    out = chat_completion("http://host/v1", "m", msgs, api_key="sk-x",
                          transport=transport)
    assert out == "reply-text"
    assert seen["url"] == "http://host/v1/chat/completions"
    assert seen["payload"]["model"] == "m"
    assert seen["payload"]["temperature"] == 0.0
    assert seen["payload"]["messages"][0]["role"] == "system"
    assert seen["headers"]["Authorization"] == "Bearer sk-x"


def test_chat_completion_no_key_no_auth_header():
    def transport(url, body, headers):
        assert "Authorization" not in headers
        return json.dumps(
            {"choices": [{"message": {"content": "ok"}}]}).encode()

    s = _sample()
    msgs = build_messages(s, s["behavior"]["probes"][0], "essence")
    assert chat_completion("http://host/v1", "m", msgs, transport=transport) == "ok"
