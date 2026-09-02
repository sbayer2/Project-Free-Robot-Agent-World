"""Tests for the F33 video condition (scripts/f33_video.py + the
eval_llm_transfer.py wiring). Pure Python -- no server, no MuJoCo.

Pins: the frozen prompt clauses (Amendments B/C/C-addendum), the verified
llama.cpp wire shape (input_video + RAW base64, video-first, no format
field), the Amendment-D tripwire constants and floor, the confabulation
scanner's marker-grid semantics, and that the F11 conditions' argparse
surface is unchanged.
"""

from __future__ import annotations

import base64
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from f33_video import (  # noqa: E402
    CLAUSE_MARKERS,
    CLAUSE_SLOWMO,
    CLAUSE_TIMEBASE,
    MARKER_INTERVAL_S,
    OVERHEAD_TOKENS,
    PAIR_TOKENS,
    TRIPWIRE_MARGIN,
    build_video_messages,
    clip_path_for,
    confab_time_claims,
    expected_prompt_tokens,
    tripwire_ok,
    video_content_part,
    video_state_text,
)

from pseudomarble.llm_transfer import SYSTEM_PROMPT  # noqa: E402

SAMPLE = {"input": {"shape": "box", "material": "m"},
          "material_truth": {"appearance_params": {
              "base_color": [0.5, 0.5, 0.5, 1.0], "roughness": 0.5,
              "metallic": 0.1, "transmission": 0.0, "ior": 1.4}},
          "physics": {"raw": {"density": 900.0, "friction": 0.5,
                              "restitution": 0.4}}}
PROBE = {"probe": "drop", "spec": {"height": 0.6}, "outcome": {}}


def test_frozen_clauses_pinned():
    assert CLAUSE_SLOWMO == "The video is 4x slow motion."
    assert "5-second intervals" in CLAUSE_MARKERS
    assert "do not infer finer timing" in CLAUSE_MARKERS
    assert "real-world seconds" in CLAUSE_TIMEBASE
    assert "settle_time" in CLAUSE_TIMEBASE
    assert "4x slow motion" in CLAUSE_TIMEBASE


def test_video_content_part_wire_shape(tmp_path):
    """Verified llama.cpp shape: input_video + nested data key, RAW base64
    (a data: prefix makes the server's raw-base64 branch throw), and NO
    format field (accepted but never read -- we don't send it)."""
    clip = tmp_path / "c.mp4"
    payload = b"\x00\x00\x00\x08ftypmp42-fake-bytes"
    clip.write_bytes(payload)
    part = video_content_part(str(clip))
    assert part["type"] == "input_video"
    assert set(part["input_video"].keys()) == {"data"}
    data = part["input_video"]["data"]
    assert not data.startswith("data:")
    assert base64.b64decode(data) == payload


def test_build_video_messages_order_and_content(tmp_path):
    clip = tmp_path / "drop_pred.mp4"
    clip.write_bytes(b"x")
    msgs = build_video_messages(SAMPLE, PROBE, str(clip), "pred")
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    content = msgs[1]["content"]
    assert content[0]["type"] == "input_video"      # VIDEO FIRST, frozen
    assert content[1]["type"] == "text"
    text = content[1]["text"]
    for clause in (CLAUSE_SLOWMO, CLAUSE_MARKERS, CLAUSE_TIMEBASE):
        assert clause in text
    assert "the outcome is not shown" in text        # pred framing
    assert "drop" in text and "0.6" in text          # F11 action text present
    assert "format" not in str(content[0])

    percep = build_video_messages(SAMPLE, PROBE, str(clip), "percep")
    assert "the outcome is shown" in percep[1]["content"][1]["text"]


def test_video_state_text_kind_validation():
    with pytest.raises(KeyError):
        video_state_text(SAMPLE, "nope")


def test_clip_path_for_mapping_and_missing(tmp_path):
    d = tmp_path / "clips" / "test_000001"
    d.mkdir(parents=True)
    (d / "tilt_percep.mp4").write_bytes(b"x")
    p = clip_path_for(str(tmp_path / "clips"), "test_000001", "tilt", "percep")
    assert p.endswith("test_000001/tilt_percep.mp4")
    with pytest.raises(FileNotFoundError):
        clip_path_for(str(tmp_path / "clips"), "test_000001", "drop", "pred")
    with pytest.raises(ValueError):
        clip_path_for(str(tmp_path / "clips"), "test_000001", "tilt", "full")


def test_tripwire_constants_and_floor():
    """Amendment D fit: 1026/pair + 1056 overhead; measured 4-frame clip
    was 3108 -- passes; a text-only ~500-token prompt fails."""
    assert (PAIR_TOKENS, OVERHEAD_TOKENS, TRIPWIRE_MARGIN) == (1026, 1056, 0.9)
    assert expected_prompt_tokens(4) == 3108        # matches measurement
    assert expected_prompt_tokens(10) == 6186       # matches measurement
    assert tripwire_ok(3108, 4)
    assert tripwire_ok(int(3108 * 0.95), 4)         # within class margin
    assert not tripwire_ok(500, 4)                  # silently-dropped video
    assert not tripwire_ok(85, 4)                   # the observed failure sig


def test_confab_scanner_marker_grid_semantics():
    assert MARKER_INTERVAL_S == 5.0
    # fabricated fine-grained claims -> flagged
    assert confab_time_claims("contact at 0:04 then rest") != []
    assert confab_time_claims("it settles around 3.2 s later") != []
    # marker-grid values -> not confabulated
    assert confab_time_claims("at 0:00 the object is high, by 0:05 at rest") == []
    assert confab_time_claims("the frame at 5 s shows rest") == []
    # the scored JSON is exempt (settle_time legitimately numeric)
    tail = 'reasoning here\n```json\n{"settle_time": 1.04}\n```'
    assert confab_time_claims(tail) == []
    # ordinal references are not time claims
    assert confab_time_claims("around the 3rd or 4th frame") == []


def test_classify_response_amendment_e_and_1():
    """Amendment E: length + empty = budget starvation (APPARATUS).
    Amendment 1: stop + empty = empty_stop (MODEL emission failure --
    the pilot's 28,252-token cell; the original ("", "stop")->scoreable
    pin was wrong and is corrected by dated amendment)."""
    from f33_video import classify_response
    assert classify_response("", "length") == "budget_starved"
    assert classify_response("   ", "length") == "budget_starved"
    assert classify_response('{"toppled": 0}', "length") == "truncated"
    assert classify_response('{"toppled": 0}', "stop") == "scoreable"
    assert classify_response("", "stop") == "empty_stop"
    assert classify_response("   ", "stop") == "empty_stop"
    assert classify_response("x", None) == "scoreable"
    assert classify_response("", None) == "scoreable"  # no finish_reason info


def test_chat_completion_usage_parses_reasoning_fields():
    """The usage twin surfaces finish_reason + reasoning_content (and
    tolerates a null content field, the starvation shape)."""
    import json as j

    from f33_video import chat_completion_usage
    raw = {"choices": [{"finish_reason": "length",
                        "message": {"content": None,
                                    "reasoning_content": "thinking..."}}],
           "usage": {"prompt_tokens": 3108, "completion_tokens": 800}}

    def transport(u, b, h):
        return j.dumps(raw).encode()

    text, info = chat_completion_usage("http://x/v1", "m", [], transport=transport)
    assert text == ""                                   # null content -> ""
    assert info["finish_reason"] == "length"
    assert info["reasoning_content"] == "thinking..."
    assert info["usage"]["prompt_tokens"] == 3108


def test_chat_completion_usage_extra_body_reaches_request():
    """reasoning_effort (and any extra field) must land in the request
    body -- the Qwen3.8 template defaults to xhigh when unset, so the
    field is load-bearing for the essence-leg calibration."""
    import json as j

    from f33_video import chat_completion_usage
    captured = {}

    def transport(u, b, h):
        captured.update(j.loads(b))
        return j.dumps({"choices": [{"finish_reason": "stop",
                                     "message": {"content": "ok"}}],
                        "usage": {}}).encode()

    chat_completion_usage("http://x/v1", "m", [], transport=transport,
                          extra_body={"reasoning_effort": "medium"})
    assert captured["reasoning_effort"] == "medium"
    captured.clear()
    chat_completion_usage("http://x/v1", "m", [], transport=transport)
    assert "reasoning_effort" not in captured    # unset stays unset


def test_eval_runner_surface():
    """Video conditions added; F11 conditions and defaults untouched."""
    import eval_llm_transfer as ev
    args = ev.parse_args(["--data", "x"])
    assert args.condition == "essence"               # default unchanged
    assert args.views == 3 and args.max_tokens == 4096
    for c in ("essence", "appearance", "vision", "video_pred", "video_percep"):
        a = ev.parse_args(["--data", "x", "--condition", c])
        assert a.condition == c
    assert ev.parse_args(["--data", "x"]).clips_root is None
    assert ev.parse_args(["--data", "x"]).reasoning_effort is None
    a = ev.parse_args(["--data", "x", "--reasoning-effort", "medium"])
    assert a.reasoning_effort == "medium"
    assert ev.parse_args(["--data", "x"]).extra_body is None
    a = ev.parse_args(["--data", "x", "--extra-body",
                       '{"chat_template_kwargs": {"enable_thinking": true}}'])
    assert a.extra_body is not None
