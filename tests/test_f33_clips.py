"""Tests for the F33 clip renderer's pure core (scripts/f33_render_clips.py).

No MuJoCo, no ffmpeg. Pins the frozen clip constants against the prereg
(docs/VIDEO_BASELINE.md section 4 + Amendments A/B), the Amendment-B frame
arithmetic, the frozen ffmpeg filter chain, the capture-time accumulator
(the naive int() divisor would silently run 25 fps and stretch the frozen
4x to ~4.17x), the pure-python moov-before-mdat scanner, and the
outcome-match gate semantics including the soft-topple exemption.
"""

from __future__ import annotations

import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from f33_render_clips import (  # noqa: E402
    CAMERA,
    ENCODE_FPS,
    EPISODE_S,
    PREFIX_S,
    RESOLUTION,
    SERVER_FPS,
    STRETCH,
    capture_times,
    expected_frames,
    ffmpeg_args,
    manifest_row,
    moov_before_mdat,
    outcome_mismatches,
)


def test_frozen_clip_constants_pinned():
    """Prereg section 4 + Amendment B. Any drift fails here."""
    assert STRETCH == 4 and SERVER_FPS == 4
    assert ENCODE_FPS == 24 and RESOLUTION == 256
    assert PREFIX_S == {"drop": 0.25, "tilt": 0.30, "push": 0.60}
    from pseudomarble.config import PhysicsConfig, RenderConfig
    assert EPISODE_S == PhysicsConfig().sim_seconds
    assert CAMERA["distance"] == RenderConfig().camera_radius


def test_expected_frames_matches_amendment_b_arithmetic():
    """real x 16: drop ~4, tilt ~5, push ~10, full episode ~32."""
    assert expected_frames(PREFIX_S["drop"]) == 4
    assert expected_frames(PREFIX_S["tilt"]) == 5
    assert expected_frames(PREFIX_S["push"]) == 10
    assert expected_frames(EPISODE_S) == 32


def test_ffmpeg_args_carry_the_frozen_chain():
    args = ffmpeg_args("out.mp4")
    joined = " ".join(args)
    assert f"setpts={STRETCH}*PTS,fps={SERVER_FPS}" in joined
    assert "+faststart" in joined
    assert "yuv420p" in joined                       # broad-decoder pixel format
    assert f"{RESOLUTION}x{RESOLUTION}" in joined
    assert args[-1] == "out.mp4"
    assert "-r" in args and args[args.index("-r") + 1] == str(ENCODE_FPS)


def test_capture_times_exact_rate_and_prefix_counts():
    """48 instants over 2.0 s at exactly 1/24 spacing; prefix membership
    yields the source-frame counts the stretch arithmetic relies on."""
    caps = capture_times(EPISODE_S)
    assert len(caps) == 48
    assert caps[0] == 0.0
    assert caps[1] == pytest.approx(1 / 24)
    assert caps[-1] < EPISODE_S
    assert sum(1 for t in caps if t < PREFIX_S["drop"]) == 6    # 0.25s @ 24fps
    assert sum(1 for t in caps if t < PREFIX_S["tilt"]) == 8    # was int()-trap
    assert sum(1 for t in caps if t < PREFIX_S["push"]) == 15


def _atom(kind: bytes, payload: bytes = b"") -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def test_moov_scanner_orders_and_negatives(tmp_path):
    good = tmp_path / "good.mp4"
    good.write_bytes(_atom(b"ftyp", b"isom") + _atom(b"moov", b"x" * 10)
                     + _atom(b"mdat", b"y" * 20))
    assert moov_before_mdat(str(good)) is True

    bad = tmp_path / "bad.mp4"
    bad.write_bytes(_atom(b"ftyp", b"isom") + _atom(b"mdat", b"y" * 20)
                    + _atom(b"moov", b"x" * 10))
    assert moov_before_mdat(str(bad)) is False

    nomoov = tmp_path / "nomoov.mp4"
    nomoov.write_bytes(_atom(b"ftyp") + _atom(b"mdat", b"y"))
    assert moov_before_mdat(str(nomoov)) is False


def test_moov_scanner_handles_largesize_and_size_zero(tmp_path):
    # 64-bit largesize mdat, then moov -> False (mdat first)
    large_mdat = struct.pack(">I", 1) + b"mdat" + struct.pack(">Q", 16 + 4) + b"data"
    f = tmp_path / "large.mp4"
    f.write_bytes(_atom(b"ftyp") + large_mdat + _atom(b"moov", b"x"))
    assert moov_before_mdat(str(f)) is False
    # size==0 (to EOF) final moov after mdat -> still False; before -> True
    f2 = tmp_path / "zero.mp4"
    f2.write_bytes(_atom(b"ftyp") + _atom(b"moov", b"x")
                   + struct.pack(">I", 0) + b"mdat" + b"tail")
    assert moov_before_mdat(str(f2)) is True


def test_outcome_mismatch_gate_tolerances():
    stored = {"toppled": 1.0, "settle_time": 1.2, "slid_distance": 0.5,
              "n_bounces": 2.0, "max_height": 0.9, "path_length": 1.1,
              "final_tilt_deg": 88.0}
    same = dict(stored)
    assert outcome_mismatches(same, stored) == []
    near = dict(stored, settle_time=1.2 + 1e-5)          # inside tolerance
    assert outcome_mismatches(near, stored) == []
    off = dict(stored, slid_distance=0.6)                # far outside
    assert outcome_mismatches(off, stored) == ["slid_distance"]
    flipped = dict(stored, toppled=0.0)                  # hard bool mismatch
    assert "toppled" in outcome_mismatches(flipped, stored)


def test_outcome_mismatch_soft_topple_exempt():
    """A soft-topple probability (not a hard 0/1) cannot match one re-sim;
    the gate must skip it rather than fail every soft-labeled scene."""
    stored = {"toppled": 0.4, "settle_time": 1.0}
    resim = {"toppled": 1.0, "settle_time": 1.0}
    assert outcome_mismatches(resim, stored) == []


def test_manifest_row_schema():
    r = manifest_row("test_000001", "drop", "pred", "x/drop_pred.mp4", 4, 4,
                     {"faststart_moov_first": True, "rotation_absent": True,
                      "frame_count_ok": True})
    for k in ("scene_id", "probe", "kind", "path", "n_frames_expected",
              "n_frames_decoded", "checks"):
        assert k in r
    import json
    json.dumps(r)
