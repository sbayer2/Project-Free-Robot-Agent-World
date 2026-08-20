"""F33 video condition: prompt construction, wire format, and tripwires
(docs/VIDEO_BASELINE.md section 4, Amendments A-D).

Everything F33-specific lives here so the frozen F11 module
(pseudomarble/llm_transfer.py) is not touched: the F11 conditions must
stay byte-identical for comparability.

Wire format (verified live by the monitoring llama.cpp session; my
convention-mirroring guess would have failed twice):
  * content-part type is "input_video" with a nested "input_video" object
    (mirroring input_audio, NOT image_url; "video_url" is a hard 400);
  * the payload key is "data" carrying RAW base64 -- no "data:" URI prefix
    (video calls reject data URIs: the raw-base64 branch would decode the
    prefix too and throw);
  * a "format" field is accepted but never read -- we do not send one;
  * ordering is preserved in the prompt: VIDEO FIRST, then text (frozen --
    it is the shape behind every measured token count).

Frozen prompt clauses (Amendments B, C, C-addendum) are module constants
pinned by tests. The tripwire constants come from Amendment D's measured
fit on OUR clips (1026 tokens per frame pair + 1056 overhead).
"""
from __future__ import annotations

import base64
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pseudomarble.llm_transfer import (  # noqa: E402
    SYSTEM_PROMPT,
    action_text,
    state_text,
)

# ---- frozen prompt clauses (pinned by tests/test_f33_video.py) -------------- #
CLAUSE_SLOWMO = "The video is 4x slow motion."
CLAUSE_MARKERS = ("Frame timestamps are labeled only at 5-second intervals of "
                  "video time; do not infer finer timing than the evidence "
                  "supports.")
CLAUSE_TIMEBASE = ("Report every quantity, including settle_time, in "
                   "real-world seconds — the simulator's time base — "
                   "accounting for the 4x slow motion.")
KIND_TEXT = {"pred": "as the action begins (the outcome is not shown)",
             "percep": "from start to finish (the outcome is shown)"}

# ---- Amendment D tripwire constants (measured on OUR clips) ----------------- #
PAIR_TOKENS = 1026        # prompt tokens per fused frame pair
OVERHEAD_TOKENS = 1056    # fixed multimodal overhead
TRIPWIRE_MARGIN = 0.9     # +/-10% class check, lower bound is the apparatus floor
MARKER_INTERVAL_S = 5.0   # server timestamp markers, video time base


def video_content_part(clip_path: str) -> dict:
    """The verified llama.cpp wire shape: input_video + RAW base64 data."""
    with open(clip_path, "rb") as fh:
        raw = base64.b64encode(fh.read()).decode()
    return {"type": "input_video", "input_video": {"data": raw}}


def clip_path_for(clips_root: str, scene_id: str, probe: str, kind: str) -> str:
    """<root>/<scene>/<probe>_{pred|percep}.mp4; loud if missing."""
    if kind not in KIND_TEXT:
        raise ValueError(f"kind must be pred|percep, got {kind!r}")
    p = os.path.join(clips_root, scene_id, f"{probe}_{kind}.mp4")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} missing -- run scripts/f33_render_clips.py first")
    return p


def video_state_text(sample: dict, kind: str) -> str:
    """The state paragraph for video legs. Reuses the F11 vision-condition
    geometry line (state_text with n_images=0 describes shape; we replace
    its material sentence) -- kept minimal so the stills-vs-video contrast
    is about the input, not the prompt (prereg section 4)."""
    # First line of the F11 vision state block is the shape/geometry line.
    shape_line = state_text(sample, "vision", n_images=0).split("\n")[0]
    return (f"{shape_line}\n"
            f"Material: unknown — the attached video shows this exact object "
            f"{KIND_TEXT[kind]}. {CLAUSE_SLOWMO} {CLAUSE_MARKERS} "
            f"{CLAUSE_TIMEBASE} Infer plausible physical properties from "
            f"what you see first.")


def build_video_messages(sample: dict, probe_record: dict, clip_path: str,
                         kind: str) -> list:
    """OpenAI-style messages for one video-leg request. VIDEO FIRST, then
    text (frozen ordering); system prompt is F11's verbatim (field spec +
    JSON instruction unchanged for comparability)."""
    user_text = (f"{video_state_text(sample, kind)}\n\n"
                 f"{action_text(probe_record)}\n\nPredict the outcome fields.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            video_content_part(clip_path),
            {"type": "text", "text": user_text},
        ]},
    ]


def expected_prompt_tokens(n_frames: int, text_tokens: int = 0) -> int:
    """Amendment D fit: overhead + pair cost (frames are even by the
    renderer's rule, so n_frames/2 is exact)."""
    return OVERHEAD_TOKENS + PAIR_TOKENS * (n_frames // 2) + text_tokens


def tripwire_ok(prompt_tokens: int, n_frames: int) -> bool:
    """Amendment A.3/D: a silently-dropped video contributes ~no vision
    tokens, so prompt_tokens below 90% of the frames-only expectation is an
    APPARATUS failure (text tokens only ever push the total higher)."""
    return prompt_tokens >= TRIPWIRE_MARGIN * expected_prompt_tokens(n_frames)


# ---- Amendment C.4: confabulation scan (descriptive, no verdict) ------------ #
_TIME_CLAIM = re.compile(
    r"(?:\bat\s+)?\b(\d{1,2}):(\d{2})\b"        # 0:04 style
    r"|(?:\bat\s+|\baround\s+|\bby\s+)(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?\b",
    re.IGNORECASE)


def confab_time_claims(text: str) -> list:
    """Metric-time claims FINER than the marker grid, scanned in the prose
    before the final JSON block (the scored JSON is exempt -- settle_time
    legitimately carries a number). A claim is confabulated when the value
    is not a multiple of the 5-second marker interval."""
    cut = text.rfind("```")
    prose = text if cut == -1 else text[:text.rfind("```json")
                                        if "```json" in text else cut]
    claims = []
    for m in _TIME_CLAIM.finditer(prose):
        if m.group(1) is not None:
            val = int(m.group(1)) * 60 + int(m.group(2))
        else:
            val = float(m.group(3))
        if val % MARKER_INTERVAL_S != 0:
            claims.append(m.group(0).strip())
    return claims


# ---- usage-returning chat client (frozen F11 client stays untouched) -------- #

def chat_completion_usage(base_url: str, model: str, messages: list,
                          temperature: float = 0.0, max_tokens: int = 4096,
                          timeout: float = 600.0, api_key: str | None = None,
                          transport=None) -> tuple:
    """(text, usage_dict). Same scheme-validated urllib client as
    llm_transfer.chat_completion (that function discards usage; the F33
    tripwire needs prompt_tokens, and the frozen F11 module is not
    modified)."""
    import json as _json
    import urllib.parse
    import urllib.request

    url = base_url.rstrip("/") + "/chat/completions"
    body = _json.dumps({"model": model, "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if transport is None:
        scheme = urllib.parse.urlsplit(url).scheme
        if scheme not in ("http", "https"):
            raise ValueError(f"chat endpoint must be http(s); got {scheme!r}")

        def transport(u, b, h):  # pragma: no cover - network
            req = urllib.request.Request(u, data=b, headers=h)
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
                return resp.read()
    raw = _json.loads(transport(url, body, headers))
    try:
        return raw["choices"][0]["message"]["content"], raw.get("usage", {})
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"unexpected chat-completions response: {raw}") from e
