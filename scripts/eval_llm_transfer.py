"""Run the LLM world-model transfer test against a local OpenAI-compatible server.

Prompts a language world model (e.g. Qwen-AgentWorld served by oMLX / mlx-lm)
with each held-out scene's state + probe action, parses its predicted outcome,
and scores it with the SAME normalized MSE as the trained model's behavior head
(see ``pseudomarble/llm_transfer.py`` for the design and the two conditions).

Responses are cached one file per (scene, probe) under ``<out>/responses/``, so
an interrupted run resumes for free and the report can be rebuilt offline.

Example (server already running)::

    python scripts/eval_llm_transfer.py --data data/pm_big --split test \
        --base-url http://127.0.0.1:8080/v1 --model qwen-agentworld-35b \
        --condition essence --out runs/llm_transfer_essence

Do NOT run this while an MLX training job is using the GPU — one shared
unified-memory pool (docs/HARDWARE.md).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))  # F33 video helpers (f33_video)

from pseudomarble.llm_transfer import (  # noqa: E402
    build_messages,
    chat_completion,
    extract_prediction,
    probe_outcomes,
    scene_view_files,
    score_predictions,
    train_mean_outcomes,
)
from pseudomarble.probes import PROBE_ORDER  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM world-model transfer eval")
    p.add_argument("--data", required=True, help="dataset root (schema v2)")
    p.add_argument("--split", default="test", help="split to evaluate (held-out=test)")
    p.add_argument("--condition", default="essence",
                   choices=("essence", "appearance", "vision",
                            "video_pred", "video_percep"))
    p.add_argument("--views", type=int, default=3,
                   help="rendered views attached per scene (vision condition only)")
    p.add_argument("--clips-root", default=None,
                   help="F33 video conditions: clip dir (default <data>_clips; "
                        "run scripts/f33_render_clips.py first)")
    p.add_argument("--reasoning-effort", default=None,
                   choices=("xhigh", "medium", "low", "none"),
                   help="per-request reasoning_effort. The Qwen3.8 template "
                        "DEFAULTS to xhigh (maximal-reasoning instruction) "
                        "when unset; 'medium' injects nothing (neutral). "
                        "Setting this routes ALL conditions through the "
                        "usage-capturing client with the Amendment-E "
                        "classifier (budget starvation != parse failure)")
    p.add_argument("--system-prompt-file", default=None,
                   help="replace the F11 system prompt with this file's "
                        "contents (a DISCLOSED condition change -- e.g. the "
                        "F33 MLX no-think 'direct answer' probe: the frozen "
                        "prompt COMMANDS step-by-step reasoning, so a "
                        "no-computation condition must replace that clause; "
                        "never use for parity legs)")
    p.add_argument("--extra-body", default=None,
                   help="JSON object merged into every request body (e.g. "
                        "'{\"chat_template_kwargs\": {\"enable_thinking\": "
                        "true}}' for mlx_vlm.server, whose template DISABLES "
                        "thinking by default -- the opposite of llama.cpp). "
                        "Setting this routes ALL conditions through the "
                        "usage-capturing client, like --reasoning-effort")
    p.add_argument("--base-url", default="http://127.0.0.1:8080/v1",
                   help="OpenAI-compatible server base URL")
    p.add_argument("--model", default="default", help="served model name")
    p.add_argument("--api-key", default=os.environ.get("OMLX_API_KEY")
                   or os.environ.get("OPENAI_API_KEY"),
                   help="Bearer token for the server (default: $OMLX_API_KEY or "
                        "$OPENAI_API_KEY; oMLX requires one even on localhost)")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=600.0, help="per-request seconds")
    p.add_argument("--limit", type=int, default=None, help="cap number of scenes")
    p.add_argument("--out", default="runs/llm_transfer", help="report + response cache dir")
    p.add_argument("--dry-run", action="store_true",
                   help="print the first scene's prompts and exit (no server needed)")
    return p.parse_args(argv)


def load_split(data: str, split: str) -> list[tuple[str, dict]]:
    manifest = json.load(open(os.path.join(data, "manifest.json")))
    out = []
    for entry in manifest["scenes"]:
        if entry["split"] != split:
            continue
        sid = entry["scene_id"]
        out.append((sid, json.load(open(os.path.join(data, sid, "sample.json")))))
    return out


def cached_response(path: str, messages: list[dict], args) -> str:
    if os.path.exists(path):
        return json.load(open(path))["response"]
    text = chat_completion(args.base_url, args.model, messages,
                           temperature=args.temperature, max_tokens=args.max_tokens,
                           timeout=args.timeout, api_key=args.api_key)
    with open(path, "w") as fh:
        json.dump({"messages": messages, "response": text}, fh, indent=2)
    return text


_RUN_SEQ = itertools.count(1)


def cached_usage_response(path: str, messages: list[dict], args,
                          clip_path: str | None = None) -> tuple[str, dict]:
    """F33 usage-capturing path (video legs always; text legs when
    --reasoning-effort is set): captures usage + finish_reason +
    reasoning_content (tripwire, Amendment-E classifier, confab scan).
    Video caches store the CLIP PATH instead of its base64 payload.
    Each actually-issued request records a per-run sequence number
    (Amendment 1 co-report: prefill warm-up is position-in-session
    dependent, and server restarts correlate with condition boundaries;
    cache hits do not consume a number)."""
    from f33_video import chat_completion_usage
    if os.path.exists(path):
        d = json.load(open(path))
        return d["response"], d.get("info", {"usage": d.get("usage", {})})
    extra = dict({"reasoning_effort": args.reasoning_effort}
                 if args.reasoning_effort else {})
    if args.extra_body:
        extra.update(json.loads(args.extra_body))
    text, info = chat_completion_usage(
        args.base_url, args.model, messages, temperature=args.temperature,
        max_tokens=args.max_tokens, timeout=args.timeout, api_key=args.api_key,
        extra_body=extra or None)
    info["run_seq"] = next(_RUN_SEQ)
    if clip_path is not None:
        cached_msgs = [messages[0],
                       {"role": "user",
                        "content": [{"type": "input_video_path",
                                     "path": clip_path},
                                    messages[1]["content"][1]]}]
    else:
        cached_msgs = messages
    with open(path, "w") as fh:
        json.dump({"messages": cached_msgs, "response": text, "info": info,
                   "reasoning_effort": args.reasoning_effort,
                   "extra_body": args.extra_body}, fh, indent=2)
    return text, info


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    scenes = load_split(args.data, args.split)
    if not scenes:
        raise SystemExit(f"no scenes in split={args.split!r} under {args.data}")
    if args.limit:
        scenes = scenes[: args.limit]
    train_mean = train_mean_outcomes([s for _, s in load_split(args.data, "train")])

    def images_for(sid: str, sample: dict) -> list[str] | None:
        if args.condition != "vision":
            return None
        renders = os.path.join(args.data, sid, "renders")
        return [os.path.join(renders, f) for f in scene_view_files(sample, args.views)]

    custom_system = (open(args.system_prompt_file).read()
                     if args.system_prompt_file else None)

    def apply_system(msgs: list[dict]) -> list[dict]:
        """Disclosed condition change: swap the F11 system prompt (which
        COMMANDS step-by-step reasoning) for the file's contents. The
        frozen module stays untouched; parity legs never pass the flag."""
        if custom_system is not None:
            msgs = [{"role": "system", "content": custom_system}] + msgs[1:]
        return msgs

    if args.dry_run:
        sid, sample = scenes[0]
        for record in sample["behavior"]["probes"]:
            if args.condition.startswith("video_"):
                from f33_video import action_text, video_state_text
                leg = args.condition.split("_", 1)[1]
                print(f"--- {sid} / {record['probe']} ({args.condition}) ---")
                print("[user] <input_video: "
                      f"{record['probe']}_{leg}.mp4 (raw base64, video-first)> +")
                print(video_state_text(sample, leg))
                print(action_text(record) + "\n\nPredict the outcome fields.\n")
                continue
            msgs = build_messages(sample, record, args.condition, images_for(sid, sample))
            print(f"--- {sid} / {record['probe']} ({args.condition}) ---")
            for m in msgs:
                c = m["content"]
                if isinstance(c, list):
                    c = " + ".join(p["type"] if p["type"] != "text" else p["text"]
                                   for p in c)
                print(f"[{m['role']}]\n{c}\n")
        return

    is_video = args.condition.startswith("video_")
    if is_video:
        from f33_video import (
            build_video_messages,
            classify_response,
            clip_path_for,
            confab_time_claims,
            tripwire_ok,
        )
        clips_root = args.clips_root or (args.data.rstrip("/") + "_clips")
        clips_manifest = json.load(
            open(os.path.join(clips_root, "clips_manifest.json")))
        frames_by_clip = {(r["scene_id"], r["probe"], r["kind"]):
                          r["n_frames_decoded"] for r in clips_manifest["clips"]}
        leg_kind = args.condition.split("_", 1)[1]  # pred | percep

    resp_dir = os.path.join(args.out, "responses")
    os.makedirs(resp_dir, exist_ok=True)
    rows: list[tuple[str, str, dict | None, dict]] = []
    tripwire_failures: list[str] = []
    budget_starved: list[str] = []
    empty_stops: list[str] = []
    n_truncated = 0
    confab_content = confab_reasoning = 0
    completion_tokens: list[int] = []
    reasoning_lens: list[int] = []
    for i, (sid, sample) in enumerate(scenes):
        truths = probe_outcomes(sample)
        for record in sample["behavior"]["probes"]:
            kind = record["probe"]
            if kind not in PROBE_ORDER:
                continue
            cache = os.path.join(resp_dir, f"{sid}.{kind}.{args.condition}.json")
            use_usage = is_video or args.reasoning_effort or args.extra_body
            if use_usage:
                if is_video:
                    clip = clip_path_for(clips_root, sid, kind, leg_kind)
                    msgs = apply_system(
                        build_video_messages(sample, record, clip, leg_kind))
                    text, info = cached_usage_response(cache, msgs, args,
                                                       clip_path=clip)
                else:
                    from f33_video import classify_response, confab_time_claims
                    msgs = apply_system(
                        build_messages(sample, record, args.condition,
                                       images_for(sid, sample)))
                    text, info = cached_usage_response(cache, msgs, args)
                usage = info.get("usage", {})
                if is_video:
                    n_frames = frames_by_clip.get((sid, kind, leg_kind), 0)
                    if not tripwire_ok(usage.get("prompt_tokens", 0), n_frames):
                        # Amendment A.3: APPARATUS failure -- excluded and
                        # counted, never scored as model capability.
                        tripwire_failures.append(f"{sid}.{kind}")
                        print(f"[llm-transfer] {i + 1}/{len(scenes)} "
                              f"{sid}.{kind}: TRIPWIRE (prompt_tokens="
                              f"{usage.get('prompt_tokens')}, "
                              f"frames={n_frames})")
                        continue
                completion_tokens.append(usage.get("completion_tokens", 0))
                reasoning = info.get("reasoning_content", "")
                reasoning_lens.append(len(reasoning))
                cls = classify_response(text, info.get("finish_reason"))
                if cls in ("truncated", "budget_starved"):
                    n_truncated += 1  # feeds the frozen truncation gate
                if cls == "budget_starved":
                    # Amendment E: length + EMPTY content = the whole budget
                    # went to reasoning_content -- APPARATUS, not a model
                    # parse-failure; excluded and counted.
                    budget_starved.append(f"{sid}.{kind}")
                    print(f"[llm-transfer] {i + 1}/{len(scenes)} {sid}.{kind}: "
                          f"BUDGET-STARVED (reasoning {len(reasoning)} chars, "
                          f"content empty)")
                    continue
                if cls == "empty_stop":
                    # Amendment 1: stop + empty content = MODEL emission
                    # failure (answer stranded in reasoning_content). Falls
                    # through to scoring as a parse failure (baseline
                    # imputation); co-reported; NOT a truncation-gate event.
                    empty_stops.append(f"{sid}.{kind}")
                if confab_time_claims(text):
                    confab_content += 1
                if confab_time_claims(reasoning):
                    confab_reasoning += 1
            else:
                cls = None
                text = cached_response(cache,
                                       build_messages(sample, record,
                                                      args.condition,
                                                      images_for(sid, sample)),
                                       args)
            pred = extract_prediction(text)
            rows.append((sid, kind, pred, truths[kind]))
            tag = ("EMPTY-STOP" if cls == "empty_stop"
                   else "ok" if pred else "PARSE-FAIL")
            print(f"[llm-transfer] {i + 1}/{len(scenes)} {sid}.{kind}: {tag}")

    report = {
        "data": args.data, "split": args.split, "condition": args.condition,
        "model": args.model, "n_scenes": len(scenes),
        **score_predictions(rows, train_mean),
    }
    # Amendment 1: baseline imputation pulls failed cells TOWARD baseline
    # (flatters below-baseline conditions, penalizes above-baseline ones),
    # so differing failure rates bias cross-condition comparisons. Report
    # the complete-case figures as a standing pair with the imputed ones.
    complete_rows = [r for r in rows if r[2] is not None]
    cc = score_predictions(complete_rows, train_mean)
    report["complete_case"] = {
        "n_rows": cc["n_rows"], "n_excluded": len(rows) - len(complete_rows),
        "mse": cc["mse"], "baseline_mse": cc["baseline_mse"],
        "gain_over_mean": cc["gain_over_mean"], "per_probe": cc["per_probe"],
    }
    # Amendment 1: empty_stop co-report (model emission failure; scored via
    # imputation above, never a truncation-gate event).
    report["n_empty_stop"] = len(empty_stops)
    report["empty_stop"] = empty_stops
    if is_video:
        n_attempted = len(rows) + len(tripwire_failures) + len(budget_starved)
        report["video"] = {
            "clips_root": clips_root, "leg_kind": leg_kind,
            "n_requests": n_attempted,
            "n_tripwire_failures": len(tripwire_failures),
            "tripwire_failed": tripwire_failures,
            "tripwire_rate": (len(tripwire_failures) / n_attempted
                              if n_attempted else 0.0),
            "n_truncated": n_truncated,  # frozen f33 gate reads this
            "n_budget_starved": len(budget_starved),
            "budget_starved": budget_starved,
            # Amendment C.4 + E: confab scanned in BOTH channels, labeled --
            # fabrications live in reasoning_content, not content.
            "confab_content_rate": (confab_content / len(rows)
                                    if rows else 0.0),
            "confab_reasoning_rate": (confab_reasoning / len(rows)
                                      if rows else 0.0),
            "mean_completion_tokens": (sum(completion_tokens)
                                       / len(completion_tokens)
                                       if completion_tokens else 0.0),
            "mean_reasoning_chars": (sum(reasoning_lens) / len(reasoning_lens)
                                     if reasoning_lens else 0.0),
        }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "transfer_report.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)

    g = report["gain_over_mean"]
    print(f"\n[llm-transfer] condition={args.condition}  scenes={len(scenes)}  "
          f"rows={report['n_rows']}  parse_failures={report['n_parse_failures']}")
    print(f"[llm-transfer] normalized MSE={report['mse']:.4f}  "
          f"predict-mean baseline={report['baseline_mse']:.4f}  "
          f"gain_over_mean={g:.2f}x" + (" (beats baseline)" if g and g > 1 else ""))
    if report["push_toppled_brier"] is not None:
        print(f"[llm-transfer] push.toppled Brier={report['push_toppled_brier']:.4f}")
    for kind, v in report["per_probe"].items():
        print(f"  {kind:5s} mse={v['mse']:.4f}  baseline={v['baseline_mse']:.4f}")
    print(f"[llm-transfer] wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
