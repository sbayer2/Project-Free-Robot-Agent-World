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
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pseudomarble.llm_transfer import (  # noqa: E402
    build_messages,
    chat_completion,
    extract_prediction,
    probe_outcomes,
    score_predictions,
    train_mean_outcomes,
)
from pseudomarble.probes import PROBE_ORDER  # noqa: E402


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM world-model transfer eval")
    p.add_argument("--data", required=True, help="dataset root (schema v2)")
    p.add_argument("--split", default="test", help="split to evaluate (held-out=test)")
    p.add_argument("--condition", default="essence", choices=("essence", "appearance"))
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


def load_split(data: str, split: str) -> List[Tuple[str, Dict]]:
    manifest = json.load(open(os.path.join(data, "manifest.json")))
    out = []
    for entry in manifest["scenes"]:
        if entry["split"] != split:
            continue
        sid = entry["scene_id"]
        out.append((sid, json.load(open(os.path.join(data, sid, "sample.json")))))
    return out


def cached_response(path: str, messages: List[Dict], args) -> str:
    if os.path.exists(path):
        return json.load(open(path))["response"]
    text = chat_completion(args.base_url, args.model, messages,
                           temperature=args.temperature, max_tokens=args.max_tokens,
                           timeout=args.timeout, api_key=args.api_key)
    with open(path, "w") as fh:
        json.dump({"messages": messages, "response": text}, fh, indent=2)
    return text


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    scenes = load_split(args.data, args.split)
    if not scenes:
        raise SystemExit(f"no scenes in split={args.split!r} under {args.data}")
    if args.limit:
        scenes = scenes[: args.limit]
    train_mean = train_mean_outcomes([s for _, s in load_split(args.data, "train")])

    if args.dry_run:
        sid, sample = scenes[0]
        for record in sample["behavior"]["probes"]:
            msgs = build_messages(sample, record, args.condition)
            print(f"--- {sid} / {record['probe']} ({args.condition}) ---")
            for m in msgs:
                print(f"[{m['role']}]\n{m['content']}\n")
        return

    resp_dir = os.path.join(args.out, "responses")
    os.makedirs(resp_dir, exist_ok=True)
    rows: List[Tuple[str, str, Optional[Dict], Dict]] = []
    for i, (sid, sample) in enumerate(scenes):
        truths = probe_outcomes(sample)
        for record in sample["behavior"]["probes"]:
            kind = record["probe"]
            if kind not in PROBE_ORDER:
                continue
            cache = os.path.join(resp_dir, f"{sid}.{kind}.{args.condition}.json")
            text = cached_response(cache, build_messages(sample, record, args.condition),
                                   args)
            pred = extract_prediction(text)
            rows.append((sid, kind, pred, truths[kind]))
            tag = "ok" if pred else "PARSE-FAIL"
            print(f"[llm-transfer] {i + 1}/{len(scenes)} {sid}.{kind}: {tag}")

    report = {
        "data": args.data, "split": args.split, "condition": args.condition,
        "model": args.model, "n_scenes": len(scenes),
        **score_predictions(rows, train_mean),
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
