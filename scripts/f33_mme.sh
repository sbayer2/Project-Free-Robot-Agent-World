#!/bin/zsh
# F33 Phase 0: the public MME benchmark against the local Qwen3.8-27B
# endpoint (docs/VIDEO_BASELINE.md section 5, Phase 0 -- context anchor,
# no verdict). USER-GATED: do not run without the in-session go; the
# provenance protocol applies (record /props build hash + uptime before
# and after; confirm no restart with the user).
#
# Harness: lmms-eval in its own venv (~/venvs/lmms-eval -- kept OUT of the
# project venv so the 302-test suite's MLX/torch pins are never perturbed).
# Model driver: lmms-eval's OpenAI-compatible API model pointed at the
# llama-server. MME downloads from HF (lmms-lab/MME) on first run.
#
# Budget: ~2.4k image questions, prompt-bound; at the measured 250-290
# tok/s prompt rate expect 3-6 h. Cache/output under runs/f33/mme.
#
# Usage:
#     zsh scripts/f33_mme.sh              # run (after user go)
#     zsh scripts/f33_mme.sh --check      # verify env + task availability only

set -e -o pipefail
cd /Users/sbm4_mac/Project-Free-Robot-Agent-World
VENV="$HOME/venvs/lmms-eval"
PY="$VENV/bin/python"
OUT="runs/f33/mme"
BASE_URL="${F33_BASE_URL:-http://127.0.0.1:8080/v1}"
MODEL="${F33_MODEL:-ggml-org/Qwen3.8-27B-GGUF:Q4_K_M}"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: $VENV missing -- run the F33 MME setup first."
  exit 2
fi

if [[ "${1:-}" == "--check" ]]; then
  "$PY" - <<'PYEOF'
import os
import lmms_eval
from lmms_eval.models import get_model
print("lmms-eval", __import__("importlib.metadata", fromlist=["version"]).version("lmms_eval"))
task_root = os.path.join(os.path.dirname(lmms_eval.__file__), "tasks")
assert os.path.isdir(os.path.join(task_root, "mme")), "mme task dir missing"
print("mme task available")
cls = get_model("openai")
print("openai driver OK:", cls.__name__, "->", cls.__mro__[1].__name__)
PYEOF
  exit 0
fi

mkdir -p "$OUT"
# Record provenance before the run (Amendment/apparatus protocol).
curl -s -m 5 "${BASE_URL%/v1}/props" | head -c 400 > "$OUT/props_before.json" || true
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$OUT/started_at.txt"

# model_args VERIFIED against installed lmms-eval 0.7.2: driver is
# "openai" (subclass of OpenAICompatible; model_version/base_url/api_key/
# timeout/num_concurrent are real init params). Two defaults corrected:
# timeout=10s would kill every 27B response (~250 tok/s prompt phase),
# and num_concurrent=32 would swamp the 4-slot server -- 2 is etiquette.
OPENAI_API_KEY="${OPENAI_API_KEY:-none}" \
"$PY" -m lmms_eval \
  --model openai \
  --model_args "model_version=$MODEL,base_url=$BASE_URL,api_key=none,timeout=300,num_concurrent=${F33_MME_CONCURRENCY:-2},batch_size=1" \
  --tasks mme \
  --batch_size 1 \
  --log_samples \
  --output_path "$OUT" 2>&1 | tee "$OUT/mme_run.log"

curl -s -m 5 "${BASE_URL%/v1}/props" | head -c 400 > "$OUT/props_after.json" || true

# lmms-eval catches evaluation errors and still exits 0 (observed live:
# the 0.7.2 wheel's missing extensionless include files crashed the task
# indexer, the CLI logged ERROR, exited 0, and the pipeline printed
# success). Detect the failure from the log instead of the exit code.
if grep -q "Error during evaluation" "$OUT/mme_run.log"; then
  echo "MME FAILED (see $OUT/mme_run.log)"
  exit 1
fi
echo "MME DONE -> $OUT (scores go into the F33 report's meta.mme)"
