#!/usr/bin/env bash
# Convenience wrapper for the PRIMARY (MuJoCo) data-generation pipeline.
#
# Usage:
#   scripts/run_datagen_mujoco.sh [OUTPUT_DIR] [NUM_SCENES] [VIEWS] [RESOLUTION]
#
# Requires the project installed with the mujoco extra on Apple silicon:
#   pip install -e ".[mujoco]"
set -euo pipefail

OUTPUT="${1:-data/pseudo_marble}"
NUM_SCENES="${2:-16}"
VIEWS="${3:-16}"
RESOLUTION="${4:-256}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if ! python -c "import mujoco" >/dev/null 2>&1; then
  echo "error: the 'mujoco' package is not importable." >&2
  echo "       pip install -e \".[mujoco]\"   (arm64 wheel on macOS)" >&2
  exit 1
fi

echo "[pseudo-marble:mujoco] generating ${NUM_SCENES} scenes -> ${OUTPUT}"
python -m pseudomarble.data.generate_mujoco \
  --output "${OUTPUT}" \
  --num-scenes "${NUM_SCENES}" \
  --views "${VIEWS}" \
  --resolution "${RESOLUTION}"
