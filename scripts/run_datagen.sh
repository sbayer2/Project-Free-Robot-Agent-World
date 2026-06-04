#!/usr/bin/env bash
# Convenience wrapper for the Blender data-generation pipeline.
#
# Usage:
#   scripts/run_datagen.sh [OUTPUT_DIR] [NUM_SCENES] [VIEWS] [RESOLUTION]
#
# Requires Blender 4.x on PATH (it brings its own Python with bpy). Everything
# after `--` is forwarded to generate_blender.py.
set -euo pipefail

OUTPUT="${1:-data/pseudo_marble}"
NUM_SCENES="${2:-16}"
VIEWS="${3:-16}"
RESOLUTION="${4:-256}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${REPO_ROOT}/src/pseudomarble/data/generate_blender.py"

if ! command -v blender >/dev/null 2>&1; then
  echo "error: 'blender' not found on PATH. Install Blender 4.x." >&2
  echo "       https://www.blender.org/download/" >&2
  exit 1
fi

echo "[pseudo-marble] generating ${NUM_SCENES} scenes -> ${OUTPUT}"
blender --background --python "${SCRIPT}" -- \
  --output "${OUTPUT}" \
  --num-scenes "${NUM_SCENES}" \
  --views "${VIEWS}" \
  --resolution "${RESOLUTION}"
