#!/bin/zsh
# F29 disjoint retrain launcher (docs/HELD_OUT_TRANSFER.md section 4).
#
# 24 disjoint pairs required by the prereg = 4 arms x 3 pairs x 2 heads.
# We do 8 SYMLINKS + 16 REAL RETRAINS (session-approved 2026-08-11 to avoid
# reproducing runs/f27b's existing artifacts):
#
#   pair0 -> SYMLINK to runs/f27b/<arm>_{renderonly,behavonly}. Those were
#            trained at seed 0, lr 2e-4, 50 epochs, image_size 128, same
#            data, same head-weight zeros -- reproducing them bit-for-bit
#            here would burn ~12 min for zero information gain.
#   pair1 -> TRAIN fresh at seed 1 (matches runs/f27b/<arm>_s1 init).
#   pair2 -> TRAIN fresh at seed 2 (matches runs/f27b/<arm>_s2 init).
#
# Frozen recipe (byte-for-byte from runs/f27b/launch.sh's control lines,
# seed varied): --lr 2e-4 --epochs 50 --image-size 128
#   render head: --behavior-weight 0 --essence-weight 0
#   behav  head: --render-weight 0
#
# Layout (each disjoint pair's two heads live in subdirs so train.py's
# default {model.safetensors, metrics.json} files land unaliased):
#   runs/f29_disjoint/<arm>_pair<k>/render/{model.safetensors,metrics.json}
#   runs/f29_disjoint/<arm>_pair<k>/behav/ {model.safetensors,metrics.json}
#
# Idempotent: skips any (arm, pair, head) whose metrics.json already exists
# (train.py writes it last, so a partial-crash checkpoint doesn't fool the
# check). Rerun on a partial train picks up where it stopped.
#
# Wall clock: ~24 min at f27b timings (16 x ~87s) on the M5 Pro, no GPU
# contention with anything else (all serial, single MLX process).
#
# Usage:
#     zsh scripts/f29_retrain.sh                    # run
#     zsh scripts/f29_retrain.sh --dry-run          # list actions, do nothing

set -e -o pipefail   # pipefail: a failing train.py must not be hidden by 'tail -1'
cd /Users/sbm4_mac/Project-Free-Robot-Agent-World
PY=.venv/bin/python
ROOT=runs/f29_disjoint
F27B=runs/f27b
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

# Sanity: refuse to run if the F27b source artifacts we symlink aren't there.
for tag in ctrl base g2 loud; do
  for src in ${tag}_renderonly ${tag}_behavonly; do
    if [[ ! -f "$F27B/$src/model.safetensors" ]]; then
      echo "ERROR: $F27B/$src/model.safetensors missing -- rerun F27b first."
      exit 2
    fi
  done
done

# ---- 8 pair0 symlinks (F27b renderonly/behavonly, seed 0) -------------- #
for tag in ctrl base g2 loud; do
  mkdir -p "$ROOT/${tag}_pair0"
  for pair in render:renderonly behav:behavonly; do
    kind=${pair%%:*}; src=${pair##*:}
    link="$ROOT/${tag}_pair0/$kind"
    if [[ -e "$link" ]]; then
      [[ $DRY -eq 1 ]] && echo "SKIP  symlink $link (exists)"
      continue
    fi
    target="../../f27b/${tag}_${src}"
    if [[ $DRY -eq 1 ]]; then
      echo "PLAN  ln -s $target $link"
    else
      ln -sf "$target" "$link"
      echo "LINK  $link -> $target"
    fi
  done
done

# ---- 16 fresh trains (pair1, pair2 at seeds 1, 2) ---------------------- #
n_run=0; n_skip=0
for tag in ctrl base g2 loud; do
  for seed in 1 2; do
    for pair in render:"--behavior-weight 0 --essence-weight 0" behav:"--render-weight 0"; do
      kind=${pair%%:*}; weights=${pair##*:}
      out="$ROOT/${tag}_pair${seed}/${kind}"
      if [[ -f "$out/metrics.json" ]]; then
        echo "SKIP  train $out (metrics.json exists)"
        (( ++n_skip ))
        continue
      fi
      if [[ $DRY -eq 1 ]]; then
        echo "PLAN  train $tag pair$seed $kind (seed=$seed, $weights)"
        (( ++n_run ))
        continue
      fi
      echo "=== TRAIN $tag pair$seed $kind (seed=$seed, lr 2e-4) ==="
      # ${=weights} forces zsh word-splitting on IFS so
      # "--behavior-weight 0 --essence-weight 0" reaches train.py as four
      # args, not one glued string. Bash-style unquoted $weights does NOT
      # split in zsh -- burned 24 min on that once, never again.
      $PY -m pseudomarble.models.train --data "data/pm_f27_$tag" \
        --out "$out" --seed "$seed" --lr 2e-4 --epochs 50 --image-size 128 \
        ${=weights} 2>&1 | tail -1
      (( ++n_run ))
    done
  done
done

if [[ $DRY -eq 1 ]]; then
  echo "=== DRY RUN: $n_run to train, $n_skip already done ==="
else
  echo "=== F29 DISJOINT RETRAIN DONE: $n_run trained, $n_skip skipped ==="
fi
