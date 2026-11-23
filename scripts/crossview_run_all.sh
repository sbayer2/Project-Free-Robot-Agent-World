#!/bin/bash
# Cross-view pooling experiment: frozen preregistration be391d0
# (docs/CROSSVIEW_POOLING.md §3, §9). A transcription of §9's commands:
# 5 arms x (20 loud seeds 0-19 + 3 ctrl seeds 0-2), 50 epochs, lr 2e-4, B16,
# then crossview_measure (CPU encode, frozen; --d1 for P0 loud) and
# crossview_verdicts. Loud runs interleave arms by seed so a stop leaves the
# arms balanced; a finished run (metrics.json present) is skipped on restart.
set -u
cd "$HOME/Project-Free-Robot-Agent-World"
WT="$HOME/.cache/pfraw-wt/wt-xv"
PY="$HOME/Project-Free-Robot-Agent-World/.venv/bin/python"
export PYTHONPATH="$WT/src"
OUT=runs/xv
CODE=$(git -C "$WT" rev-parse HEAD)
CLEAN=$([ -z "$(git -C "$WT" status --porcelain)" ] && echo 1 || echo 0)
echo "code $CODE clean=$CLEAN started $(date -u +%FT%TZ)" >> $OUT/PROVENANCE.txt
"$PY" -c "import pseudomarble; print('pseudomarble from', pseudomarble.__file__)"

pool() { case $1 in P0) echo mean;; P1) echo stats;; P1c) echo stats_ctrl;;
                     P2) echo attn;; P2c) echo attn_ctrl;; esac; }
ARMS="P0 P1 P1c P2 P2c"

train() {  # arm world seed
  local d=$OUT/$1_$2_s$3
  if [ -f "$d/metrics.json" ]; then echo "skip $d"; return; fi
  echo "[$(date -u +%FT%TZ)] train $d ($(pool $1))"
  "$PY" -m pseudomarble.models.train --data data/pm_f31fix_$2_cam --epochs 50 \
    --lr 2e-4 --batch-size 16 --seed $3 --view-pool "$(pool $1)" --out "$d" \
    > "$OUT/logs/$1_$2_s$3.log" 2>&1 || echo "FAILED $d"
}

for s in $(seq 0 19); do for a in $ARMS; do train $a loud $s; done; done
for s in 0 1 2; do for a in $ARMS; do train $a ctrl $s; done; done
echo "training finished $(date -u +%FT%TZ)" >> $OUT/PROVENANCE.txt

for a in $ARMS; do for w in loud ctrl; do
  d1=""; [ "$a" = P0 ] && [ "$w" = loud ] && d1="--d1"
  echo "[$(date -u +%FT%TZ)] measure ${a}_${w}"
  "$PY" "$WT/scripts/crossview_measure.py" --data data/pm_f31fix_${w}_cam \
    --runs $OUT/${a}_${w}_s* $d1 --out $OUT/${a}_${w}.json \
    > "$OUT/logs/measure_${a}_${w}.log" 2>&1 || echo "MEASURE FAILED ${a}_${w}"
done; done
echo "measured $(date -u +%FT%TZ)" >> $OUT/PROVENANCE.txt

"$PY" "$WT/scripts/crossview_verdicts.py" --root $OUT > $OUT/verdicts.log 2>&1 \
  || echo "VERDICTS FAILED"
echo "verdicts $(date -u +%FT%TZ)" >> $OUT/PROVENANCE.txt
echo "done $(date -u +%FT%TZ)"
