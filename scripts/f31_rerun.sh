#!/usr/bin/env bash
# F31a Amendment 1 rerun on the repaired-ramp worlds (docs/TRAJECTORY_SUPERVISION.md §10).
# Stage A: loud-joint seeds 0-4 -> measure -> C1 co-report.
# Stage B: loud-traj seeds 0-4, ctrl-joint and ctrl-traj seeds 0-2 -> measure ->
#          frozen verdicts + escaped-only co-report.
# Needs data/pm_f31fix_{loud,ctrl} (generate_mujoco with the F27b flags plus
# --keep-trajectory, on code that includes the PR #45 ramp fix).
set -o pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROOT=runs/f31fix
TRAJ="--trajectory-weight 7.0 --trajectory-frames 16 --trajectory-probes drop,tilt --trajectory-pos-only"
mkdir -p $ROOT

run() {  # name world seed [extra flags]
  local name=$1 world=$2 seed=$3; shift 3
  local out=$ROOT/${name}_s${seed}
  if [ -f "$out/metrics.json" ]; then echo "$name s$seed exists, skipped"; return; fi
  local t0; t0=$(date +%s)
  $PY -m pseudomarble.models.train_torch --data data/pm_f31fix_$world \
    --epochs 50 --lr 2e-4 --batch-size 16 --seed "$seed" --out "$out" "$@" > "$out.log" 2>&1 \
    || { echo "$name s$seed FAILED"; tail -5 "$out.log"; return 1; }
  echo "$name s$seed done in $(( $(date +%s) - t0 ))s: $(grep -o 'heldout_gain [0-9.]*' "$out.log" | tail -1)"
}
measure() {  # arm world
  $PY scripts/f31_measure.py --data data/pm_f31fix_$2 --runs $ROOT/${1}_s[0-9] \
    --out $ROOT/$1.json | tail -7
}

# Step mode: run one unit per invocation, so each call fits a short
# background-job limit. Units: `run <name> <world> <seed> [traj]`,
# `stage-a-report`, `stage-b-report`.
if [ "${1:-}" = "run" ]; then
  if [ "${5:-}" = "traj" ]; then run "$2" "$3" "$4" $TRAJ; else run "$2" "$3" "$4"; fi
  exit $?
elif [ "${1:-}" = "stage-a-report" ]; then
  measure loud_joint loud
  $PY scripts/f31_coreport.py --root $ROOT --broken runs/f31 --stage-a
  exit $?
elif [ "${1:-}" = "stage-b-report" ]; then
  measure loud_joint loud; measure loud_traj loud; measure ctrl_joint ctrl; measure ctrl_traj ctrl
  $PY scripts/f31_verdicts.py --root $ROOT
  $PY scripts/f31_coreport.py --root $ROOT --broken runs/f31
  exit $?
fi

echo "== stage A =="
for s in 0 1 2 3 4; do run loud_joint loud $s; done
measure loud_joint loud
$PY scripts/f31_coreport.py --root $ROOT --broken runs/f31 --stage-a
echo "STAGE A COMPLETE"

echo "== stage B =="
for s in 0 1 2 3 4; do run loud_traj loud $s $TRAJ; done
for s in 0 1 2; do run ctrl_joint ctrl $s; done
for s in 0 1 2; do run ctrl_traj ctrl $s $TRAJ; done
measure loud_traj loud
measure ctrl_joint ctrl
measure ctrl_traj ctrl
$PY scripts/f31_verdicts.py --root $ROOT
$PY scripts/f31_coreport.py --root $ROOT --broken runs/f31
echo "STAGE B COMPLETE"
