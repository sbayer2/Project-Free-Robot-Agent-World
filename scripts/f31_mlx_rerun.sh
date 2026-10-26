#!/usr/bin/env bash
# MLX replication of the F31a repaired-ramp arms (docs/F31A_MLX_REPLICATION.md),
# mirroring scripts/f31_rerun.sh on the canonical MLX trainer: same worlds,
# frozen F27b recipe and F31 head flags; loud joint/traj seeds 0-4, ctrl
# joint/traj seeds 0-2. Mac only. The report step measures with
# scripts/f31_measure_mlx.py and grades with the frozen scripts/f31_verdicts.py,
# plus Amendment 2's co-reports (a) and (b) from scripts/f31_coreport.py.
# C1 is not reported: it needs broken-world runs on the same backend.
# Needs data/pm_f31fix_{loud,ctrl}. ROOT=<dir> redirects the runs (the
# bias-correction diagnostic used ROOT=runs/f31fix_mlx_bc with a local patch).
#
#   bash scripts/f31_mlx_rerun.sh run <arm> <world> <seed> [traj]
#   bash scripts/f31_mlx_rerun.sh report
#   bash scripts/f31_mlx_rerun.sh             # everything, then the report
set -o pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROOT=${ROOT:-runs/f31fix_mlx}
TRAJ="--trajectory-weight 7.0 --trajectory-frames 16 --trajectory-probes drop,tilt --trajectory-pos-only"
mkdir -p "$ROOT"

run() {  # name world seed [extra flags]
  local name=$1 world=$2 seed=$3; shift 3
  local out=$ROOT/${name}_s${seed}
  if [ -f "$out/metrics.json" ]; then echo "$name s$seed exists, skipped"; return; fi
  local t0; t0=$(date +%s)
  $PY -m pseudomarble.models.train --data data/pm_f31fix_$world \
    --epochs 50 --lr 2e-4 --batch-size 16 --seed "$seed" --out "$out" "$@" > "$out.log" 2>&1 \
    || { echo "$name s$seed FAILED"; tail -5 "$out.log"; return 1; }
  echo "$name s$seed done in $(( $(date +%s) - t0 ))s"
}
measure() {  # arm world
  $PY scripts/f31_measure_mlx.py --data data/pm_f31fix_$2 --runs $ROOT/${1}_s[0-9] \
    --out $ROOT/$1.json | tail -7
}
report() {
  measure loud_joint loud; measure loud_traj loud
  if ls -d $ROOT/ctrl_joint_s[0-9] > /dev/null 2>&1; then
    measure ctrl_joint ctrl; measure ctrl_traj ctrl
    $PY scripts/f31_verdicts.py --root $ROOT
    $PY - "$ROOT" <<'EOF'
import sys
sys.path.insert(0, "scripts")
import f31_coreport as C
reps = {a: C.load(sys.argv[1], a) for a in C.V.ARMS}
la = C.leakage_adjusted(reps)
print(f"[f31-mlx-A2a] D_L {la['delta_L']:+.4f} (loud {la['loud_diff']:+.4f} - ctrl "
      f"{la['ctrl_diff']:+.4f}), SE {la['se']:.4f}, t {la['t']:+.2f}")
for a, runs in C.heldout_readout(reps).items():
    print(f"[f31-mlx-A2b] held-out friction R2 {a}: "
          + " ".join(f"{v:+.3f}" for v in runs.values()))
EOF
  fi
  $PY scripts/f31_head_diag_mlx.py --data data/pm_f31fix_loud \
    --runs $ROOT/loud_traj_s[0-9] --out $ROOT/head_diag_loud_traj.json
}

if [ "${1:-}" = "run" ]; then
  if [ "${5:-}" = "traj" ]; then run "$2" "$3" "$4" $TRAJ; else run "$2" "$3" "$4"; fi
  exit $?
elif [ "${1:-}" = "report" ]; then
  report; exit $?
fi

for s in 0 1 2 3 4; do run loud_joint loud $s; run loud_traj loud $s $TRAJ; done
for s in 0 1 2; do run ctrl_joint ctrl $s; run ctrl_traj ctrl $s $TRAJ; done
report
