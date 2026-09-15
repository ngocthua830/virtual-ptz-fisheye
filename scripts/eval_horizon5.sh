#!/usr/bin/env bash
# Corrected horizon experiment on ALL 5 full-budget seeds + baselines (review #1,#2,#4).
# One 300-step deterministic rollout per (model, clip); every horizon derived from it with
# a FIXED full-clip cohort. Output: results/full128/horizon5/<tag>.json
#
# Usage: GPUS="3 5" bash scripts/eval_horizon5.sh
set -u
ROOT="/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye"
SIB="/home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect"
DET="yolo26n_obb_topview_person_290526.pt"; IMG="ptz-control-obb"
OUT="results/full128/horizon5"; GPUS=(${GPUS:-3 5})
cd "$ROOT" || exit 1; mkdir -p "$OUT"

run() {  # gpu out policies extra...
  local gpu="$1" out="$2" pol="$3"; shift 3
  docker run --rm --gpus "device=$gpu" \
    -v "$ROOT":/app -w /app -v "$SIB":/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app \
    "$IMG" python -u -m evaluation.horizon_curve --data_path data/test \
      --detector_weights "$DET" --state_dim 29 --gate_deg 15 \
      --policies "$pol" "$@" --out "$out" \
    > "${out%.json}.log" 2>&1
}

JOBS=()
for d in results/full128/fisheye_dual_*; do
  [ -d "$d" ] || continue
  tag=$(basename "$d" | sed -E 's/^fisheye_dual_[0-9]{8}_[0-9]{6}_?//')
  [ -f "$d/tracker_best.pt" ] || continue
  JOBS+=("$OUT/$tag.json|rl|--tracker $d/tracker_best.pt --explorer $d/explorer_best.pt")
done
JOBS+=("$OUT/baselines.json|random,sweep,coordsweep,greedy,heuristic|--random_seeds 0,1,2,3,4")

echo "== ${#JOBS[@]} corrected-horizon jobs | GPUS=${GPUS[*]} =="
declare -A PID_GPU; free=("${GPUS[@]}"); i=0; N=${#JOBS[@]}
while [ $i -lt $N ] || [ ${#PID_GPU[@]} -gt 0 ]; do
  while [ $i -lt $N ] && [ ${#free[@]} -gt 0 ]; do
    IFS='|' read -r out pol extra <<< "${JOBS[$i]}"
    gpu="${free[0]}"; free=("${free[@]:1}")
    echo "[hz5] GPU$gpu -> $(basename "$out")"
    ( run "$gpu" "$out" "$pol" $extra ) & PID_GPU[$!]="$gpu"; i=$((i+1))
  done
  if [ ${#PID_GPU[@]} -gt 0 ]; then
    wait -n
    for pid in "${!PID_GPU[@]}"; do
      kill -0 "$pid" 2>/dev/null || { free+=("${PID_GPU[$pid]}"); unset 'PID_GPU[$pid]'; }
    done
  fi
done
echo "== CORRECTED HORIZON DONE -> $OUT =="
