#!/usr/bin/env bash
# Deterministic clip-balanced eval of the single-agent (shared-policy) models. Each is
# evaluated as RLPolicy(single, single): the SAME weights drive both cameras. Same
# protocol as eval_ablation/eval_full128 (steps 300, episodes 3, forced clip per ep).
# Output: results/single/eval/<tag>.json.
#
# Usage: GPUS="0 1" bash scripts/eval_single.sh
set -u
ROOT="/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye"
SIB="/home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect"
DET="yolo26n_obb_topview_person_290526.pt"; IMG="ptz-control-obb"
EVALDIR="results/single/eval"; GPUS=(${GPUS:-0 1})
cd "$ROOT" || exit 1; mkdir -p "$EVALDIR"

run_metrics() {  # gpu out model
  local gpu="$1" out="$2" model="$3"
  docker run --rm --gpus "device=$gpu" \
    -v "$ROOT":/app -w /app -v "$SIB":/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app \
    "$IMG" python -u evaluation/metrics.py --data_path data/test \
      --detector_weights "$DET" --steps 300 --episodes 3 \
      --seed_list 0 --state_dim 29 --gate_deg 15 \
      --policies rl --tracker "$model" --explorer "$model" --out "$out" \
    > "${out%.json}.log" 2>&1
}

JOBS=()
for d in results/single/fisheye_single_*; do
  [ -d "$d" ] || continue
  tag=$(basename "$d" | sed -E 's/^fisheye_single_[0-9]{8}_[0-9]{6}_?//')
  [ -f "$d/single_best.pt" ] || { echo "[skip] $d"; continue; }
  JOBS+=("$EVALDIR/$tag.json|$d/single_best.pt")
done

echo "== ${#JOBS[@]} single-agent eval jobs | GPUS=${GPUS[*]} =="
declare -A PID_GPU; free=("${GPUS[@]}"); i=0; N=${#JOBS[@]}
while [ $i -lt $N ] || [ ${#PID_GPU[@]} -gt 0 ]; do
  while [ $i -lt $N ] && [ ${#free[@]} -gt 0 ]; do
    IFS='|' read -r out model <<< "${JOBS[$i]}"
    gpu="${free[0]}"; free=("${free[@]:1}")
    echo "[eval] GPU$gpu -> $(basename "$out")"
    ( run_metrics "$gpu" "$out" "$model" ) & PID_GPU[$!]="$gpu"; i=$((i+1))
  done
  if [ ${#PID_GPU[@]} -gt 0 ]; then
    wait -n
    for pid in "${!PID_GPU[@]}"; do
      kill -0 "$pid" 2>/dev/null || { free+=("${PID_GPU[$pid]}"); unset 'PID_GPU[$pid]'; }
    done
  fi
done
echo "== SINGLE-AGENT EVAL DONE -> $EVALDIR =="
