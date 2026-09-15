#!/usr/bin/env bash
# Train the single-agent (shared-policy) baseline: ONE MP-DQN drives both virtual cameras.
# Matched to the dual ablation budget (episodes=80, max_steps=100), 3 seeds (42/43/44) so
# it pairs directly with the dual "Full model (80x100)" in Table 2. GPU-parallel.
#
# Usage: GPUS="0 1 4" bash scripts/run_single.sh
set -u
ROOT="/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye"
SIB="/home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect"
DET="yolo26n_obb_topview_person_290526.pt"; IMG="ptz-control-obb"
OUT="results/single"; EPISODES=80; STEPS=100; GPUS=(${GPUS:-0 1 4})
cd "$ROOT" || exit 1; mkdir -p "$OUT/logs"

SEEDS=(42 43 44)
train() {  # gpu seed
  local gpu="$1" seed="$2" tag="single_s${2}"
  docker run --rm --gpus "device=$gpu" \
    -v "$ROOT":/app -w /app -v "$SIB":/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app \
    "$IMG" python -u main_train_single.py \
      --data_path data/train --detector_weights "$DET" \
      --episodes "$EPISODES" --max_steps "$STEPS" --seed "$seed" \
      --tag "$tag" --results_dir "$OUT" \
    > "$OUT/logs/${tag}.log" 2>&1
}

echo "== single-agent training: ${#SEEDS[@]} seeds | ${EPISODES}x${STEPS} | GPUS=${GPUS[*]} =="
declare -A PID_GPU; free=("${GPUS[@]}"); i=0; N=${#SEEDS[@]}
while [ $i -lt $N ] || [ ${#PID_GPU[@]} -gt 0 ]; do
  while [ $i -lt $N ] && [ ${#free[@]} -gt 0 ]; do
    gpu="${free[0]}"; free=("${free[@]:1}")
    echo "[train] GPU$gpu -> single_s${SEEDS[$i]}"
    ( train "$gpu" "${SEEDS[$i]}" ) & PID_GPU[$!]="$gpu"; i=$((i+1))
  done
  if [ ${#PID_GPU[@]} -gt 0 ]; then
    wait -n
    for pid in "${!PID_GPU[@]}"; do
      kill -0 "$pid" 2>/dev/null || { free+=("${PID_GPU[$pid]}"); unset 'PID_GPU[$pid]'; }
    done
  fi
done
echo "== SINGLE-AGENT TRAINING DONE -> $OUT =="
