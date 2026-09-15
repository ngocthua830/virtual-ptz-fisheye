#!/usr/bin/env bash
# Full-BUDGET retrain of the 5 full-model seeds at the paper's original
# 100 episodes x 128 steps, to give RL its fairest shot vs the baselines.
# Kept separate from the 80x100 ablation models (results/ablation/) so the
# ablation table stays internally consistent.
#
# Usage: GPUS="3 5" bash scripts/run_full128.sh
set -u
ROOT="/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye"
SIB="/home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect"
DET="yolo26n_obb_topview_person_290526.pt"; IMG="ptz-control-obb"
OUT="results/full128"; GPUS=(${GPUS:-3 5})
EPISODES=100; STEPS=128
cd "$ROOT" || exit 1; mkdir -p "$OUT/logs"

JOBS=(full_s42:42 full_s43:43 full_s44:44 full_s45:45 full_s46:46)

launch() {
  local gpu="$1" tag="$2" seed="$3"
  echo "[launch] GPU$gpu $tag seed=$seed (100x128)"
  docker run --rm --gpus "device=$gpu" \
    -v "$ROOT":/app -w /app -v "$SIB":/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app --name "t128_$tag" \
    "$IMG" python -u main_train_dual.py --data_path data/train \
      --detector_weights "$DET" --episodes $EPISODES --max_steps $STEPS \
      --seed "$seed" --tag "$tag" --results_dir "$OUT" \
    > "$OUT/logs/$tag.log" 2>&1
  echo "[done] GPU$gpu $tag exit=$?"
}

declare -A PID_GPU; free_gpus=("${GPUS[@]}"); i=0; N=${#JOBS[@]}
echo "== full128: $N trainings (100x128) | GPUS=${GPUS[*]} =="
while [ $i -lt $N ] || [ ${#PID_GPU[@]} -gt 0 ]; do
  while [ $i -lt $N ] && [ ${#free_gpus[@]} -gt 0 ]; do
    IFS=':' read -r tag seed <<< "${JOBS[$i]}"
    gpu="${free_gpus[0]}"; free_gpus=("${free_gpus[@]:1}")
    launch "$gpu" "$tag" "$seed" & PID_GPU[$!]="$gpu"; i=$((i+1))
  done
  if [ ${#PID_GPU[@]} -gt 0 ]; then
    wait -n
    for pid in "${!PID_GPU[@]}"; do
      kill -0 "$pid" 2>/dev/null || { free_gpus+=("${PID_GPU[$pid]}"); unset 'PID_GPU[$pid]'; }
    done
  fi
done
echo "== FULL128 TRAININGS DONE =="
