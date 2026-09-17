#!/bin/bash
# Retrain the dual-agent policies against the FIXED novelty signal (world-bearing
# identity association). Same budget and seeds as the 2026-07-12 run:
# 100 episodes x 128 steps, seeds 42-46, one seed per GPU.
set -u
cd /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye
GATE="${GATE:-10.0}"
OUT=results/full128_bearing
mkdir -p "$OUT/logs"
GPUS=(0 3 4 5 2)          # host GPU ids with free memory
SEEDS=(42 43 44 45 46)
for i in "${!SEEDS[@]}"; do
  s=${SEEDS[$i]}; g=${GPUS[$i]}
  docker run --rm --gpus "\"device=$g\"" -v "$PWD":/app -w /app \
    -v /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect:/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app -e YOLO_CONFIG_DIR=/tmp/ul \
    -e ASSOC_GATE_DEG="$GATE" -e ASSOC_MAX_AGE="${AGE:-40}" \
    --name retrain_s${s} -d \
    ptz-control-obb python -u main_train_dual.py \
      --episodes 100 --max_steps 128 --seed "$s" \
      --data_path ./data/train/ \
      --detector_weights yolo26n_obb_topview_person_290526.pt \
      --device cuda --results_dir "$OUT" --tag "bearing_s${s}" > /dev/null
  echo "launched seed $s on host GPU $g"
done
sleep 20
docker ps --filter "name=retrain_" --format '  {{.Names}}  {{.Status}}'
