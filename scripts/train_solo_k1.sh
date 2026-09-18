#!/bin/bash
# K=1 arm: 5 independent seeds, same budget/hyper-parameters as the K=2 runs
# (100 episodes x 128 steps). One seed per GPU.
set -u
ROOT="${ROOT:-/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye}"
SIB="${SIB:-/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect}"
cd "$ROOT" || exit 1
OUT=results/solo_k1
mkdir -p "$OUT/logs"
GPUS=(${GPUS:-0 1 3 0 1})
SEEDS=(42 43 44 45 46)
for i in "${!SEEDS[@]}"; do
  s=${SEEDS[$i]}; g=${GPUS[$i]}
  docker run --rm --gpus "\"device=$g\"" -v "$PWD":/app -w /app \
    -v "$SIB":/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app -e YOLO_CONFIG_DIR=/tmp/ul \
    -e ASSOC_GATE_DEG="${GATE:-10.0}" -e ASSOC_MAX_AGE="${AGE:-40}" \
    --name solo_s${s} -d \
    ptz-control-obb python -u main_train_solo.py \
      --episodes 100 --max_steps 128 --seed "$s" \
      --data_path ./data/train/ \
      --detector_weights yolo26n_obb_topview_person_290526.pt \
      --device cuda --results_dir "$OUT" --tag "k1_s${s}" > /dev/null
  echo "launched K=1 seed $s on GPU $g"
done
