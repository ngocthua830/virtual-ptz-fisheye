#!/bin/bash
# MP-DQN at 4x the environment budget (400 episodes x 128 steps), mirroring the PPO 4x
# convergence check. Answers: does the MP-DQN deficit close with more training?
set -u
ROOT="${ROOT:-/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye}"
SIB="${SIB:-/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect}"
cd "$ROOT" || exit 1
OUT=results/full128_bearing_4x; mkdir -p "$OUT/logs"
GPUS=(${GPUS:-1 3}); SEEDS=(${SEEDS:-42 43})
for i in "${!SEEDS[@]}"; do
  s=${SEEDS[$i]}; g=${GPUS[$i]}
  docker run --rm --gpus "\"device=$g\"" -v "$ROOT":/app -w /app \
    -v "$SIB":/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app -e YOLO_CONFIG_DIR=/tmp/ul \
    -e ASSOC_GATE_DEG="${GATE:-10.0}" -e ASSOC_MAX_AGE="${AGE:-40}" \
    --name mpdqn4x_s${s} -d \
    ptz-control-obb python -u main_train_dual.py \
      --episodes 400 --max_steps 128 --seed "$s" \
      --data_path ./data/train/ \
      --detector_weights yolo26n_obb_topview_person_290526.pt \
      --device cuda --results_dir "$OUT" --tag "4x_s${s}" > /dev/null
  echo "launched MP-DQN 4x seed $s on GPU $g"
done
