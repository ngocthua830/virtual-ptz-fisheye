#!/bin/bash
# Dense-trained policies: train on LOAF TRAIN sequences (val held out), picking the
# densest ones the detector can actually see (yield >= ~0.15, 15-38 people/frame).
set -u
cd /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye
SEQS=0043,0048,0046,0021,0017,0012
OUT=results/loaf_dense
mkdir -p $OUT
GPUS=(0 3 4 5 2); SEEDS=(42 43 44 45 46)
for i in 0 1 2 3 4; do
  s=${SEEDS[$i]}; g=${GPUS[$i]}
  docker run --rm --gpus "\"device=$g\"" -v "$PWD":/app -w /app \
    -v /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect:/sibling:ro \
    -v /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect/datasets/loaf:/loaf:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app -e YOLO_CONFIG_DIR=/tmp/ul \
    -e ASSOC_GATE_DEG=10.0 -e ASSOC_MAX_AGE=40 \
    --name dense_s${s} -d \
    ptz-control-obb python -u main_train_dual.py \
      --episodes 100 --max_steps 128 --seed $s --data_path ./data/train/ \
      --loaf_root /loaf --loaf_split train --loaf_seqs $SEQS \
      --detector_weights yolo26n_obb_topview_person_290526.pt \
      --device cuda --results_dir $OUT --tag dense_s${s} > /dev/null
  echo "launched dense seed $s on GPU $g"
done
sleep 25
for s in 42 43 44 45 46; do
  echo "  s$s: $(docker logs dense_s$s 2>&1 | grep -ac 'LOAF source') LOAF-source line, $(docker logs dense_s$s 2>&1 | grep -ac 'detector loaded') detector"
done
