#!/bin/bash
set -u
cd /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye
for seq in 0055 0062 0071 0051 0052; do
  for s in 43 44 45 46; do
    out=results/loaf/rl_${seq}_s${s}.json
    [ -f "$out" ] && { echo "skip $seq s$s"; continue; }
    D=$(ls -d results/full128_bearing/*bearing_s$s | head -1)
    docker run --rm --gpus '"device=3"' -v "$PWD":/app -w /app \
      -v /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect:/sibling:ro \
      -v /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect/datasets/loaf:/loaf:ro \
      -e PYTHONPATH=/sibling/ultralytics:/sibling:/app -e YOLO_CONFIG_DIR=/tmp/ul \
      -e ASSOC_GATE_DEG=10.0 -e ASSOC_MAX_AGE=40 \
      ptz-control-obb python -u evaluation/run_loaf.py \
        --loaf_root /loaf --sequences $seq --policies rl \
        --detector_weights yolo26n_obb_topview_person_290526.pt \
        --tracker "$D/tracker_best.pt" --explorer "$D/explorer_best.pt" \
        --steps 300 --out "$out" 2>&1 | grep -aE "disc |Traceback|FATAL"
  done
done
echo "ALL SEED RUNS DONE"
