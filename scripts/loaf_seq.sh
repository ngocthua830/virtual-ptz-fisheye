#!/bin/bash
# One sequence per container: a memory kill costs at most one sequence.
set -u
cd /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye
D=$(ls -d results/full128_bearing/*bearing_s42 | head -1)
for seq in 0071 0051 0052 0054; do
  echo "=== $seq ==="
  docker run --rm --gpus '"device=3"' -v "$PWD":/app -w /app \
    -v /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect:/sibling:ro \
    -v /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect/datasets/loaf:/loaf:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app -e YOLO_CONFIG_DIR=/tmp/ul \
    -e ASSOC_GATE_DEG=10.0 -e ASSOC_MAX_AGE=40 \
    ptz-control-obb python -u evaluation/run_loaf.py \
      --loaf_root /loaf --sequences $seq --policies ovsweep,tiles2,random,rl \
      --detector_weights yolo26n_obb_topview_person_290526.pt \
      --tracker "$D/tracker_best.pt" --explorer "$D/explorer_best.pt" \
      --steps 300 --out results/loaf/loaf_eval.json 2>&1 \
    | grep -aE "disc |FATAL|Error|Traceback|already complete"
done
echo "ALL SEQUENCES ATTEMPTED"
