#!/bin/bash
# Evaluate the retrained (world-bearing novelty) policies with the corrected
# frustum overlap metric, on the same held-out clips and budget as Table 1.
set -u
cd /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye
OUT=results/full128_bearing/eval
mkdir -p "$OUT"
for s in 42 43 44 45 46; do
  d=$(ls -d results/full128_bearing/*bearing_s${s} 2>/dev/null | head -1)
  if [ -z "$d" ]; then echo "seed $s: no run dir yet"; continue; fi
  docker run --rm --gpus '"device=4"' -v "$PWD":/app -w /app \
    -v /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect:/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app -e YOLO_CONFIG_DIR=/tmp/ul \
    -e ASSOC_GATE_DEG="${GATE:-10.0}" -e ASSOC_MAX_AGE="${AGE:-40}" \
    ptz-control-obb python -u evaluation/metrics.py \
      --data_path data/test --detector_weights yolo26n_obb_topview_person_290526.pt \
      --steps 300 --episodes 3 --seeds 5 --seed_list 0 --policies rl \
      --tracker "$d/tracker_best.pt" --explorer "$d/explorer_best.pt" --device cuda \
      --out "$OUT/full_s${s}.json" > "$OUT/rl_s${s}.log" 2>&1
  echo "seed $s evaluated"
done
