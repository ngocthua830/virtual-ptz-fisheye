set -u
cd /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye
mkdir -p results/full128_fullmap/eval
for s in 42 43 44 45 46; do
  d=$(ls -d results/full128_fullmap/*fmap_s${s} | head -1)
  docker run --rm --gpus '"device=3"' -v "$PWD":/app -w /app \
    -v /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect:/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app -e YOLO_CONFIG_DIR=/tmp/ul \
    -e ASSOC_GATE_DEG=10.0 -e ASSOC_MAX_AGE=40 \
    ptz-control-obb python -u evaluation/metrics.py \
      --data_path data/test --detector_weights yolo26n_obb_topview_person_290526.pt \
      --steps 300 --episodes 3 --seeds 5 --seed_list 0 --policies rl \
      --state_dim 157 --full_map --tracker "$d/tracker_best.pt" --explorer "$d/explorer_best.pt" \
      --device cuda --out results/full128_fullmap/eval/s${s}.json 2>&1 \
    | grep -aE "^rl |disc=|FATAL|Traceback" | tail -2 | sed "s/^/ s$s /"
done
echo FMAP_EVAL_DONE
