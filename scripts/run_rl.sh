set -e
cd /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye
declare -A RUN=( [42]=fisheye_dual_20260712_131048_full_s42 [43]=fisheye_dual_20260712_131048_full_s43 \
                 [44]=fisheye_dual_20260712_140347_full_s44 [45]=fisheye_dual_20260712_140633_full_s45 \
                 [46]=fisheye_dual_20260712_145043_full_s46 )
for s in 42 43 44 45 46; do
  d=results/full128/${RUN[$s]}
  docker run --rm --gpus all -v "$PWD":/app -w /app \
    -v /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect:/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app -e YOLO_CONFIG_DIR=/tmp/ul \
    ptz-control-obb python -u evaluation/metrics.py \
    --data_path data/test --detector_weights yolo26n_obb_topview_person_290526.pt \
    --steps 300 --episodes 3 --seeds 5 --seed_list 0 --policies rl \
    --tracker $d/tracker_best.pt --explorer $d/explorer_best.pt --device cuda:4 \
    --out results/full128/eval_frustum/full_s${s}.json \
    > results/full128/eval_frustum/rl_s${s}.log 2>&1
  echo "seed $s done"
done
