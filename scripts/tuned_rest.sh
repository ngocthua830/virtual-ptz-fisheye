set -u
cd /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye
for sq in 0055 0062 0071 0052; do
  docker run --rm --gpus '"device=4"' -v "$PWD":/app -w /app \
    -v /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect:/sibling:ro \
    -v /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect/datasets/loaf:/loaf:ro \
    \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app -e YOLO_CONFIG_DIR=/tmp/ul \
    -e ASSOC_GATE_DEG=10.0 -e ASSOC_MAX_AGE=40 \
    ptz-control-obb python -u scripts/loaf_tilt.py $sq 2>&1 \
    | grep -avE "Gym|migration|Users of|See the|Ultralytics|settings|WARNING|FisheyePTZEnv"
done
echo "TUNED SWEEP DONE"
