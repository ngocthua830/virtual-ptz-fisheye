set -u
cd /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye
OUT=results/full128_fullmap; mkdir -p $OUT
GPUS=(0 3 4 5 2); SEEDS=(42 43 44 45 46)
for i in 0 1 2 3 4; do
  s=${SEEDS[$i]}; g=${GPUS[$i]}
  docker run --rm --gpus "\"device=$g\"" -v "$PWD":/app -w /app \
    -v /home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect:/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app -e YOLO_CONFIG_DIR=/tmp/ul \
    -e ASSOC_GATE_DEG=10.0 -e ASSOC_MAX_AGE=40 --name fmap_s${s} -d \
    ptz-control-obb python -u main_train_dual.py \
      --episodes 100 --max_steps 128 --seed $s --data_path ./data/train/ \
      --state_dim 157 --full_map \
      --detector_weights yolo26n_obb_topview_person_290526.pt \
      --device cuda --results_dir $OUT --tag fmap_s${s} > /dev/null
  echo "launched full-map seed $s on GPU $g"
done
sleep 25; for s in 42 43 44 45 46; do
  echo "  s$s: $(docker logs fmap_s$s 2>&1 | grep -ac 'detector loaded') detector, state $(docker logs fmap_s$s 2>&1 | grep -ao 'State dim: [0-9]*' | head -1)"
done
