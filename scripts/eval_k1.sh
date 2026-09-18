#!/bin/bash
# K=1 arm evaluation: the 5 solo seeds + the K=1 no-learning baselines, on the
# same held-out clips, same detector, same protocol as the K=2 numbers.
set -u
ROOT="${ROOT:-/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye}"
SIB="${SIB:-/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect}"
DET="yolo26n_obb_topview_person_290526.pt"; IMG="ptz-control-obb"
OUT="results/solo_k1/eval"; GPUS=(${GPUS:-0 1 3})
cd "$ROOT" || exit 1; mkdir -p "$OUT"

run() { local gpu="$1" out="$2" pol="$3"; shift 3
  docker run --rm --gpus "device=$gpu" -v "$ROOT":/app -w /app -v "$SIB":/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app -e YOLO_CONFIG_DIR=/tmp/ul "$IMG" \
    python -u evaluation/metrics.py --data_path data/test --detector_weights "$DET" \
      --steps 300 --episodes 3 --seed_list 0 --state_dim 29 --gate_deg 15 \
      --policies "$pol" "$@" --out "$out" > "${out%.json}.log" 2>&1; }

JOBS=()
for d in results/solo_k1/fisheye_solo_*_k1_s4*; do
  [ -f "$d/solo_best.pt" ] || continue
  tag=$(basename "$d" | sed -E 's/.*_(s4[0-9])$/\1/')
  JOBS+=("$OUT/solo_$tag.json|solo|--solo $d/solo_best.pt")
done
# K=1 no-learning references: one camera rastering, and one acting at random.
JOBS+=("$OUT/baselines_k1.json|ovsweep,random,greedy|--n_cams 1 --seed_list 0,1,2,3,4")

echo "== ${#JOBS[@]} K=1 eval jobs | GPUS=${GPUS[*]} =="
declare -A PID_GPU; free=("${GPUS[@]}"); i=0; N=${#JOBS[@]}
while [ $i -lt $N ] || [ ${#PID_GPU[@]} -gt 0 ]; do
  while [ $i -lt $N ] && [ ${#free[@]} -gt 0 ]; do
    IFS='|' read -r out pol extra <<< "${JOBS[$i]}"
    gpu="${free[0]}"; free=("${free[@]:1}")
    echo "[k1] GPU$gpu -> $(basename "$out")"
    ( run "$gpu" "$out" "$pol" $extra ) & PID_GPU[$!]="$gpu"; i=$((i+1))
  done
  if [ ${#PID_GPU[@]} -gt 0 ]; then wait -n
    for pid in "${!PID_GPU[@]}"; do
      kill -0 "$pid" 2>/dev/null || { free+=("${PID_GPU[$pid]}"); unset 'PID_GPU[$pid]'; }
    done
  fi
done
echo "== K=1 EVAL DONE -> $OUT =="
