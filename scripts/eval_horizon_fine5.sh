#!/usr/bin/env bash
# 5-seed replay at FINE horizons (5,10,15,20,...) on the bearing-corrected models.
# Answers: is the H=15-20 crossover real, or a single-seed artefact?
# One 300-step deterministic rollout per (model, clip); every H derived from it.
set -u
ROOT="${ROOT:-/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye}"
SIB="${SIB:-/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect}"
DET="yolo26n_obb_topview_person_290526.pt"; IMG="ptz-control-obb"
OUT="results/full128_bearing/horizon_fine5"; GPUS=(${GPUS:-0 1 3})
cd "$ROOT" || exit 1; mkdir -p "$OUT"

run() { local gpu="$1" out="$2" pol="$3"; shift 3
  docker run --rm --gpus "device=$gpu" -v "$ROOT":/app -w /app -v "$SIB":/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app "$IMG" \
    python -u -m evaluation.horizon_curve --data_path data/test \
      --detector_weights "$DET" --state_dim 29 --gate_deg 15 \
      --policies "$pol" "$@" --out "$out" > "${out%.json}.log" 2>&1; }

JOBS=()
for d in results/full128_bearing/fisheye_dual_*_bearing_s4*; do
  [ -f "$d/tracker_best.pt" ] || continue
  tag=$(basename "$d" | sed -E 's/.*_(s4[0-9])$/\1/')
  JOBS+=("$OUT/rl_$tag.json|rl|--tracker $d/tracker_best.pt --explorer $d/explorer_best.pt")
done
JOBS+=("$OUT/baselines.json|ovsweep,random|--random_seeds 0,1,2,3,4")

echo "== ${#JOBS[@]} fine-horizon jobs | GPUS=${GPUS[*]} =="
declare -A PID_GPU; free=("${GPUS[@]}"); i=0; N=${#JOBS[@]}
while [ $i -lt $N ] || [ ${#PID_GPU[@]} -gt 0 ]; do
  while [ $i -lt $N ] && [ ${#free[@]} -gt 0 ]; do
    IFS='|' read -r out pol extra <<< "${JOBS[$i]}"
    gpu="${free[0]}"; free=("${free[@]:1}")
    echo "[fine5] GPU$gpu -> $(basename "$out")"
    ( run "$gpu" "$out" "$pol" $extra ) & PID_GPU[$!]="$gpu"; i=$((i+1))
  done
  if [ ${#PID_GPU[@]} -gt 0 ]; then wait -n
    for pid in "${!PID_GPU[@]}"; do
      kill -0 "$pid" 2>/dev/null || { free+=("${PID_GPU[$pid]}"); unset 'PID_GPU[$pid]'; }
    done
  fi
done
echo "== FINE HORIZON DONE -> $OUT =="
