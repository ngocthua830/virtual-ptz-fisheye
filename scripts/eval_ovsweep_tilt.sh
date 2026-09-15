#!/usr/bin/env bash
# Tilt-sensitivity sweep of the overlap-optimized sweep (ovsweep), review-hardening:
# shows the central negative result is a GEOMETRIC property (overlap collapses to 0 once
# 2*tilt exceeds the 90-deg FOV-cone sum), not a test-set-tuned choice of target_tilt=48.
# Also re-runs the sweep on the TRAIN clips (held-in) so the tilt choice can be shown to be
# selectable without touching test data.
#
# Usage: GPUS="0 1 3 4" bash scripts/eval_ovsweep_tilt.sh
set -u
ROOT="/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye"
SIB="/home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect"
DET="yolo26n_obb_topview_person_290526.pt"; IMG="ptz-control-obb"
OUTDIR="results/full128/ovsweep_tilt"; GPUS=(${GPUS:-0 1 3 4})
TILTS=(30 35 40 44 46 48 50 55 60 65 70)
cd "$ROOT" || exit 1; mkdir -p "$OUTDIR"

run_tilt() {  # gpu split tilt
  local gpu="$1" split="$2" tilt="$3"
  local eps=3; [ "$split" = train ] && eps=5
  local out="$OUTDIR/${split}_t${tilt}.json"
  docker run --rm --gpus "device=$gpu" \
    -v "$ROOT":/app -w /app -v "$SIB":/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app \
    "$IMG" python -u evaluation/metrics.py --data_path "data/$split" \
      --detector_weights "$DET" --steps 300 --episodes "$eps" \
      --seed_list 0 --state_dim 29 --gate_deg 15 \
      --policies ovsweep --ov_target_tilt "$tilt" --out "$out" \
    > "${out%.json}.log" 2>&1
}

JOBS=()
for t in "${TILTS[@]}"; do JOBS+=("test|$t"); done
for t in "${TILTS[@]}"; do JOBS+=("train|$t"); done

echo "== ${#JOBS[@]} ovsweep tilt jobs | GPUS=${GPUS[*]} =="
declare -A PID_GPU; free_gpus=("${GPUS[@]}"); i=0; N=${#JOBS[@]}
while [ $i -lt $N ] || [ ${#PID_GPU[@]} -gt 0 ]; do
  while [ $i -lt $N ] && [ ${#free_gpus[@]} -gt 0 ]; do
    IFS='|' read -r split tilt <<< "${JOBS[$i]}"
    gpu="${free_gpus[0]}"; free_gpus=("${free_gpus[@]:1}")
    echo "[eval] GPU$gpu -> ${split} tilt=${tilt}"
    ( run_tilt "$gpu" "$split" "$tilt" ) & PID_GPU[$!]="$gpu"; i=$((i+1))
  done
  if [ ${#PID_GPU[@]} -gt 0 ]; then
    wait -n
    for pid in "${!PID_GPU[@]}"; do
      kill -0 "$pid" 2>/dev/null || { free_gpus+=("${PID_GPU[$pid]}"); unset 'PID_GPU[$pid]'; }
    done
  fi
done
echo "== OVSWEEP TILT SWEEP DONE -> $OUTDIR =="
