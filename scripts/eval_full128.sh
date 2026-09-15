#!/usr/bin/env bash
# Deterministic, clip-balanced un-looped eval of the full-budget (100x128) models in
# results/full128/, plus all baselines incl. the coordinated sweep. Each of the 3 test
# clips is scored exactly once (episodes=3, forced_clip_idx), so metrics are not
# confounded by random clip draws. Output: results/full128/eval/<tag>.json.
#
# Usage: GPUS="3 5" bash scripts/eval_full128.sh
set -u
ROOT="/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye"
SIB="/home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect"
DET="yolo26n_obb_topview_person_290526.pt"; IMG="ptz-control-obb"
EVALDIR="results/full128/eval"; GPUS=(${GPUS:-3 5})
cd "$ROOT" || exit 1; mkdir -p "$EVALDIR"

run_metrics() {  # gpu out policies tracker explorer
  local gpu="$1" out="$2" pol="$3" tr="$4" ex="$5"
  local trarg=""; [ -n "$tr" ] && trarg="--tracker $tr --explorer $ex"
  docker run --rm --gpus "device=$gpu" \
    -v "$ROOT":/app -w /app -v "$SIB":/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app \
    "$IMG" python -u evaluation/metrics.py --data_path data/test \
      --detector_weights "$DET" --steps 300 --episodes 3 \
      --seed_list 0 --state_dim 29 --gate_deg 15 \
      --policies "$pol" $trarg --out "$out" \
    > "${out%.json}.log" 2>&1
}

JOBS=()
for d in results/full128/fisheye_dual_*; do
  [ -d "$d" ] || continue
  tag=$(basename "$d" | sed -E 's/^fisheye_dual_[0-9]{8}_[0-9]{6}_?//')
  [ -f "$d/tracker_best.pt" ] || { echo "[skip] $d"; continue; }
  JOBS+=("$EVALDIR/$tag.json|rl|$d/tracker_best.pt|$d/explorer_best.pt")
done
JOBS+=("$EVALDIR/baselines.json|random,sweep,coordsweep,greedy,heuristic||")

echo "== ${#JOBS[@]} eval jobs (deterministic per-clip) | GPUS=${GPUS[*]} =="
declare -A PID_GPU; free_gpus=("${GPUS[@]}"); i=0; N=${#JOBS[@]}
while [ $i -lt $N ] || [ ${#PID_GPU[@]} -gt 0 ]; do
  while [ $i -lt $N ] && [ ${#free_gpus[@]} -gt 0 ]; do
    IFS='|' read -r out pol tr ex <<< "${JOBS[$i]}"
    gpu="${free_gpus[0]}"; free_gpus=("${free_gpus[@]:1}")
    echo "[eval] GPU$gpu -> $(basename "$out")"
    ( run_metrics "$gpu" "$out" "$pol" "$tr" "$ex" ) & PID_GPU[$!]="$gpu"; i=$((i+1))
  done
  if [ ${#PID_GPU[@]} -gt 0 ]; then
    wait -n
    for pid in "${!PID_GPU[@]}"; do
      kill -0 "$pid" 2>/dev/null || { free_gpus+=("${PID_GPU[$pid]}"); unset 'PID_GPU[$pid]'; }
    done
  fi
done
echo "== FULL128 EVAL DONE -> $EVALDIR =="
