#!/usr/bin/env bash
# Un-looped ground-truth evaluation of every trained model in results/ablation/,
# plus the non-learned baselines once. Each model is scored on a FIXED set of
# held-out eval episodes (same eval seed for all runs), so per-run RL scores are
# comparable. Output: one JSON per run in results/ablation/eval/.
#
# Usage: GPUS="0 1 3" EP=6 bash scripts/eval_ablation.sh
set -u
ROOT="/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye"
SIB="/home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect"
DET="yolo26n_obb_topview_person_290526.pt"
IMG="ptz-control-obb"
EVALDIR="results/ablation/eval"
GPUS=(${GPUS:-0 1 3})
EP="${EP:-6}"          # eval episodes (fixed eval seed -> same clips for every run)
STEPS="${STEPS:-300}"  # cap; un-looped episodes end at true clip length (~144)
ESEED="${ESEED:-0}"

cd "$ROOT" || exit 1
mkdir -p "$EVALDIR"

run_metrics() {  # gpu, out, policies, tracker, explorer
  local gpu="$1" out="$2" pol="$3" tr="$4" ex="$5"
  local trarg="" ; [ -n "$tr" ] && trarg="--tracker $tr --explorer $ex"
  docker run --rm --gpus "device=$gpu" \
    -v "$ROOT":/app -w /app -v "$SIB":/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app \
    "$IMG" python -u evaluation/metrics.py --data_path data/test \
      --detector_weights "$DET" --steps "$STEPS" --episodes "$EP" \
      --seed_list "$ESEED" --state_dim 29 --gate_deg 15 \
      --policies "$pol" $trarg --out "$out" \
    > "${out%.json}.log" 2>&1
}

# ---- collect jobs: one per run dir (RL only) + one baselines job ----
JOBS=()   # "out|policies|tracker|explorer"
for d in results/ablation/fisheye_dual_*; do
  [ -d "$d" ] || continue
  tag=$(basename "$d" | sed -E 's/^fisheye_dual_[0-9]{8}_[0-9]{6}_?//')
  [ -f "$d/tracker_best.pt" ] || { echo "[skip] no ckpt in $d"; continue; }
  JOBS+=("$EVALDIR/$tag.json|rl|$d/tracker_best.pt|$d/explorer_best.pt")
done
JOBS+=("$EVALDIR/baselines.json|random,sweep,greedy,heuristic||")

echo "== ${#JOBS[@]} eval jobs | EP=$EP eval_seed=$ESEED | GPUS=${GPUS[*]} =="

declare -A PID_GPU
free_gpus=("${GPUS[@]}"); i=0; N=${#JOBS[@]}
while [ $i -lt $N ] || [ ${#PID_GPU[@]} -gt 0 ]; do
  while [ $i -lt $N ] && [ ${#free_gpus[@]} -gt 0 ]; do
    IFS='|' read -r out pol tr ex <<< "${JOBS[$i]}"
    gpu="${free_gpus[0]}"; free_gpus=("${free_gpus[@]:1}")
    echo "[eval] GPU$gpu -> $(basename "$out")"
    ( run_metrics "$gpu" "$out" "$pol" "$tr" "$ex" ) &
    PID_GPU[$!]="$gpu"; i=$((i+1))
  done
  if [ ${#PID_GPU[@]} -gt 0 ]; then
    wait -n
    for pid in "${!PID_GPU[@]}"; do
      kill -0 "$pid" 2>/dev/null || { free_gpus+=("${PID_GPU[$pid]}"); unset 'PID_GPU[$pid]'; }
    done
  fi
done
echo "== ALL EVALS DONE -> $EVALDIR =="
