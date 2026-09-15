#!/usr/bin/env bash
# Train the full model (5 independent seeds) + 4 single-mechanism ablations
# (3 seeds each) = 17 independent trainings, distributed over the free GPUs.
#
# Each --seed gives an independent run: different network init, exploration,
# replay order, AND clip-sampling order (data_path is a directory). This makes
# the statistical unit an independent training run (the reviewer's §3.2 fix),
# not a reshuffle of the same eval clips.
#
# Usage:
#   EPISODES=60 STEPS=96 GPUS="0 1 3" bash scripts/run_matrix.sh
set -u

ROOT="/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye"
SIB="/home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect"
DET="yolo26n_obb_topview_person_290526.pt"
IMG="ptz-control-obb"
OUT="results/ablation"
LOGDIR="$OUT/logs"

EPISODES="${EPISODES:-60}"
STEPS="${STEPS:-96}"
GPUS=(${GPUS:-0 1 3})

cd "$ROOT" || exit 1
mkdir -p "$LOGDIR"

# ---- run matrix: "tag|seed|extra training args" ----
JOBS=(
  "full_s42|42|"
  "full_s43|43|"
  "full_s44|44|"
  "full_s45|45|"
  "full_s46|46|"
  "nocov_s42|42|--coverage_weight 0"
  "nocov_s43|43|--coverage_weight 0"
  "nocov_s44|44|--coverage_weight 0"
  "nonov_s42|42|--novelty_weight 0"
  "nonov_s43|43|--novelty_weight 0"
  "nonov_s44|44|--novelty_weight 0"
  "noovlp_s42|42|--overlap_thresh_deg 0"
  "noovlp_s43|43|--overlap_thresh_deg 0"
  "noovlp_s44|44|--overlap_thresh_deg 0"
  "unif_s42|42|--peripheral_weight 1.0"
  "unif_s43|43|--peripheral_weight 1.0"
  "unif_s44|44|--peripheral_weight 1.0"
)

launch() {
  local gpu="$1" tag="$2" seed="$3" extra="$4"
  echo "[launch] GPU$gpu  $tag  seed=$seed  extra='$extra'"
  docker run --rm --gpus "device=$gpu" \
    -v "$ROOT":/app -w /app -v "$SIB":/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app \
    --name "train_$tag" \
    "$IMG" \
    python -u main_train_dual.py \
      --data_path data/train --detector_weights "$DET" \
      --episodes "$EPISODES" --max_steps "$STEPS" --seed "$seed" $extra \
      --tag "$tag" --results_dir "$OUT" \
    > "$LOGDIR/$tag.log" 2>&1
  echo "[done]   GPU$gpu  $tag  exit=$?"
}

# ---- simple GPU-pool scheduler: one job per GPU, launch next as a GPU frees ----
declare -A PID_GPU   # pid -> gpu
free_gpus=("${GPUS[@]}")
i=0
N=${#JOBS[@]}
echo "== matrix: $N trainings | EPISODES=$EPISODES STEPS=$STEPS | GPUS=${GPUS[*]} =="

while [ $i -lt $N ] || [ ${#PID_GPU[@]} -gt 0 ]; do
  # fill free GPUs
  while [ $i -lt $N ] && [ ${#free_gpus[@]} -gt 0 ]; do
    IFS='|' read -r tag seed extra <<< "${JOBS[$i]}"
    gpu="${free_gpus[0]}"; free_gpus=("${free_gpus[@]:1}")
    launch "$gpu" "$tag" "$seed" "$extra" &
    PID_GPU[$!]="$gpu"
    i=$((i+1))
  done
  # wait for any one to finish, reclaim its GPU
  if [ ${#PID_GPU[@]} -gt 0 ]; then
    wait -n
    for pid in "${!PID_GPU[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        free_gpus+=("${PID_GPU[$pid]}")
        unset 'PID_GPU[$pid]'
      fi
    done
  fi
done
echo "== ALL TRAININGS DONE =="
