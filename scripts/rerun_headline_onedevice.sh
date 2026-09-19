#!/bin/bash
# Regenerate EVERY Table 1 number on a SINGLE device, sequentially, so no cell can
# differ because of GPU assignment. Fixes the 0.63/0.68 MP-DQN discrepancy between
# Tables 1 and 2 and bounds the <=0.07 closed-loop nondeterminism.
set -u
ROOT="${ROOT:-/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye}"
SIB="${SIB:-/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect}"
DET="yolo26n_obb_topview_person_290526.pt"; IMG="ptz-control-obb"
OUT="results/onedevice"; GPU="${GPU:-0}"
cd "$ROOT" || exit 1; mkdir -p "$OUT"

run() { local out="$1" pol="$2"; shift 2
  echo "[onedev] GPU$GPU -> $(basename "$out")"
  docker run --rm --gpus "device=$GPU" -v "$ROOT":/app -w /app -v "$SIB":/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app -e YOLO_CONFIG_DIR=/tmp/ul "$IMG" \
    python -u evaluation/metrics.py --data_path data/test --detector_weights "$DET" \
      --steps 300 --episodes 3 --state_dim 29 --gate_deg 15 \
      --policies "$pol" "$@" --out "$out" > "${out%.json}.log" 2>&1; }

# 1) every non-learned policy, stochastic ones over 5 action seeds
run "$OUT/baselines.json" ovsweep,sweep,coordsweep,random,greedy,heuristic --seed_list 0,1,2,3,4

# 2) the 5 MP-DQN seeds
for d in results/full128_bearing/fisheye_dual_*_bearing_s4*; do
  [ -f "$d/tracker_best.pt" ] || continue
  tag=$(basename "$d" | sed -E 's/.*_(s4[0-9])$/\1/')
  run "$OUT/rl_$tag.json" rl --seed_list 0 --tracker "$d/tracker_best.pt" --explorer "$d/explorer_best.pt"
done

# 3) the 5 continuous-action PPO seeds
for d in results/ppo_continuous/s4*; do
  [ -d "$d" ] || continue
  tag=$(basename "$d")
  run "$OUT/ppo_$tag.json" ppo --seed_list 0 --ppo_dir "$d"
done

echo "== ONE-DEVICE HEADLINE RERUN DONE (GPU $GPU) -> $OUT =="
