#!/usr/bin/env bash
# Horizon experiment (reviewer §8.2): does RL close the gap to the sweep when there is
# not enough time to raster the whole scene? Evaluate RL + sweep + coordsweep at several
# horizons (short -> full). With un-looped clips, --steps below the clip length caps the
# episode early, so a small --steps = a short observation budget.
#
# Usage: TRK=<tracker.pt> EXP=<explorer.pt> GPU=3 bash scripts/eval_horizon.sh
set -u
ROOT="/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye"
SIB="/home/thuann/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect"
DET="yolo26n_obb_topview_person_290526.pt"; IMG="ptz-control-obb"
OUT="results/ablation/horizon"; GPU="${GPU:-3}"
TRK="${TRK:?set TRK=path/to/tracker_best.pt}"; EXP="${EXP:?set EXP=path/to/explorer_best.pt}"
HORIZONS=(${HORIZONS:-30 60 90 150})
EP="${EP:-6}"; ESEED="${ESEED:-0}"
cd "$ROOT" || exit 1; mkdir -p "$OUT"

for H in "${HORIZONS[@]}"; do
  echo "[horizon] steps=$H"
  docker run --rm --gpus "device=$GPU" \
    -v "$ROOT":/app -w /app -v "$SIB":/sibling:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app \
    "$IMG" python -u evaluation/metrics.py --data_path data/test \
      --detector_weights "$DET" --steps "$H" --episodes "$EP" \
      --seed_list "$ESEED" --state_dim 29 --gate_deg 15 \
      --policies rl,sweep,coordsweep --tracker "$TRK" --explorer "$EXP" \
      --out "$OUT/h${H}.json" \
    > "$OUT/h${H}.log" 2>&1
  echo "[done] steps=$H -> $OUT/h${H}.json"
done
echo "== HORIZON SWEEP DONE -> $OUT =="
