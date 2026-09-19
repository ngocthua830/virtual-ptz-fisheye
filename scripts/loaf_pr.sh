#!/bin/bash
# Per-frame, one-to-one precision/recall on LOAF's human boxes, ID-free (paper Sec. 4.5, 5.5).
# Complements the discovery family, which has no precision term and never charges for a
# false positive.
#
# OV_TILT is the geometric sweep's POLAR TILT and it matters: run_loaf.py defaults it to 48,
# while the paper's GeoSweep column on LOAF is the leave-one-sequence-out choice, 65. Passing
# anything else silently evaluates a different geometry -- we lost a run to exactly that.
# Sanity-check any rerun against the published discovery cells before trusting its P/R:
#   0062 0.603 | 0055 0.372 | 0071 0.559 | 0051 0.908 | 0052 0.535 | 0054 0.740
# The tiles/random/rl policies ignore OV_TILT.
set -u
ROOT="${ROOT:-/CoreTeam_NAS/WorkingSpace/thuann/PhD/lab_fisheye}"
SIB="${SIB:-/CoreTeam_NAS/WorkingSpace/thuann/PhD/fisheye_person_detect}"
LOAF="${LOAF:-$SIB/datasets/loaf}"
OV_TILT="${OV_TILT:-65}"
GPU="${GPU:-0}"
OUT="${OUT:-results/loaf/loaf_pr.json}"
POLICIES="${POLICIES:-ovsweep,tiles2,random,rl}"

cd "$ROOT"
D=$(ls -d results/full128_bearing/*bearing_s42 | head -1)
for seq in 0062 0055 0071 0051 0052 0054; do
  echo "=== $seq ==="
  docker run --rm --gpus "\"device=$GPU\"" -v "$PWD":/app -w /app \
    -v "$SIB":/sibling:ro -v "$LOAF":/loaf:ro \
    -e PYTHONPATH=/sibling/ultralytics:/sibling:/app -e YOLO_CONFIG_DIR=/tmp/ul \
    -e ASSOC_GATE_DEG=10.0 -e ASSOC_MAX_AGE=40 \
    ptz-control-obb python -u evaluation/run_loaf.py \
      --loaf_root /loaf --sequences "$seq" --policies "$POLICIES" \
      --ov_target_tilt "$OV_TILT" \
      --detector_weights yolo26n_obb_topview_person_290526.pt \
      --tracker "$D/tracker_best.pt" --explorer "$D/explorer_best.pt" \
      --steps 300 --out "$OUT" 2>&1 \
    | grep -aE "disc |pr_|FATAL|Error|Traceback|already complete"
done
echo "LOAF PR DONE -> $OUT"
