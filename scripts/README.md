# Experiment scripts

Every script that produced a number in the paper. Preserved here from the working
scratchpad on 2026-09-17 so the results stay reproducible; they assume the repo is
mounted at `/app` inside the `ptz-control-obb` image.

## Table 1 (room, pseudo-reference)
- `retrain.sh` / `eval_retrained.sh` — the 5 RL seeds (29-D observation) and their evaluation.
- `run_full128.sh`, `eval_full128.sh` — earlier full-budget runs.
- `tile_baseline.py` — **static tiles K=2 / K=4** (`TILE_K` env var). Writes
  `results/full128_bearing/tiles_K{K}.json`.
- `tile_cover.py` — the coverage mechanism behind the tile result (45.1% of the hemisphere
  holds 85.9% of reference detections).
- `fullmap_train.sh` / `fmap_eval.sh` — the **full-map observation control** (157-D), the
  `RL +full map` row.

## Geometry / metric correctness
- `frustum_check.py` — exact rendered-frustum overlap; pins VFOV 58.7155 deg, corner 48.9254 deg.
- `tilt_curve_frustum.py` — sweep tilt curve under the frustum metric.
- `nadir_cone.py` — the ~16 deg blind cone about nadir and the occupancy band.
- `bearing_invariance.py` — world-bearing association sanity check.
- `id_churn*.py` — identity-churn diagnostics that motivated the world-bearing fix.

## LOAF (public, human annotations)
- `loaf_selftest.py`, `loaf_validity.py`, `loaf_diag.py` — loader checks.
- `loaf_seq.sh`, `loaf_seeds.sh` — per-sequence and per-seed runs.
- `loaf_tilt.py` — **leave-one-sequence-out tilt selection** (returns 65 deg everywhere).
  This is the script that replaced the earlier per-sequence tuning a reviewer flagged as
  test-set tuning; read it before trusting any LOAF number.
- `loaf_nocontrol.py` — static-tiles and full-frame rows on LOAF.
- `tuned_rest.sh` — tuned-baseline sweep across all six sequences.
- `dense_train.sh` / `dense_eval.sh` — policies trained on the dense LOAF scenes.
- `loaf_cov.py`, `loaf_circle.py`, `loaf_disp.py`, `loaf_gate.py`, `loaf_img_ground.py`,
  `loaf_endtoend.py` — supporting analyses.

## Cost
- `timing.py` — the ~3.4x two-crop vs full-frame cost in Sec. 4.5.
- `train_yield.py`, `val_yield.py`, `zoom_yield.py` — detector yield measurements.

## Ablations / variants
- `eval_ablation.sh`, `run_matrix.sh`, `run_rl.sh`, `run_single.sh`, `eval_single.sh`,
  `eval_horizon.sh`, `eval_horizon5.sh`, `eval_ovsweep_tilt.sh`.


## Added 2026-09-18

- `ppo_continuous.py` — the **continuous-action PPO baseline** (Table 1, `PPO cont.` row). Each
  agent emits (d_pan, d_tilt, d_zoom) scaled by the same per-step speed limits the discrete
  vocabulary uses. Evaluate it through the normal harness:
  `python -m evaluation.metrics --policies ppo --ppo_dir <run dir> ...`
  (`PPOContinuousPolicy` lives in `baselines/evaluate.py`; `--ppo_dir` is in `evaluation/metrics.py`.)
- `evaluation/horizon_curve.py` — `HORIZONS` now starts at 5, so the paper's "5–300 step budgets"
  claim is reproducible. It was `[30, 60, 90, 150, 300]`.
- `mkfig.py` — builds Fig. 1 (fisheye + the two rendered crops). Replicates
  `fisheye_env.project_view` exactly rather than approximating it.
- `oracle_headroom.py` — hindsight-oracle view selection. **Not used in the paper.** The
  cumulative "ever covered" measure it computes saturates at 1.000 (two freely-placed views
  eventually cover everyone over 300 steps), so it bounds nothing useful. Kept because the
  per-step variant of the same measurement is informative; see the header comment.
