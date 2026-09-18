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

### K=1 view-budget arm (paper §5.3, "Nor of the view budget")
| script | produces |
|---|---|
| `train_solo_k1.sh` | trains the 5 K=1 seeds (`main_train_solo.py`, `n_cams=1`), same 100×128 budget and hyper-parameters as the K=2 runs |
| `eval_k1.sh` | evaluates those 5 seeds plus the K=1 no-learning references (ovsweep/random/greedy) on the same held-out clips |
| `k1_summary.py` | prints both arms side by side and Δ(K) = sweep − learned, with per-seed win counts |

K=1 is selected with `--n_cams 1` (or `--solo <solo_best.pt>`, which implies it) in
`evaluation/metrics.py`. At K=1 the pair-geometry observation features `s[12]` (azimuth
separation) and `s[23]` (partner pan) are held at zero and the anti-overlap reward term is
inactive, since there is no partner view; the single agent receives **both** the tracker and
explorer reward terms, because it must follow *and* discover. `OverlapOptimizedSweepPolicy`
degenerates to a plain single-camera raster at K=1 for the same reason — worth remembering when
reading the narrowed gap.

### Fine-grained horizon sweep (paper Table 2)
`eval_horizon_fine5.sh` replays all 5 bearing-corrected seeds at H = 5,10,15,20,30,60,90,150,300.
The earlier `eval_horizon5.sh` only covered H ≥ 30; the H ≤ 20 rows are what showed the
apparent 15–20 crossover to be a single-seed artefact.
