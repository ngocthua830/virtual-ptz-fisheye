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

### Fine-grained horizon sweep (paper §5.1)
`eval_horizon_fine5.sh` replays all 5 bearing-corrected seeds at H = 5,10,15,20,30,60,90,150,300.
The earlier `eval_horizon5.sh` only covered H ≥ 30; the H ≤ 20 rows are what showed the
apparent 15–20 crossover to be a single-seed artefact. (These had their own table in earlier
drafts; the paper now states the numbers inline, so this script is the full record.)

### Per-frame one-to-one precision/recall (paper §5.5)
| script | produces |
|---|---|
| `loaf_pr.sh` | runs every policy on the six LOAF sequences with `per_frame_pr` wired in, writing `results/loaf/loaf_pr.json` |

`per_frame_pr()` (in `evaluation/metrics.py`) is greedy closest-pair assignment inside the same
15° gate, each reference and each detection used at most once. It charges for false positives,
which the discovery family never does, and uses **no identities**.

**Two traps, both of which cost us a rerun:**
- `--ov_target_tilt` is the polar tilt itself and defaults to **48**, not 45 and not 65. The
  paper's `sw@65` needs `--ov_target_tilt 65`. Check any new run against the published discovery
  cells (0.603 / 0.372 / 0.559 / 0.908 / 0.535 / 0.740 for 0062/0055/0071/0051/0052/0054) before
  trusting its P/R — that is how we caught a run at tilt 68.
- `recall_inview` is **broken** and now returns `NaN` with `recall_inview_valid=False` unless
  `TP_inview == TP` holds. Do not report it until the frustum test is fixed.

### Dataset screening, and datasets we rejected
| script | produces |
|---|---|
| `allocation_hardness.py` | ground-truth-only screen: `C_static*` (best fixed view pair, hindsight), `C_oracle` (best pair per frame), `H_alloc` = the headroom adaptive control competes for. No detector, no policy, no GPU. |
| `frida_selftest.py` | known-answer checks on the FRIDA loader (bearing model, box centres, and that identities are *given* rather than re-derived) |

Detection difficulty is **not** allocation difficulty: a crowded scene whose people all sit in one
sector is trivial to allocate. We screened candidate datasets on `H_alloc` before spending GPU
time, and that is why FRIDA (loader in `evaluation/frida.py`, driver `run_frida.py`) is released
but not in the paper — its headroom is too small to separate controllers.

### Reruns for reproducibility
| script | produces |
|---|---|
| `rerun_headline_onedevice.sh` | re-runs every headline row on **one** device in one sequential pass. An earlier release mixed hardware; this moved the heuristic by 0.107 and MP-DQN from 0.68 to 0.63. |
| `retrain_4x.sh` | 4× environment-budget retrain for PPO and MP-DQN, testing whether the deficit is undertraining (it is not) |
| `fsac_baseline.py` | FSAC, four-sector alternating coverage: a budget-matched non-learned baseline that picks poses directly rather than slewing |
