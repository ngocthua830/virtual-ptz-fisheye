# virtual-ptz-fisheye

Turning **one overhead fisheye camera into a bank of virtual PTZ views**, and asking the
question the paper is named after:

> **When does *learned* control of those views actually help, and when does simple geometry
> suffice?**

Code and evaluation harness for *"When Does Learned Virtual-PTZ Control Help? Geometric,
Learned, and Hybrid Policies from a Single Fisheye Camera"* (under review).

---

## The result, stated up front

**On the regime we studied, learned control does not help.** **GeoSweep** (called `Ovlp-Sweep`
in the code and in releases before 2026-09-19) — hold the two virtual cameras 180° apart in
azimuth at a tilt high enough that their rendered frusta are disjoint by geometry, then co-rotate
them to raster the hemisphere — leads every *learned* controller on every metric at once:

| policy | discovery ↑ | time-to-detect ↓ | observed-time frac ↑ | solid-angle overlap ↓ |
|---|---|---|---|---|
| **GeoSweep (45°, no learning)** | **0.92** | **1.09** | 0.64 | **0.00** |
| *FSAC — 4-sector alternation, no control* | *0.98* | *39.72* | *0.52* | *0.00* |
| *static tiles K=2 (no control at all)* | *0.95* | *12.51* | *0.85* | — |
| MP-DQN (5 seeds) | 0.63 ± 0.09 | 3.64 ± 2.47 | 0.44 ± 0.14 | 0.06 ± 0.02 |
| PPO, continuous action (5 seeds) | 0.66 ± 0.04 | 2.48 ± 0.44 | 0.49 ± 0.10 | 0.01 ± 0.00 |
| RL + full coverage map (5 seeds) | 0.58 ± 0.11 | 4.69 ± 3.03 | 0.34 ± 0.14 | 0.03 ± 0.02 |
| coordsweep (no learning) | 0.77 | 3.08 | 0.48 | 0.11 |
| heuristic (no learning) | 0.64 | 9.06 | 0.33 | 0.07 |
| naive sweep (no learning) | 0.63 | 4.68 | 0.40 | 0.14 |
| random (5 action seeds) | 0.61 | 5.14 | 0.39 | 0.07 |
| K=1 solo agent (5 seeds) | 0.52 ± 0.13 | 11.9 | 0.33 | — |
| K=1 sweep (no learning) | 0.71 | 2.81 | 0.39 | — |

(std is over the 5 training seeds, population convention. The sweep is *requested* at 48° with a
±3° deadband, so the realised geometry is ~45°; the paper labels it by the realised tilt.
**Every row above is from a single device in one sequential run** — an earlier release mixed
hardware, which moved the heuristic by 0.107 and put MP-DQN at 0.68 rather than 0.63. On one
device MP-DQN *ties* the naive sweep and the heuristic; it does not beat them.
FSAC and tiles pick poses directly while GeoSweep and the learned policies move ≤25°/step, so
they share a crop budget but not an action space.)

Three honest readings we want to keep attached to those numbers:

- **The `random` row was wrong until 2026-09-18, and it mattered.** It had been computed from a
  *single* action seed (0.74 discovery). Re-run over five action seeds it is **0.61**, so the
  learned policy does beat random — the earlier numbers had made the negative result look
  stronger than the data supports. (On the later single-device rerun the margin is thinner still:
  MP-DQN 0.63 against random 0.61.) Two claims in the paper were corrected with it. Lesson: the
  "one seed is not a sample" rule applies to *stochastic baselines*, not just to learned policies.
- **Halving the view budget narrows the gap but does not close it.** At `K=1` (one crop, one
  agent, five seeds retrained at the same budget) the sweep leads by 0.19 instead of 0.25 — but
  learning still trails on 5/5 seeds at *both* budgets. Part of that narrowing is structural: at
  K=1 the sweep's antipodal-disjointness construction has no partner to exploit, so it degenerates
  to a plain raster. We read it as a trend, not a crossover.
- **Giving the learned agent *more* observation does not help.** The hand-coded heuristic reads
  the 16×8 coverage map directly while the policy saw only aggregates of it, so we ran the
  control: re-training all five seeds with the raw map appended (157-D observation, same budget)
  *widens* the gap rather than closing it — the `RL + full coverage map` row above. At fixed
  capacity and episode count the wider observation is harder to learn from, not more
  informative. We cannot exclude that a larger network or longer schedule would exploit it.
- **Earlier versions of this work claimed the opposite, twice.** First, a "2.8× faster, ties the
  sweep" headline turned out to be an artefact of (i) evaluating a single training seed and
  (ii) a video-looping bug that gave every policy ~5× its real observation window. Second, on
  LOAF we briefly measured learned control *beating* geometry on dense scenes — that was a
  **mis-tuned baseline**: the sweep was pinned at a tilt chosen for our room while the learned
  policy was free to vary tilt (60° ± 21° vs 45° ± 1°). Re-tuning the baseline on the new scene,
  with the tilt selected **leave-one-sequence-out** so no test label enters the choice, reversed
  it again. Both corrections are why the evaluation harness looks the way it does.

The regime matters: two wide, non-overlapping views already blanket a small, fully-observable
room, so the prioritisation and zoom that active control buys have nothing to win. We expect the
answer to flip in scenes too large, dense or occluded for wide static views.

## It is not just our room: LOAF

The same comparison runs on six held-out sequences of the public **LOAF** dataset (ICCV 2023) —
ceiling-mounted overhead fisheye, 2048², 5.1–31.3 people/frame, scored against **human**
annotations rather than a detector-based pseudo-reference, with a sequence-level split so the
detector never saw these frames.

With the geometric baseline's aim chosen **leave-one-sequence-out** (which returns 65° for every
sequence — per-scene tuning turned out to be unnecessary), the tuned sweep beats the
room-trained policy on **4/6** sequences, and policies *retrained on those dense scenes* on
**6/6**. Static tiles, which win in our room, win only **1/6** here — the tile advantage tracks
how concentrated occupancy is, and is not a general claim that control is unnecessary.

### Per-frame one-to-one matching — where the gap closes

Discovery and its relatives are an **angular-neighbourhood coverage** test: a person counts as
seen if *any* detection lands within 15°, so duplicates are free and false positives are never
charged. `per_frame_pr()` in `evaluation/metrics.py` is the complement — greedy closest-pair
assignment per frame, each reference and each detection used at most once, and **no identities**,
so it is the one number free of the track-association gate every other LOAF figure inherits.

| policy (6 LOAF sequences, means) | precision | recall | F1 | FP per TP |
|---|---|---|---|---|
| **GeoSweep @65°** | 0.336 | 0.149 | **0.200** | 2.5 |
| MP-DQN | 0.346 | 0.147 | 0.197 | 2.2 |
| static tiles | 0.443 | 0.101 | 0.144 | 3.6 |
| random | 0.273 | 0.061 | 0.096 | 3.9 |

**Under this metric GeoSweep and MP-DQN are indistinguishable.** We are stating that plainly
because it qualifies the headline: the learned deficit we report is specific to the *cumulative,
identity-based* measures. Instant by instant the two controllers see about equally much.

It also shows nobody is doing well. At most 15% of annotated person-frames are matched, at
2.2–3.9 false positives per true positive — 15.9 for static tiles at 31.3 people/frame. An
allocation oracle says why: re-choosing the view pair *every frame* with hindsight covers only
0.02–0.10 more people than the best *fixed* pair (`scripts/allocation_hardness.py`). At K=2 there
is almost nothing left for adaptation to win.

⚠️ **`recall_inview` in `per_frame_pr()` is broken — do not quote it.** It should be ≥ `recall`
by construction and is not (LOAF 0051: recall 0.292, "inview" 0.076). It now returns `NaN` and
sets `recall_inview_valid=False` whenever the invariant `TP_inview == TP` fails, rather than a
plausible wrong number. It is not used in the paper. Fix and re-validate before relying on it.

Reproduce with `evaluation/run_loaf.py` (loader: `evaluation/loaf.py`), drivers in
`scripts/loaf_*.sh` / `scripts/loaf_*.py`; `scripts/loaf_tilt.py` is the LOSO tilt selection.
**Gotcha:** `--ov_target_tilt` is the *polar tilt itself*, and `run_loaf.py` defaults it to **48**.
The paper's `sw@65` column is `--ov_target_tilt 65`; passing anything else silently evaluates a
different geometry.


## Layout

```
environments/     virtual-PTZ gym environments
  fisheye_env.py      equidistant fisheye -> perspective unwarp at arbitrary pan/tilt/zoom
                      (project_view); FOV = base_fov / zoom
  dual_fisheye_env.py two cooperating virtual cameras + shared coverage map
  scene_state.py      track registry / scene bookkeeping
evaluation/       the reward-independent measurement harness
  metrics.py          discovery, time-to-detect, coverage%, inter-camera + solid-angle overlap,
                      observed-time fraction, max unobserved gap; oracle on the raw fisheye.
                      Overlap uses the EXACT rendered frustum, not a cone: at base_fov=90 the
                      view is a 90.0°×58.7° pyramid (corners 48.9°, top/bottom edges 29.4°), so
                      a 45° spherical cap overstates its solid angle by 1.30×. Two axes 180°
                      apart are disjoint for tilt >= VFOV/2 = 29.4°, not 45°.
  survival_ttd.py     censoring-aware latency (Kaplan-Meier, restricted-mean TTD)
  horizon_curve.py    discovery vs observation budget, fixed-cohort
  tilt_curve.py       tilt sensitivity of the geometric sweep
  loaf.py             LOAF dataset loader: sequence split, human annotations -> reference tracks
  run_loaf.py         run any policy on a LOAF sequence against those references
  frida.py            FRIDA loader (person_id is ANNOTATED, so identity metrics need no
                      association step); run_frida.py drives it. Self-test:
                      scripts/frida_selftest.py. Screened but not used in the paper --
                      its allocation headroom is too small to separate controllers.
models/agent/     MP-DQN dual-agent (tracker + explorer), parameterized action space
baselines/        non-learned policies: sweep, coordinated sweep, overlap-optimized sweep,
                  greedy, heuristic, random  (+ multi-seed evaluation drivers)
configs/          scenario configuration
scripts/          every driver that produced a number in the paper; scripts/README.md maps
                  each script to the claim it backs (tile_baseline.py, loaf_tilt.py,
                  fullmap_train.sh, frustum_check.py, timing.py, ...)
main_train*.py    training entry points (single / dual)
visualize*.py     rollout visualisation
```

## Not included, and why

- **Video data.** The clips are of consenting participants in a private indoor scene and are not
  redistributable. `DATA_CAPTURE_SPEC.md` documents the capture protocol so the setup can be
  reproduced.
- **Detector weights** (`yolo26n_obb_topview_person*.pt`). See the licence note below.
- **`results/`** (~358 MB of run artefacts).

`scripts/*.sh` contain absolute paths from the machine they were run on — set the repo root at
the top of each script before use.

## Licence

**GNU Affero General Public License v3.0** (`LICENSE`).

AGPL-3.0 is chosen for compatibility, not preference: `environments/fisheye_env.py` uses
**Ultralytics**, which is AGPL-3.0 absent a commercial licence, and AGPL is copyleft — work
distributed on top of it must carry the same terms. The fine-tuned `yolo26n_*` detector weights
are derived from the same stack and are therefore not distributed here either.

If you need this under permissive terms, the detector is the only AGPL coupling and it is
shallow — two call sites (`fisheye_env.py`, `evaluation/metrics.py`) behind a
`predict(frame) -> boxes` boundary. Decoupling it behind an interface, with the Ultralytics
adapter as an optional dependency the user installs, would leave the environments and the
evaluation harness free to carry a permissive licence. Contributions doing that are welcome.

Note that the **evaluation harness itself** (`evaluation/`) has no detector dependency beyond
the oracle call and is the part most likely to be reusable elsewhere.

## Citation

Paper under review; citation block will be added on acceptance.

## Status

Source and evaluation harness are published here as the artefact accompanying the submission,
synced 2026-09-17 to match the submitted version of the paper — this adds the LOAF evaluation
(`evaluation/loaf.py`, `run_loaf.py`), the exact rendered-frustum overlap metric, the full-map
observation control (`--full_map`), the MP-DQN agent under `models/agent/`, and every driver
script behind the paper's tables.

The LOAF dataset itself is not redistributed here; obtain it from its authors and point
`--loaf_root` at it. Figures and the run artefacts the tables were computed from are being
prepared for release.
