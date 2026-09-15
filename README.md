# virtual-ptz-fisheye

Turning **one overhead fisheye camera into a bank of virtual PTZ views**, and asking the
question the paper is named after:

> **When does *learned* control of those views actually help, and when does simple geometry
> suffice?**

Code and evaluation harness for *"When Does Learned Virtual-PTZ Control Help? Geometric,
Learned, and Hybrid Policies from a Single Fisheye Camera"* (under review).

---

## The result, stated up front

**On the regime we studied, learned control does not help.** An *overlap-optimized geometric
sweep* — hold the two virtual cameras 180° apart in azimuth at a high tilt so their view cones
are disjoint by geometry, then co-rotate them to raster the hemisphere — beats the learned
policy on **every** metric at once:

| policy | discovery ↑ | time-to-detect ↓ | observed-time frac ↑ | solid-angle overlap ↓ |
|---|---|---|---|---|
| **Ovlp-Sweep (48°, no learning)** | **0.92** | **1.09** | **0.64** | **0.00** |
| RL (5 seeds) | 0.68 ± 0.06 | 2.62 ± 0.96 | 0.50 ± 0.10 | 0.08 ± 0.07 |
| Hybrid (5 seeds) | 0.72 ± 0.05 | 2.45 ± 1.01 | 0.51 ± 0.09 | 0.12 ± 0.02 |
| random | 0.75 | 2.84 | 0.50 | 0.15 |

Two honest readings we want to keep attached to those numbers:

- **A `random` policy also reaches 0.50 observed-time fraction.** On this scene the learned
  controller buys no sustained-observation advantage over acting at random; its only edge there
  is worst-case gap.
- **An earlier version of this work claimed the opposite.** A "2.8× faster, ties the sweep"
  headline turned out to be an artefact of (i) evaluating a single training seed and (ii) a
  video-looping bug that gave every policy ~5× its real observation window. Fixing both — five
  independent seeds as the unit of analysis, each clip observed exactly once — reversed the
  conclusion. That correction is why the evaluation harness in this repo looks the way it does.

The regime matters: two wide, non-overlapping views already blanket a small, fully-observable
room, so the prioritisation and zoom that active control buys have nothing to win. We expect the
answer to flip in scenes too large, dense or occluded for wide static views.

## Layout

```
environments/     virtual-PTZ gym environments
  fisheye_env.py      equidistant fisheye -> perspective unwarp at arbitrary pan/tilt/zoom
                      (project_view); FOV = base_fov / zoom
  dual_fisheye_env.py two cooperating virtual cameras + shared coverage map
  scene_state.py      track registry / scene bookkeeping
evaluation/       the reward-independent measurement harness
  metrics.py          discovery, time-to-detect, coverage%, inter-camera + solid-angle overlap,
                      observed-time fraction, max unobserved gap; oracle on the raw fisheye
  survival_ttd.py     censoring-aware latency (Kaplan-Meier, restricted-mean TTD)
  horizon_curve.py    discovery vs observation budget, fixed-cohort
  tilt_curve.py       tilt sensitivity of the geometric sweep
baselines/        non-learned policies: sweep, coordinated sweep, overlap-optimized sweep,
                  greedy, heuristic, random  (+ multi-seed evaluation drivers)
configs/          scenario configuration
scripts/          run/eval drivers used for the paper's tables
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

Source and evaluation harness are published here as the artefact accompanying the submission.
Figures, tables and the run artefacts they were computed from are being prepared for release.
