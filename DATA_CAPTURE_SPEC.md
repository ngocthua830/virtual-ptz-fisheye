# Data-capture spec — to make learned control actually win

**Why this exists.** On our current data (one small room, fully observable from the ceiling,
~2.4 min test), a *systematic coordinated sweep* saturates discovery: given time it visits
every sector regardless of where people are, so learning cannot out-discover it. Our honest
result is that learning only reduces redundancy. To get a regime where **learning wins on
discovery**, the camera must *not* be able to cover everything in time — so prioritization
pays. This spec describes the recordings that create that regime. **Record harder, not just
more.**

## The single most important variable: coverage-time pressure
A learned policy beats a sweep only when a full raster of the scene takes **longer than the
window in which people appear and matter**. Every item below serves that.

## What to record

| Dimension | Current | Target | Why it lets learning win |
|---|---|---|---|
| **Scene size / FoV to cover** | one small room | larger hall, or **multi-room** / L-shaped space | a single sweep can't raster it within the horizon → must prioritize |
| **Crowd density** | few people | **6–15+ simultaneous**, with mutual **occlusion** | people visible only briefly → being there at the right moment beats rastering |
| **Motion** | mixed | many **fast movers**, entries/exits, groups forming/splitting | rewards tracking + anticipation over uniform scanning |
| **Session length** | ~7 min total | **≥30 min train + ≥10 min test**, several distinct sessions | enough for real train/test generalization; more arrival events |
| **Distinct scenes/days** | 1 | **≥3 sessions** (different layouts/lighting/crowds) | split train/test **by session** → tests generalization, not memorization |
| **Camera** | 1 ceiling fisheye, 2320², 15 fps | **same** (keep the virtual-PTZ pipeline unchanged) | comparability with existing results |

## Quantities (minimum viable → comfortable)
- **Train:** ≥30 min across ≥2 sessions (comfortable: 60 min, 3 sessions).
- **Test:** ≥10 min across ≥1 *held-out* session (comfortable: 3 held-out sessions).
- Keep sessions physically separate so the train/test split is by session, never by clip
  within one recording (avoids the pseudo-replication we already had to fix).

## Annotation (addresses the pseudo-reference limitation)
- Our metrics currently use a detector-on-raw-fisheye *oracle*, not ground truth. For the
  **test set only**, human-annotate person tracks (bounding boxes + IDs) on a sampled subset
  (e.g. every 5th frame). Even ~500 annotated frames let us (a) report true discovery/latency
  and (b) quantify oracle bias. Tools: CVAT or Label Studio on the raw fisheye frames.

## Multi-camera note (for the 3+ virtual-camera direction)
No new hardware needed — extra "cameras" are extra virtual crops of the *same* fisheye. But
denser/larger scenes are what make ≥3 cameras worthwhile (a fixed sweep can't partition a
big busy scene as well as a learned allocator). So this spec also unblocks that experiment.

## What NOT to do
- Don't just record more of the same small, sparse room — it won't change the discovery
  result (the sweep already saturates it); it only reduces RL's overfitting.
- Don't loop or reuse clips across train/test.

## Success criterion
After retraining on the harder data, learning should **out-discover the coordinated sweep at
short-to-mid horizons** (not just reduce redundancy). If it does, the paper's thesis flips
from "sweep is a strong baseline learning matches on efficiency" to "learning wins where a
sweep cannot cover in time" — the positive result MMM rewards.
