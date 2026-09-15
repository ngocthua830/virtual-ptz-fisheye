"""Scene-level state shared across an episode.

The PTZ env's per-step reward only sees the current view. Once the agent
finds someone, the greedy policy stays parked because moving costs `-0.5` per
empty step. This module adds *scene-level* memory so the reward can credit
the agent for visiting parts of the fisheye disk that haven't been looked at
recently — breaking the local-optimum at its root.

Design notes (v2)
-----------------
The v1 CoverageMap used a wide rectangular slab (±fov/2 in both axes) as the
"in view" test and uniform per-bin weights. With M=4 polar bins of 20° each
and a zoom=1 view spanning 90° polar, one visit marked all 4 polar bins fresh
at every azimuth — so the agent saw zero coverage incentive to ever raise tilt
to look toward the horizon, and the policy parked at tilt=30°.

v2 fixes both:
  * Finer polar grid (M=8 bands × 10° each) plus a tighter "view core" check
    (sector marked only if its centre lies within ``core_frac · fov`` of the
    camera axis, default ``core_frac=0.33`` → 30° at zoom=1, so a visit covers
    ~5 of 8 polar bins instead of all 4 of 4).
  * Peripheral weighting: bins closer to the rim contribute ``peripheral_weight``
    times more to the staleness reward (default 3.0 at the outermost bin,
    linear ramp from 1.0 at nadir). This makes "look toward the horizon" a
    rewarding action even before any detection is found there.

Two collaborators:
  - CoverageMap   — K×M sector freshness grid over the fisheye hemisphere.
                    `visit(pan, tilt, zoom)` marks newly-seen sectors and
                    returns the staleness sum reward.
  - TrackRegistry — per-episode set of seen BoT-SORT track_ids with last-seen
                    timestep (extends the previous `seen_track_ids` set).
"""

from __future__ import annotations

import numpy as np


# Cover the fisheye hemisphere as a (K azimuth × M polar) grid.
DEFAULT_K = 16                  # finer azimuth than v1 (was 12)
DEFAULT_M = 8                   # finer polar than v1 (was 4)
DEFAULT_DECAY = 0.04            # freshness *= exp(-decay) per step
DEFAULT_CORE_FRAC = 0.33        # bin centre must lie within core_frac·fov of cam axis
DEFAULT_PERIPHERAL_W = 3.0      # weight of outermost polar bin vs nadir bin
POLAR_MAX_DEG = 80.0            # matches FisheyePTZEnvironment.tilt_range upper


class CoverageMap:
    """K×M sector grid over the fisheye hemisphere with exponential freshness."""

    def __init__(self, n_azimuth: int = DEFAULT_K,
                 n_polar: int = DEFAULT_M,
                 decay_per_step: float = DEFAULT_DECAY,
                 polar_max_deg: float = POLAR_MAX_DEG,
                 core_frac: float = DEFAULT_CORE_FRAC,
                 peripheral_weight: float = DEFAULT_PERIPHERAL_W):
        self.K = int(n_azimuth)
        self.M = int(n_polar)
        self.decay = float(decay_per_step)
        self.polar_max = float(polar_max_deg)
        self.core_frac = float(core_frac)
        self.peripheral_weight = float(peripheral_weight)
        self._neg_inf = -1.0e9
        self.last_visit = np.full((self.K, self.M), self._neg_inf, dtype=np.float32)
        self.step = 0

        # Pre-compute bin centres (degrees).
        self._az_step = 360.0 / self.K
        self._pol_step = self.polar_max / self.M
        self._az_centres = (np.arange(self.K) + 0.5) * self._az_step - 180.0
        self._pol_centres = (np.arange(self.M) + 0.5) * self._pol_step

        # Per-polar-bin weight: linear ramp from 1.0 (nadir) → peripheral_weight (rim).
        if self.M > 1:
            self._polar_w = np.linspace(1.0, self.peripheral_weight, self.M, dtype=np.float32)
        else:
            self._polar_w = np.array([1.0], dtype=np.float32)
        self._weight_grid = np.broadcast_to(self._polar_w[None, :], (self.K, self.M))

    # ----------------------------------------------------------------- API

    def reset(self) -> None:
        self.last_visit[:] = self._neg_inf
        self.step = 0

    def tick(self) -> None:
        """Advance the internal clock by one step (called once per env.step)."""
        self.step += 1

    def freshness(self) -> np.ndarray:
        """(K, M) freshness in [0, 1]; 1 == just-visited, 0 == never-visited."""
        age = np.maximum(0, self.step - self.last_visit)
        return np.exp(-self.decay * age, dtype=np.float32)

    def sectors_in_view(self, pan: float, tilt: float, zoom: float,
                        base_fov: float = 90.0) -> np.ndarray:
        """Boolean (K, M) mask of sectors whose centre lies within the view core.

        We don't use the full ±fov/2 slab — that's so wide one zoom=1 visit
        marks all polar bins fresh, leaving no incentive to tilt. Instead the
        bin centre must be within ±(core_frac · fov) of the camera axis.
        """
        fov = base_fov / max(zoom, 1e-6)
        half = fov * self.core_frac
        # Always credit at least the bin containing the camera axis (so highly
        # zoomed views — where the core is narrower than one bin — aren't free).
        az_half = max(half, self._az_step / 2.0)
        pol_half = max(half, self._pol_step / 2.0)

        az_diff = (self._az_centres - pan + 180.0) % 360.0 - 180.0
        az_mask = np.abs(az_diff) <= az_half                       # (K,)

        pol_diff = np.abs(self._pol_centres - tilt)
        pol_mask = pol_diff <= pol_half                            # (M,)

        return az_mask[:, None] & pol_mask[None, :]                # (K, M)

    def visit(self, pan: float, tilt: float, zoom: float,
              base_fov: float = 90.0) -> tuple[float, int]:
        """Mark sectors visited; return (weighted_staleness_reward, n_sectors_in_view).

        Reward = Σ_sec_in_view  weight[sec] * (1 − freshness_before_visit[sec]).
        The peripheral weight ramp gives the agent a stronger pull toward the
        outer polar bins, where the FOV reaches the disk rim.
        """
        mask = self.sectors_in_view(pan, tilt, zoom, base_fov)
        before = self.freshness()
        reward = float(((1.0 - before) * self._weight_grid)[mask].sum())
        self.last_visit[mask] = float(self.step)
        return reward, int(mask.sum())

    # ------------------------------------------------------------ summary

    def summary(self) -> dict[str, float | list[float]]:
        f = self.freshness()
        stale = 1.0 - f
        # Compress polar bands into 4 super-bins so the agent's state vector
        # has a fixed shape regardless of M.
        polar_band_means = stale.mean(axis=0)                                # (M,)
        if self.M >= 4:
            band = self.M // 4
            super_polar = [float(polar_band_means[i * band:(i + 1) * band].mean())
                           for i in range(4)]
        else:
            super_polar = polar_band_means.tolist() + [0.0] * (4 - self.M)
        # Peripheral focus: staleness in outer half of polar bins.
        outer_stale = float(polar_band_means[self.M // 2:].mean()) if self.M > 1 else 0.0
        return {
            'stale_total': float(stale.mean()),
            'polar_staleness': super_polar,            # always length 4
            'azimuth_staleness': stale.mean(axis=1).tolist(),  # length K (debug)
            'covered_frac': float((f > 0.3).mean()),
            'outer_stale': outer_stale,
        }


class TrackRegistry:
    """Per-episode track_id → last-seen-step lookup."""

    def __init__(self) -> None:
        self.last_seen: dict[int, int] = {}

    def reset(self) -> None:
        self.last_seen.clear()

    def update(self, detections, step: int) -> int:
        """Add any new ids; return count of brand-new ids in this batch."""
        novel = 0
        for d in detections:
            tid = int(d.get('track_id', -1))
            if tid == -1:
                continue
            if tid not in self.last_seen:
                novel += 1
            self.last_seen[tid] = step
        return novel

    def __len__(self) -> int:
        return len(self.last_seen)


if __name__ == '__main__':
    cm = CoverageMap()
    cm.reset()
    print('initial summary:', cm.summary())
    # Mark a slab at pan=0, tilt=30
    r, n = cm.visit(0.0, 30.0, 1.0)
    print(f'after visit (pan=0, tilt=30, z=1): r={r:.2f} n={n}')
    cm.tick(); cm.tick(); cm.tick()
    print('summary after 3 ticks:', cm.summary())
    r, n = cm.visit(180.0, 60.0, 1.0)
    print(f'after visit (pan=180, tilt=60): r={r:.2f} n={n}')
    print('final summary:', cm.summary())
