"""Dual-PTZ fisheye environment.

Two virtual perspective cameras share a single fisheye stream:

* TRACKER  — pursues whichever person it locked onto first. Reward favors
  detection continuity (keeping the same BoT-SORT track_id) and on-screen
  centering.
* EXPLORER — actively looks elsewhere. Reward favors discovering NEW track_ids
  (never seen this episode) and penalises azimuthal overlap with the tracker
  so it doesn't redundantly cover the same patch of the room.

The two cameras also share a single CoverageMap so that *joint* scene coverage
becomes the implicit team objective: a sector visited by the tracker is fresh
for the explorer too (so revisiting it pays nothing), and vice-versa.
Visiting order is tracker-first each step so the explorer's coverage reward is
computed after the tracker has marked its slice.

Each step expects a 4-tuple of agent decisions
    (action_tracker, param_tracker, action_explorer, param_explorer)
and returns
    (state_pair, reward_pair, done, info)
where state_pair / reward_pair are length-2 tuples in (tracker, explorer) order.
"""

import os

import cv2
import gym
import numpy as np

from .fisheye_env import FisheyePTZEnvironment
from .scene_state import CoverageMap, TrackRegistry


# Same 7-action scheme as the single-camera env.
NUM_ACTIONS = 7


def _wrap_pan(p):
    if p > 180.0:
        return p - 360.0
    if p < -180.0:
        return p + 360.0
    return p


def _ang_diff(a, b):
    """Smallest absolute angular difference in degrees, in [0, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


class DualFisheyePTZEnvironment(gym.Env):
    """Two-camera fisheye PTZ env: tracker + explorer."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}

        # Reuse the single-cam env purely as a host for video reading, YOLO and
        # the projection helper. We never call its step()/reset() rewards.
        self._host = FisheyePTZEnvironment(self.config)

        # ---------------- per-camera PTZ state ----------------
        self.pan = np.zeros(2, dtype=np.float32)
        self.tilt = np.array([30.0, 30.0], dtype=np.float32)
        self.zoom = np.array([1.0, 1.0], dtype=np.float32)
        # Start the explorer 180° away from the tracker so they diverge fast.
        self.pan[1] = 180.0

        self.pan_range = self._host.pan_range
        self.tilt_range = self._host.tilt_range
        self.zoom_range = self._host.zoom_range
        self.pan_speed = self._host.pan_speed
        self.tilt_speed = self._host.tilt_speed
        self.zoom_speed = self._host.zoom_speed

        # ---------------- bookkeeping ----------------
        self.current_step = 0
        self.max_steps = int(self.config.get('max_steps', 128))
        # state_dim 29 = 12 per-camera + 12 scene/pair + 5 motion/zoom-control features.
        self.state_dim = int(self.config.get('state_dim', 29))
        # Give the learned policy the full coverage grid instead of aggregates.
        self.full_map = bool(self.config.get('full_map', False))
        # A 157-D policy evaluated in a 29-D-aggregates environment would silently
        # receive a zero map tail and look far worse than it is; refuse that.
        if self.state_dim > 29 and not self.full_map:
            raise ValueError(
                f'state_dim={self.state_dim} > 29 but full_map=False: the extra '
                'dimensions would be all zeros. Pass full_map=True to match how '
                'such a checkpoint was trained.')
        self.num_actions = NUM_ACTIONS

        # K = number of rendered+detected crops per source frame. K=2 is the
        # tracker+explorer pair the paper uses throughout; K=1 is the tighter
        # sensing budget, a single agent with no partner to coordinate with.
        self.n_cams = int(self.config.get('n_cams', 2))
        if self.n_cams not in (1, 2):
            raise ValueError(f'n_cams must be 1 or 2, got {self.n_cams}')

        self.action_space = gym.spaces.MultiDiscrete([NUM_ACTIONS] * self.n_cams)
        self.observation_space = gym.spaces.Box(
            low=-2.0, high=10.0, shape=(self.n_cams, self.state_dim),
        )

        # Per-camera last_action / reward / detection history (length 2)
        self.last_action = [0, 0]
        self.last_reward = [0.0, 0.0]
        self.reward_history = [[], []]
        self.detections_history = [[], []]
        self.last_detections = [[], []]
        self.target_id = None              # tracker's locked id
        self.seen_track_ids = set()        # for explorer novelty bonus

        # Overlap penalty kicks in below this azimuth separation (degrees).
        self.overlap_thresh_deg = float(
            self.config.get('overlap_thresh_deg', 40.0)
        )

        # ---------------- shared scene-level memory ----------------
        # Both cameras visit the same CoverageMap; tracker visits first each
        # step, then explorer. This makes "joint coverage" the implicit team
        # objective without a centralised critic.
        # peripheral_weight is an ablation knob (uniform=1.0 vs peripheral=3.0).
        self.scene = CoverageMap(
            peripheral_weight=float(self.config.get('peripheral_weight', 3.0))
        )
        self.tracks = TrackRegistry()
        self.coverage_weight = float(self.config.get('coverage_weight', 0.5))
        # Explorer novelty-bonus weight (ablation: 0.0 disables the new-identity reward).
        self.novelty_weight = float(self.config.get('novelty_weight', 2.0))
        # Per-step bookkeeping for state vectors / info dict.
        self.last_cov_reward = [0.0, 0.0]

        # ---------------- motion-aware tracking + zoom-control knobs ----------
        cfg = self.config
        self.motion_alpha = float(cfg.get('motion_alpha', 1.0))   # mover priority weight
        self.v_ref_deg = float(cfg.get('v_ref_deg', 2.5))         # deg/step -> v_norm 1.0
        self.v_static_deg = float(cfg.get('v_static_deg', 0.5))   # below = "still"
        self.static_patience = int(cfg.get('static_patience', 12))
        self.static_tau = float(cfg.get('static_tau', 6.0))       # reward-decay timescale
        self.mover_thresh = float(cfg.get('mover_thresh', 0.4))   # v_norm to count as mover
        self.relock_cooldown_steps = int(cfg.get('relock_cooldown_steps', 8))
        self.m_lo = float(cfg.get('m_lo', 0.5))                   # motion_factor floor
        self.m_hi = float(cfg.get('m_hi', 1.5))                   # motion_factor ceil
        self.area_lo = float(cfg.get('area_lo', 0.12))            # framing band (view frac)
        self.area_hi = float(cfg.get('area_hi', 0.30))
        self.z_frame = float(cfg.get('z_frame', 0.6))             # in-band framing reward
        self.z_fov = float(cfg.get('z_fov', 0.4))                 # wide-FOV reward
        self.z_overzoom = float(cfg.get('z_overzoom', 1.5))       # over-zoom penalty gain
        self.pulse_period = int(cfg.get('pulse_period', 25))      # forced context-scan period
        self.widen_target_zoom = float(cfg.get('widen_target_zoom', 2.0))
        self.lockloss_zoom_decay = float(cfg.get('lockloss_zoom_decay', 0.85))

        # ---------------- motion bookkeeping (tracker / cam-0 only) -----------
        self._bearing_hist = {}      # track_id -> (az, el, step)
        self._track_vel = {}         # track_id -> EMA deg/step
        self.static_steps = 0        # consecutive still steps for the locked target
        self._relock_cooldown = {}   # track_id -> step until which re-lock is blocked
        self.steps_since_widen = 0
        self._lock_released = False  # for info / viz narration
        self._pulse_active = False

    # ----------------------------------------------------------- gym API

    def reset(self):
        self.current_step = 0
        self.pan = np.array([0.0, 180.0], dtype=np.float32)
        self.tilt = np.array([30.0, 30.0], dtype=np.float32)
        self.zoom = np.array([1.0, 1.0], dtype=np.float32)
        self.last_action = [0, 0]
        self.last_reward = [0.0, 0.0]
        self.reward_history = [[], []]
        self.detections_history = [[], []]
        self.last_detections = [[], []]
        self.target_id = None
        self.seen_track_ids = set()
        self._host.reset_tracks()
        self.scene.reset()
        self.tracks.reset()
        self.last_cov_reward = [0.0, 0.0]
        self._bearing_hist = {}
        self._track_vel = {}
        self.static_steps = 0
        self._relock_cooldown = {}
        self.steps_since_widen = 0
        self._lock_released = False
        self._pulse_active = False

        # Drive the host env's reset to open a (random) clip and seed fisheye R.
        self._host.reset()

        frame = self._host._get_frame()
        for i in range(self.n_cams):
            view = self._host.project_view(frame, self.pan[i], self.tilt[i], self.zoom[i])
            self.last_detections[i] = self._host._detect_objects(
                view, pose=(float(self.pan[i]), float(self.tilt[i]), float(self.zoom[i])),
                step=self.current_step)
            for d in self.last_detections[i]:
                tid = d.get('track_id', -1)
                if tid != -1:
                    self.seen_track_ids.add(tid)

        # Seed bearings, then lock onto the highest-priority detection. No motion
        # history exists yet, so v_norm is 0 and this reduces to highest-confidence.
        self._update_motion(self.last_detections[0])
        self.target_id = self._select_target(self.last_detections[0])

        return self._get_states()

    def step(self, actions, params=(1.0, 1.0)):
        """actions: (a_tracker, a_explorer); params: (p_tracker, p_explorer).
        At K=1 both are length-1 and only camera 0 exists."""
        actions = list(actions)[:self.n_cams]
        params = list(params)[:self.n_cams]
        self.current_step += 1
        self._host.current_frame += self._host.frame_skip
        self.scene.tick()

        for i, (a, p) in enumerate(zip(actions, params)):
            self._apply_action(i, int(a), float(p))

        frame = self._host._get_frame()
        views = []
        for i in range(self.n_cams):
            v = self._host.project_view(frame, self.pan[i], self.tilt[i], self.zoom[i])
            views.append(v)
            self.last_detections[i] = self._host._detect_objects(
                v, pose=(float(self.pan[i]), float(self.tilt[i]), float(self.zoom[i])),
                step=self.current_step)

        # Update per-track world-bearing velocities from the tracker (cam-0) view.
        self._update_motion(self.last_detections[0])

        # ---- Shared scene-state updates (tracker first, then explorer) ----
        # Visiting order matters: a sector marked fresh by the tracker pays
        # nothing to the explorer. That's the collaboration mechanism.
        for i in range(self.n_cams):
            cov_raw, n_sec = self.scene.visit(
                float(self.pan[i]), float(self.tilt[i]), float(self.zoom[i]),
                base_fov=self._host.base_fov,
            )
            self.last_cov_reward[i] = cov_raw / max(n_sec, 1)
        # Joint track registry update (used by explorer novelty + state).
        det_all = [d for i in range(self.n_cams) for d in self.last_detections[i]]
        self.tracks.update(det_all, self.scene.step)

        if self.n_cams == 1:
            # One agent, one budget: it must both follow and discover, so it
            # receives BOTH reward terms on its single view. The anti-overlap
            # penalty is skipped inside _explorer_reward (no partner to avoid).
            rewards = [self._tracker_reward(self.last_detections[0], int(actions[0]))
                       + self._explorer_reward(self.last_detections[0], int(actions[0]))]
        else:
            rewards = [
                self._tracker_reward(self.last_detections[0], int(actions[0])),
                self._explorer_reward(self.last_detections[1], int(actions[1])),
            ]
        for d in det_all:
            tid = d.get('track_id', -1)
            if tid != -1:
                self.seen_track_ids.add(tid)

        for i in range(self.n_cams):
            self.last_action[i] = int(actions[i])
            self.last_reward[i] = rewards[i]
            self.reward_history[i].append(rewards[i])
            self.detections_history[i].append(len(self.last_detections[i]))

        states = self._get_states()
        done = self.current_step >= self.max_steps
        # In no-loop (evaluation) mode, also terminate at the true end of the clip
        # so each clip is observed exactly once (no video wrap-around inflating
        # discovery/coverage over a replayed window).
        if getattr(self._host, '_clip_ended', False):
            done = True

        info = {
            'frame': frame,
            'views': views,
            'detections': self.last_detections,
            'pan': self.pan.copy(),
            'tilt': self.tilt.copy(),
            'zoom': self.zoom.copy(),
            'target_id': self.target_id,
            'seen_ids': len(self.seen_track_ids),
            'rewards': tuple(rewards),
            'coverage_rewards': tuple(self.last_cov_reward),
            'stale_total': self.scene.summary()['stale_total'],
            'target_vel': self._target_vel(),
            'static_steps': self.static_steps,
            'lock_released': self._lock_released,
            'pulse_active': self._pulse_active,
        }
        return states, rewards, done, info

    # ------------------------------------------------------- motion estimation

    def _update_motion(self, cam0_dets):
        """Update per-track world-bearing velocities from the tracker's view.

        Velocity is the angular displacement of a track's world bearing between
        consecutive sightings (deg/step), EMA-smoothed. Because world bearing is
        camera-pose-invariant, a person the camera pans to follow registers ~0
        velocity while someone physically moving registers a large one.
        """
        step = self.current_step
        for d in cam0_dets:
            tid = d.get('track_id', -1)
            if tid == -1:
                continue
            az, el = self._host.world_bearing(
                d['bbox'], float(self.pan[0]), float(self.tilt[0]),
                float(self.zoom[0]), self._host.ptz_out_w, self._host.ptz_out_h,
                self._host.base_fov,
            )
            prev = self._bearing_hist.get(tid)
            if prev is not None:
                paz, pel, pstep = prev
                dt = max(step - pstep, 1)
                daz = _ang_diff(az, paz)
                deg = float(np.sqrt(daz * daz + (el - pel) ** 2)) / dt
                old = self._track_vel.get(tid, deg)
                self._track_vel[tid] = 0.5 * old + 0.5 * deg   # EMA
            self._bearing_hist[tid] = (az, el, step)

        # Forget tracks not seen for a while (keeps the dicts bounded).
        stale = [t for t, (_, _, s) in self._bearing_hist.items() if step - s > 30]
        for t in stale:
            self._bearing_hist.pop(t, None)
            self._track_vel.pop(t, None)

    def _vel_norm(self, tid):
        """Normalised motion magnitude in [0, 1] for a track id."""
        if tid is None or tid == -1:
            return 0.0
        return float(np.clip(self._track_vel.get(tid, 0.0) / self.v_ref_deg, 0.0, 1.0))

    def _target_vel(self):
        """Raw deg/step velocity of the currently locked target (for info/viz)."""
        if self.target_id in (None, -1):
            return 0.0
        return float(self._track_vel.get(self.target_id, 0.0))

    def _select_target(self, dets):
        """Pick the highest-priority detection: confidence boosted by motion,
        excluding ids on re-lock cooldown. Returns a track_id (or -1)."""
        step = self.current_step
        cands = [
            d for d in dets
            if self._relock_cooldown.get(d.get('track_id', -1), -1) < step
        ] or dets
        if not cands:
            return None
        best = max(
            cands,
            key=lambda d: d['confidence'] * (
                1.0 + self.motion_alpha * self._vel_norm(d.get('track_id', -1))
            ),
        )
        return best.get('track_id', -1)

    # ----------------------------------------------------------- actions

    def _apply_action(self, cam, action, param):
        param = float(np.clip(param, 0.0, 1.0))
        if action == 0:                       # STAY
            pass
        elif action == 1:                     # PAN LEFT
            self.pan[cam] = _wrap_pan(self.pan[cam] - self.pan_speed * param)
        elif action == 2:                     # PAN RIGHT
            self.pan[cam] = _wrap_pan(self.pan[cam] + self.pan_speed * param)
        elif action == 3:                     # ZOOM IN
            self.zoom[cam] = float(min(self.zoom_range[1],
                                       self.zoom[cam] + self.zoom_speed * param))
        elif action == 4:                     # ZOOM OUT
            self.zoom[cam] = float(max(self.zoom_range[0],
                                       self.zoom[cam] - self.zoom_speed * param))
        elif action == 5:                     # TILT UP
            self.tilt[cam] = float(min(self.tilt_range[1],
                                       self.tilt[cam] + self.tilt_speed * param))
        elif action == 6:                     # TILT DOWN
            self.tilt[cam] = float(max(self.tilt_range[0],
                                       self.tilt[cam] - self.tilt_speed * param))

        # Tracker zoom hygiene: auto-widen toward 1x while it has no lock so the
        # FOV reopens to re-acquire, and run the periodic context-scan counter.
        if cam == 0:
            if self.target_id in (None, -1):
                self.zoom[0] = float(max(self.zoom_range[0],
                                         self.zoom[0] * self.lockloss_zoom_decay))
            self.steps_since_widen += 1
            if self.zoom[0] < self.widen_target_zoom:
                self.steps_since_widen = 0

    # ----------------------------------------------------------- rewards

    def _tracker_reward(self, detections, action):
        """Detect + centre + continuity, scaled by the locked target's MOTION
        (movers pay more; statics decay and time out), plus zoom regulation."""
        reward = 0.0
        self._lock_released = False
        step = self.current_step

        if detections:
            avg_conf = float(np.mean([d['confidence'] for d in detections]))
            reward += avg_conf * 2.0

            w, h = self._host.ptz_out_w, self._host.ptz_out_h
            cx, cy = w / 2.0, h / 2.0
            max_dist = np.sqrt(cx * cx + cy * cy)

            # (Re-)acquire a lock, preferring movers, if we have none.
            if self.target_id in (None, -1):
                self.target_id = self._select_target(detections)
                self.static_steps = 0

            target_det = next(
                (d for d in detections if d.get('track_id') == self.target_id), None
            )

            # Motion factor from the locked target's velocity: a moving target
            # is worth more to keep framed than a stationary one.
            v_norm = self._vel_norm(self.target_id)
            motion_factor = self.m_lo + (self.m_hi - self.m_lo) * v_norm

            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                dx_ = (x1 + x2) / 2.0 - cx
                dy_ = (y1 + y2) / 2.0 - cy
                center_score = 1.0 - (np.sqrt(dx_ * dx_ + dy_ * dy_) / max_dist)
                reward += float(center_score) * 0.3 * motion_factor
            reward += 1.0

            if target_det is not None:
                reward += 1.0 * motion_factor          # continuity, motion-scaled

                # ---- static decay + timeout: deprioritise still people --------
                if self._track_vel.get(self.target_id, 0.0) < self.v_static_deg:
                    self.static_steps += 1
                else:
                    self.static_steps = 0
                reward *= float(np.exp(-self.static_steps / self.static_tau))

                # After patience, hand off to a genuine mover if one exists.
                mover = next(
                    (d for d in detections
                     if d.get('track_id') not in (self.target_id, -1)
                     and self._vel_norm(d.get('track_id', -1)) > self.mover_thresh
                     and self._relock_cooldown.get(d.get('track_id', -1), -1) < step),
                    None,
                )
                if self.static_steps > self.static_patience and mover is not None:
                    self._relock_cooldown[self.target_id] = step + self.relock_cooldown_steps
                    self.target_id = mover.get('track_id', -1)
                    self.static_steps = 0
                    self._lock_released = True
                    reward += 0.5                      # bonus for switching to the mover
            else:
                self.target_id = None                  # locked id not in view; re-acquire next step

            reward += self._zoom_reward(target_det, action)
        else:
            reward -= 0.5
            self.target_id = None                      # lost lock; _apply_action auto-widens
            self.static_steps = 0
            reward += self._zoom_reward(None, action)

        if action != self.last_action[0]:
            reward -= 0.05
        # Tracker also gets a (small) shared-coverage credit so it doesn't
        # sit still while the explorer does all the scene work.
        reward += self.coverage_weight * 0.5 * self.last_cov_reward[0]
        return float(reward)

    def _zoom_reward(self, target_det, action):
        """Zoom regulation: wide-FOV incentive + frame-to-band + forced context
        pulse. Returns the zoom-shaping contribution to the tracker reward."""
        z = float(self.zoom[0])
        r = self.z_fov * (1.0 / z)                     # wider FOV = small bonus

        if target_det is not None:
            w, h = self._host.ptz_out_w, self._host.ptz_out_h
            x1, y1, x2, y2 = target_det['bbox']
            area = ((x2 - x1) * (y2 - y1)) / (w * h)
            if area > self.area_hi:                    # over-zoomed / too close
                r -= self.z_overzoom * (area - self.area_hi)
            elif area < self.area_lo:                  # too small; allow zoom-in
                r -= 0.3 * (self.area_lo - area)
            else:                                      # well framed
                r += self.z_frame

        # Forced context pulse: after a long stretch pinned narrow, pay to widen.
        self._pulse_active = self.steps_since_widen > self.pulse_period
        if self._pulse_active:
            if action == 4:                            # ZOOM OUT
                r += 1.0
            elif z >= self.widen_target_zoom:          # still refusing to widen
                r -= 0.5
        return float(r)

    def _explorer_reward(self, detections, action):
        """Novelty + anti-overlap with tracker."""
        reward = 0.0
        if detections:
            avg_conf = float(np.mean([d['confidence'] for d in detections]))
            reward += avg_conf * 1.0

            w, h = self._host.ptz_out_w, self._host.ptz_out_h
            cx, cy = w / 2.0, h / 2.0
            max_dist = np.sqrt(cx * cx + cy * cy)

            novel = 0
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                dx_ = (x1 + x2) / 2.0 - cx
                dy_ = (y1 + y2) / 2.0 - cy
                reward += (1.0 - np.sqrt(dx_ * dx_ + dy_ * dy_) / max_dist) * 0.2
                tid = d.get('track_id', -1)
                if tid != -1 and tid not in self.seen_track_ids and tid != self.target_id:
                    novel += 1
            reward += self.novelty_weight * min(novel, 5)  # novelty bonus (capped vs noisy tracker)
            reward += 0.5                     # base "saw someone" credit
        else:
            reward -= 0.5

        # Anti-overlap with tracker: penalise small azimuth separation.
        # Undefined at K=1 -- there is no partner view to be redundant with.
        if self.n_cams > 1:
            sep = _ang_diff(self.pan[1], self.pan[0])
            if sep < self.overlap_thresh_deg:
                reward -= (self.overlap_thresh_deg - sep) / self.overlap_thresh_deg

        if action != self.last_action[self.n_cams - 1]:
            reward -= 0.05
        # Explorer is the primary coverage agent — gets the full shared-map
        # staleness credit. After the tracker visits first, any sector still
        # stale is "uncovered" — perfect for the explorer to claim.
        reward += self.coverage_weight * self.last_cov_reward[self.n_cams - 1]
        return float(reward)

    # ----------------------------------------------------------- state

    def _get_states(self):
        s = np.zeros((self.n_cams, self.state_dim), dtype=np.float32)
        for i in range(self.n_cams):
            s[i] = self._cam_state(i)
        return s

    def _cam_state(self, i):
        dets = self.last_detections[i]
        num_dets = self.detections_history[i][-1] if self.detections_history[i] else 0
        avg_dets = float(np.mean(self.detections_history[i][-10:])) if self.detections_history[i] else 0.0
        reward_trend = float(np.mean(self.reward_history[i][-5:])) if self.reward_history[i] else 0.0

        rel_x, rel_y, rel_area, best_conf = -2.0, -2.0, -1.0, 0.0
        target_det = None
        if dets and i == 0 and self.target_id not in (None, -1):
            for d in dets:
                if d.get('track_id') == self.target_id:
                    target_det = d
                    break
        if target_det is None and dets:
            target_det = max(dets, key=lambda d: d['confidence'])

        if target_det:
            x1, y1, x2, y2 = target_det['bbox']
            w, h = self._host.ptz_out_w, self._host.ptz_out_h
            rel_x = ((x1 + x2) / 2.0 - w / 2.0) / (w / 2.0)
            rel_y = ((y1 + y2) / 2.0 - h / 2.0) / (h / 2.0)
            rel_area = ((x2 - x1) * (y2 - y1)) / (w * h)
            best_conf = target_det['confidence']

        # No partner at K=1: the pair-geometry features are held at zero so the
        # 29-D layout (and every checkpoint shape) is unchanged.
        sep = _ang_diff(self.pan[1], self.pan[0]) if self.n_cams > 1 else 0.0

        s = np.zeros(self.state_dim, dtype=np.float32)
        s[0] = num_dets / 5.0
        s[1] = avg_dets / 5.0
        s[2] = self.pan[i] / 180.0
        s[3] = self.tilt[i] / 80.0
        s[4] = self.zoom[i] / 8.0
        s[5] = rel_x
        s[6] = rel_y
        s[7] = rel_area
        s[8] = best_conf
        s[9] = self.last_action[i] / 7.0
        s[10] = reward_trend / 5.0
        s[11] = self.current_step / max(1, self.max_steps)

        # Scene-level features [12:24] — joint coverage map + pair geometry.
        if self.state_dim > 12:
            summary = self.scene.summary()
            s[12] = sep / 180.0
            s[13] = 1.0 if i == 0 else 0.0           # role flag (tracker=1, explorer=0)
            s[14] = summary['stale_total']
            for k, v in enumerate(summary['polar_staleness'][:4]):
                s[15 + k] = float(v)                 # s[15..18] polar super-bands
            s[19] = summary['outer_stale']           # peripheral pull
            s[20] = summary['covered_frac']
            s[21] = min(len(self.tracks), 20) / 20.0
            s[22] = float(np.tanh(self.last_cov_reward[i]))
            # Each agent sees its OWN coverage gain, plus partner's pan (so it
            # can model where the other one is right now).
            s[23] = (self.pan[1 - i] / 180.0) if self.n_cams > 1 else 0.0

        # Motion / zoom-control features [24:29] — drive the tracker's
        # prefer-movers and zoom-regulation behaviour. The explorer (separate
        # network) gets benign zeros for the tracker-only signals.
        if self.state_dim > 24:
            band_center = (self.area_lo + self.area_hi) / 2.0
            s[26] = float(rel_area - band_center)        # framing error (both cams)
            s[27] = 1.0 / max(float(self.zoom[i]), 1e-6)  # FOV width fraction
            if i == 0:
                s[24] = self._vel_norm(self.target_id)    # locked-target motion
                s[25] = min(self.static_steps / max(self.static_patience, 1), 1.0)
                s[28] = min(self.steps_since_widen / max(self.pulse_period, 1), 1.0)

        # Optional: append the FULL K x M coverage map (flattened) after the 29
        # engineered features. The hand-coded heuristic baseline reads this grid
        # directly while the learned policy otherwise sees only aggregates of it
        # (total/per-band staleness, covered fraction, own gain) -- a state
        # advantage in the baseline's favour. With `full_map` the two see the
        # same information, which is what makes the comparison fair.
        if self.full_map and self.state_dim > 29:
            fm = self.scene.freshness().reshape(-1)
            n = min(len(fm), self.state_dim - 29)
            s[29:29 + n] = fm[:n]
        return s

    # ----------------------------------------------------------- misc

    def close(self):
        self._host.close()


if __name__ == '__main__':
    env = DualFisheyePTZEnvironment({'max_steps': 6})
    states = env.reset()
    print('reset states shape', states.shape)
    for i in range(6):
        a = (env.action_space.nvec[0].item() // 2, env.action_space.nvec[1].item() // 2)
        s, r, d, info = env.step(a)
        print(f'step {i}: rT={r[0]:+.2f} rE={r[1]:+.2f} dets={[len(x) for x in info["detections"]]} '
              f'zoom={info["zoom"]} tgt_vel={info["target_vel"]:.2f} '
              f'static={info["static_steps"]} released={info["lock_released"]} '
              f'pulse={info["pulse_active"]} | state{s.shape}')
        if d:
            break
    env.close()
