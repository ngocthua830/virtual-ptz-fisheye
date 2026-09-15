"""
PTZ Camera Environment simulated from a SINGLE ceiling-mounted circular
fisheye video. Pan/tilt/zoom map to a virtual perspective camera whose
rays are warped back into the fisheye image via cv2.remap (equidistant
model: r_pix = f_fish * theta).

Conventions
-----------
- Fisheye optical axis points along -Z (camera looks straight down).
- Pan rotates the virtual camera azimuth about world Z (degrees, wraps mod 360).
- Tilt is the polar angle off the optical axis (0 = straight down,
  90 = horizon).
- Zoom narrows the virtual perspective FOV: fov = base_fov / zoom.
"""

import os
from pathlib import Path

import cv2
import gym
import numpy as np
import torch

from .scene_state import CoverageMap, TrackRegistry


class FisheyePTZEnvironment(gym.Env):
    """Gym environment: a virtual PTZ camera unwarped from a fisheye video."""

    def __init__(self, config=None):
        super().__init__()

        self.config = config or {}

        # ------------------ data ------------------
        # data_path may be a single file or a directory. Directories yield a
        # list of clips; reset() samples one per episode so the agent trains
        # across all available footage.
        default_paths = [
            self.config.get('data_path'),
            './data/',
            './data/1767931881294.mp4',
            '/data/',
            '/data/1767931881294.mp4',
        ]
        self.video_paths = []
        for p in default_paths:
            if p and os.path.exists(p):
                self.video_paths = self._resolve_videos(p)
                if self.video_paths:
                    break
        self.video_path = self.video_paths[0] if self.video_paths else None
        self._rng = np.random.default_rng(self.config.get('seed'))

        # ------------------ PTZ state ------------------
        self.pan = 0.0
        self.tilt = 30.0
        self.zoom = 1.0

        # ranges
        self.pan_range = (-180.0, 180.0)        # wraps
        self.tilt_range = (0.0, 80.0)           # 0 = straight down, 80 = near horizon
        self.zoom_range = (1.0, 8.0)

        # speeds (deg / units per discrete step at param=1.0)
        self.pan_speed = float(self.config.get('pan_speed', 25.0))
        self.tilt_speed = float(self.config.get('tilt_speed', 15.0))
        self.zoom_speed = float(self.config.get('zoom_speed', 0.5))

        # ------------------ fisheye intrinsics ------------------
        # Set once we open the video and detect the circle.
        self.fish_cx = None
        self.fish_cy = None
        self.fish_R = None                       # pixel radius of fisheye disk
        self.fish_fov_deg = float(self.config.get('fish_fov_deg', 180.0))

        # ------------------ virtual PTZ output ------------------
        self.ptz_out_w = int(self.config.get('ptz_out_w', 960))
        self.ptz_out_h = int(self.config.get('ptz_out_h', 540))
        self.base_fov = float(self.config.get('base_fov', 90.0))  # perspective FOV at zoom=1

        # Detection confidence threshold. A COCO YOLO needs a low threshold on
        # top-down poses (0.2). A top-view-fine-tuned OBB detector is confident
        # and over-fires at 0.2, so default it higher (0.4).
        default_conf = 0.4 if self.config.get('detector_weights') else 0.2
        self.yolo_conf = float(self.config.get('yolo_conf', default_conf))

        # ------------------ episode bookkeeping ------------------
        self.current_step = 0
        self.current_frame = 0
        self.frame_skip = int(self.config.get('frame_skip', 5))
        self.max_steps = int(self.config.get('max_steps', 128))
        # When False, the episode ends at the true end of the clip instead of
        # wrapping the video (used for a contamination-free evaluation: each clip
        # is observed exactly once). Training keeps the default (looping) behaviour.
        self.loop_video = bool(self.config.get('loop_video', True))
        self._clip_ended = False
        # state_dim 24 = 12 base + 12 scene (coverage summary + track registry)
        self.state_dim = int(self.config.get('state_dim', 24))

        # ------------------ scene-level memory ------------------
        # Coverage map: forces visits to stale sectors of the fisheye hemisphere.
        # Reward = beta * (Σ staleness collected this step), normalised by view size.
        self.scene = CoverageMap()
        self.tracks = TrackRegistry()
        self.coverage_weight = float(self.config.get('coverage_weight', 0.5))
        self.novelty_weight = float(self.config.get('novelty_weight', 1.0))

        # 7 discrete actions: STAY, PAN_LEFT, PAN_RIGHT, ZOOM_IN, ZOOM_OUT, TILT_UP, TILT_DOWN
        self.num_actions = 7
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Box(low=-2.0, high=10.0, shape=(self.state_dim,))

        # ------------------ runtime ------------------
        self.cap = None
        self.frame_count = 0
        self.detector = None
        self._init_detector()

        self.last_action = 0
        self.last_reward = 0.0
        self.reward_history = []
        self.detections_history = []
        self.last_detections = []
        self.target_id = None

        print(f"[FisheyePTZEnv] videos={len(self.video_paths)} "
              f"({', '.join(os.path.basename(v) for v in self.video_paths)})")

    # ----------------------------------------------------------- setup

    @staticmethod
    def _resolve_videos(path):
        p = Path(path)
        if p.is_file():
            return [str(p)]
        if p.is_dir():
            out = []
            for ext in ('*.mp4', '*.MP4', '*.mov', '*.OSV'):
                out.extend(sorted(p.glob(ext)))
            return [str(f) for f in out]
        return []

    def _init_detector(self):
        weights = self.config.get('detector_weights') or 'yolov8n.pt'
        self.detector_weights = weights
        self.detector_is_obb = False
        try:
            # Optional torchvision NMS CPU fallback (needed when torchvision is
            # CPU-only and torch is CUDA — see sibling fisheye_person_detect).
            try:
                import torchvision
                _orig_nms = torchvision.ops.nms
                def _nms_cpu_fallback(boxes, scores, iou_threshold):
                    if boxes.is_cuda:
                        keep = _orig_nms(boxes.cpu(), scores.cpu(), iou_threshold)
                        return keep.to(boxes.device)
                    return _orig_nms(boxes, scores, iou_threshold)
                torchvision.ops.nms = _nms_cpu_fallback
                try:
                    import torchvision.ops.boxes as _tv_boxes
                    _tv_boxes.nms = _nms_cpu_fallback
                except Exception:
                    pass
            except Exception:
                pass

            from ultralytics import YOLO
            self.detector = YOLO(weights)
            self.detector.to('cuda' if torch.cuda.is_available() else 'cpu')
            task = (getattr(self.detector, 'task', '') or '').lower()
            self.detector_is_obb = 'obb' in task
            print(f"[FisheyePTZEnv] detector loaded: {weights} "
                  f"(task={task!r}, obb={self.detector_is_obb})")
        except Exception as e:
            print(f"[FisheyePTZEnv] Warning: detector unavailable ({e})")
            self.detector = None

    def _detect_fisheye_circle(self, frame):
        """Estimate fisheye disk (cx, cy, R) from a frame. Assumes square image
        with the circle inscribed, which matches typical EZVIZ-style fisheyes."""
        h, w = frame.shape[:2]
        # Heuristic: the disk is inscribed in the shorter side.
        self.fish_cx = w / 2.0
        self.fish_cy = h / 2.0
        self.fish_R = min(w, h) / 2.0

    # ----------------------------------------------------------- gym API

    def reset(self):
        self.current_step = 0
        self.current_frame = 0
        self.pan = 0.0
        self.tilt = 30.0
        self.zoom = 1.0
        self.last_action = 0
        self.last_reward = 0.0
        self.reward_history = []
        self.detections_history = []
        self.last_detections = []
        self.target_id = None
        self.scene.reset()
        self.tracks.reset()
        self.last_cov_reward = 0.0
        self.last_novel = 0
        # Reset OBB tracker pool so ids don't leak across episodes.
        self._obb_tracks = {}
        self._obb_next_id = 1
        self._obb_step = 0

        self._clip_ended = False
        if self.cap is not None:
            self.cap.release()

        # Sample a clip for this episode if multiple are available. For deterministic,
        # clip-balanced evaluation, `forced_clip_idx` (if set) pins the clip so every
        # policy/seed is scored on the identical set of clips (each exactly once) rather
        # than a random draw that, with only a few test clips, dominates the metrics.
        if self.video_paths:
            fci = getattr(self, 'forced_clip_idx', None)
            if fci is not None:
                idx = int(fci) % len(self.video_paths)
            else:
                idx = int(self._rng.integers(0, len(self.video_paths)))
            self.video_path = self.video_paths[idx]

        if self.video_path and os.path.exists(self.video_path):
            self.cap = cv2.VideoCapture(self.video_path)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._next_read = 0  # decoder position for sequential reads (see _get_frame)
            print(f"[FisheyePTZEnv] opened {os.path.basename(self.video_path)}, "
                  f"frames={self.frame_count}")

        frame = self._get_frame()
        if self.fish_R is None and frame is not None:
            self._detect_fisheye_circle(frame)

        ptz_view = self._get_ptz_view(frame)
        self.last_detections = self._detect_objects(ptz_view)
        if self.last_detections:
            best = max(self.last_detections, key=lambda d: d['confidence'])
            self.target_id = best.get('track_id', -1)

        return self._get_state()

    def step(self, action, action_param=1.0):
        self.current_step += 1
        self.current_frame += self.frame_skip
        self.scene.tick()

        self._apply_action(action, action_param)

        frame = self._get_frame()
        ptz_view = self._get_ptz_view(frame)
        self.last_detections = self._detect_objects(ptz_view)

        # Scene-level updates BEFORE building reward / state so both can use them.
        cov_reward_raw, n_sectors = self.scene.visit(
            self.pan, self.tilt, self.zoom, base_fov=self.base_fov,
        )
        # Normalise so the scale is ~[0, 1] regardless of view size.
        self.last_cov_reward = cov_reward_raw / max(n_sectors, 1)
        self.last_novel = self.tracks.update(self.last_detections, self.scene.step)

        reward = self._compute_reward(self.last_detections, action)
        state = self._get_state()
        done = self.current_step >= self.max_steps

        info = {
            'num_detections': len(self.last_detections),
            'detections': self.last_detections,
            'ptz_view': ptz_view,
            'fisheye_frame': frame,
            'pan': self.pan,
            'tilt': self.tilt,
            'zoom': self.zoom,
            'frame_idx': self.current_frame,
            'coverage_reward': self.last_cov_reward,
            'novel_ids': self.last_novel,
            'seen_ids': len(self.tracks),
            'stale_total': self.scene.summary()['stale_total'],
        }

        self.last_action = action
        self.last_reward = reward
        self.reward_history.append(reward)
        self.detections_history.append(len(self.last_detections))

        return state, reward, done, info

    # ----------------------------------------------------------- actions

    def _apply_action(self, action, param=1.0):
        # param in [0, 1] scales movement magnitude (from MP-DQN)
        param = float(np.clip(param, 0.0, 1.0))
        if action == 0:       # STAY
            return
        if action == 1:       # PAN LEFT
            self.pan -= self.pan_speed * param
        elif action == 2:     # PAN RIGHT
            self.pan += self.pan_speed * param
        elif action == 3:     # ZOOM IN
            self.zoom = min(self.zoom_range[1], self.zoom + self.zoom_speed * param)
        elif action == 4:     # ZOOM OUT
            self.zoom = max(self.zoom_range[0], self.zoom - self.zoom_speed * param)
        elif action == 5:     # TILT UP (toward horizon)
            self.tilt = min(self.tilt_range[1], self.tilt + self.tilt_speed * param)
        elif action == 6:     # TILT DOWN (toward nadir)
            self.tilt = max(self.tilt_range[0], self.tilt - self.tilt_speed * param)

        # Wrap pan
        if self.pan > 180.0:
            self.pan -= 360.0
        elif self.pan < -180.0:
            self.pan += 360.0

    # ----------------------------------------------------------- video

    def _get_frame(self):
        if self.cap is None or not self.cap.isOpened() or self.frame_count <= 0:
            return np.zeros((2320, 2320, 3), dtype=np.uint8)
        # No-loop mode: once the requested frame passes the clip end, mark the
        # episode as ended and hold on the last real frame (no wrap-around).
        if not self.loop_video and self.current_frame >= self.frame_count:
            self._clip_ended = True
            idx = int(self.frame_count - 1)
        else:
            idx = int(self.current_frame % self.frame_count)
        # Sequential decode: seeking with CAP_PROP_POS_FRAMES every step forces an
        # H.264 keyframe re-decode (~700 ms). The access pattern is monotonic
        # (current_frame grows by frame_skip), so grab()-forward to the target and
        # only hard-seek when the index jumps backwards (episode wrap) or far ahead.
        nxt = getattr(self, '_next_read', None)
        if nxt is None or idx < nxt or (idx - nxt) > 90:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._next_read = idx
        while self._next_read < idx:
            if not self.cap.grab():
                break
            self._next_read += 1
        ok, frame = self.cap.read()
        if ok:
            self._next_read = idx + 1
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            self._next_read = 1
        return frame if ok else np.zeros((2320, 2320, 3), dtype=np.uint8)

    # ----------------------------------------------------------- projection

    def _get_ptz_view(self, frame):
        """Unwarp the fisheye into a perspective image at the current pan/tilt/zoom."""
        return self.project_view(frame, self.pan, self.tilt, self.zoom)

    def project_view(self, frame, pan, tilt, zoom):
        """Equidistant fisheye → perspective unwarp at arbitrary pan/tilt/zoom.

        Pure function w.r.t. (pan, tilt, zoom) so callers can render multiple
        virtual cameras from one shared frame (used by DualFisheyePTZEnvironment).

            r_pix = (R / (fov/2)) * theta,  theta = angle from optical axis (-Z)
        """
        if frame is None or frame.size == 0:
            return np.zeros((self.ptz_out_h, self.ptz_out_w, 3), dtype=np.uint8)

        if self.fish_R is None:
            self._detect_fisheye_circle(frame)

        out_h, out_w = self.ptz_out_h, self.ptz_out_w
        fov = self.base_fov / max(zoom, 1e-6)
        fov_rad = np.radians(fov)
        f = (out_w / 2.0) / np.tan(fov_rad / 2.0)

        # Pixel grid in the virtual perspective image (centered)
        xs = np.arange(out_w, dtype=np.float32) - out_w / 2.0
        ys = np.arange(out_h, dtype=np.float32) - out_h / 2.0
        xx, yy = np.meshgrid(xs, ys)

        # Camera-space rays (looking along -Z by convention).
        dx = xx
        dy = yy
        dz = -np.full_like(xx, f, dtype=np.float32)

        # Rotate the virtual camera by tilt (about world X) then pan (about world Z).
        t = np.radians(tilt)
        cos_t, sin_t = np.cos(t), np.sin(t)
        rx = dx
        ry = dy * cos_t - dz * sin_t
        rz = dy * sin_t + dz * cos_t

        p = np.radians(pan)
        cos_p, sin_p = np.cos(p), np.sin(p)
        wx = rx * cos_p - ry * sin_p
        wy = rx * sin_p + ry * cos_p
        wz = rz

        # World ray (wx, wy, wz). Theta is angle from -Z (down).
        r3 = np.sqrt(wx * wx + wy * wy + wz * wz) + 1e-9
        cos_theta = -wz / r3                       # because optical axis is -Z
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        phi = np.arctan2(wy, wx)

        # Equidistant projection back into fisheye pixels.
        fish_fov_rad = np.radians(self.fish_fov_deg)
        f_fish = self.fish_R / (fish_fov_rad / 2.0)
        r_pix = f_fish * theta

        map_x = (self.fish_cx + r_pix * np.cos(phi)).astype(np.float32)
        map_y = (self.fish_cy + r_pix * np.sin(phi)).astype(np.float32)

        # Mark rays that fell outside the fisheye disk
        inside = r_pix <= self.fish_R
        map_x = np.where(inside, map_x, -1.0).astype(np.float32)
        map_y = np.where(inside, map_y, -1.0).astype(np.float32)

        view = cv2.remap(
            frame, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return view

    @staticmethod
    def world_bearing(bbox, pan, tilt, zoom, out_w, out_h, base_fov):
        """Map a detection's view-pixel centre back to its world bearing.

        Exact inverse of ``project_view``'s camera->world rotation, so the result
        is independent of the camera's current pan/tilt/zoom. Tracking the same
        person's bearing across steps therefore yields the person's *true* angular
        motion with the camera's own motion cancelled out.

        Returns (azimuth_deg, polar_deg) where polar matches the tilt convention
        (0 = nadir / straight down). Pure function — used for motion estimation.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0 - out_w / 2.0
        cy = (y1 + y2) / 2.0 - out_h / 2.0

        fov_rad = np.radians(base_fov / max(zoom, 1e-6))
        f = (out_w / 2.0) / np.tan(fov_rad / 2.0)
        dx, dy, dz = cx, cy, -f

        t = np.radians(tilt)
        cos_t, sin_t = np.cos(t), np.sin(t)
        rx = dx
        ry = dy * cos_t - dz * sin_t
        rz = dy * sin_t + dz * cos_t

        p = np.radians(pan)
        cos_p, sin_p = np.cos(p), np.sin(p)
        wx = rx * cos_p - ry * sin_p
        wy = rx * sin_p + ry * cos_p
        wz = rz

        r3 = float(np.sqrt(wx * wx + wy * wy + wz * wz)) + 1e-9
        az = float(np.degrees(np.arctan2(wy, wx)))
        polar = float(np.degrees(np.arccos(np.clip(-wz / r3, -1.0, 1.0))))
        return az, polar

    # ----------------------------------------------------------- detection

    def _detect_objects(self, frame):
        if self.detector is None or frame is None:
            return []
        try:
            if self.detector_is_obb:
                return self._detect_objects_obb(frame)
            results = self.detector.track(
                frame, persist=True, tracker='botsort.yaml',
                conf=self.yolo_conf, verbose=False,
            )
            out = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.detector.names[cls_id]
                    if cls_name != 'person':
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    track_id = int(box.id[0]) if box.id is not None else -1
                    out.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(box.conf[0]),
                        'class': cls_name,
                        'track_id': track_id,
                    })
            return out
        except Exception:
            return []

    def _detect_objects_obb(self, frame):
        """OBB-model path: predict, convert each OBB to its axis-aligned envelope,
        and assign cheap IoU-based track_ids across consecutive calls so the
        agent's RE-ID continuity / novelty bonuses keep firing.

        Single-class detector (nc=1, 'person') is assumed — matches the
        sibling fisheye_person_detect cfg.
        """
        results = self.detector.predict(
            frame, conf=self.yolo_conf, verbose=False,
        )
        dets = []
        for r in results:
            obb = getattr(r, 'obb', None)
            if obb is None or len(obb) == 0:
                continue
            for i in range(len(obb)):
                # Axis-aligned envelope from the 4 rotated corners.
                xyxyxyxy = obb.xyxyxyxy[i].detach().cpu().numpy().reshape(-1, 2)
                x1 = float(xyxyxyxy[:, 0].min())
                y1 = float(xyxyxyxy[:, 1].min())
                x2 = float(xyxyxyxy[:, 0].max())
                y2 = float(xyxyxyxy[:, 1].max())
                conf = float(obb.conf[i])
                dets.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': 'person',
                    'track_id': -1,
                })

        # ---- IoU-based RE-ID with a persistent track pool ----
        # Each track keeps its last bbox + last-seen step so a brief miss
        # (occlusion, dropped frame) doesn't spawn a new id. Matching against
        # tracks seen within the last `max_age` steps, not just the previous
        # frame, is what stops the runaway id inflation that made the explorer
        # reward explode.
        if not hasattr(self, '_obb_tracks'):
            self._obb_tracks = {}        # id -> {'bbox', 'last_step'}
            self._obb_next_id = 1
            self._obb_step = 0
        self._obb_step += 1
        max_age = 8                      # frames a track survives without a hit
        iou_match = 0.4                  # stricter than the old 0.3

        def _iou(a, b):
            x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
            x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            union = area_a + area_b - inter
            return inter / union if union > 1e-6 else 0.0

        # Drop stale tracks.
        self._obb_tracks = {
            tid: t for tid, t in self._obb_tracks.items()
            if self._obb_step - t['last_step'] <= max_age
        }

        used = set()
        for d in dets:
            best, best_iou = -1, iou_match
            for tid, t in self._obb_tracks.items():
                if tid in used:
                    continue
                iou = _iou(d['bbox'], t['bbox'])
                if iou > best_iou:
                    best, best_iou = tid, iou
            if best >= 0:
                d['track_id'] = best
                self._obb_tracks[best] = {'bbox': d['bbox'], 'last_step': self._obb_step}
                used.add(best)
            else:
                tid = self._obb_next_id
                self._obb_next_id += 1
                d['track_id'] = tid
                self._obb_tracks[tid] = {'bbox': d['bbox'], 'last_step': self._obb_step}
                used.add(tid)
        return dets

    # ----------------------------------------------------------- reward + state

    def _compute_reward(self, detections, action):
        reward = 0.0
        if detections:
            avg_conf = float(np.mean([d['confidence'] for d in detections]))
            reward += avg_conf * 2.0

            w, h = self.ptz_out_w, self.ptz_out_h
            cx, cy = w / 2.0, h / 2.0
            max_dist = np.sqrt(cx * cx + cy * cy)
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                dx_ = (x1 + x2) / 2.0 - cx
                dy_ = (y1 + y2) / 2.0 - cy
                center_score = 1.0 - (np.sqrt(dx_ * dx_ + dy_ * dy_) / max_dist)
                reward += float(center_score) * 0.3
            reward += 1.0
        else:
            reward -= 0.5

        if action != self.last_action:
            reward -= 0.05

        # Scene-level terms: encourages leaving local optima.
        # Coverage staleness pays for visiting stale sectors regardless of detections.
        # Novelty pays a one-time bonus per never-before-seen track_id this episode.
        # Cap novelty per step so a noisy detector / tracker can't blow up the reward.
        reward += self.coverage_weight * self.last_cov_reward
        reward += self.novelty_weight * min(self.last_novel, 5)
        return reward

    def _get_state(self):
        num_dets = self.detections_history[-1] if self.detections_history else 0
        avg_dets = float(np.mean(self.detections_history[-10:])) if self.detections_history else 0.0
        reward_trend = float(np.mean(self.reward_history[-5:])) if self.reward_history else 0.0

        rel_x, rel_y, rel_area, best_conf = -2.0, -2.0, -1.0, 0.0
        target_det = None
        if self.last_detections and self.target_id is not None:
            for d in self.last_detections:
                if d.get('track_id') == self.target_id:
                    target_det = d
                    break
        if target_det is None and self.last_detections:
            target_det = max(self.last_detections, key=lambda d: d['confidence'])

        if target_det:
            x1, y1, x2, y2 = target_det['bbox']
            w, h = self.ptz_out_w, self.ptz_out_h
            rel_x = ((x1 + x2) / 2.0 - w / 2.0) / (w / 2.0)
            rel_y = ((y1 + y2) / 2.0 - h / 2.0) / (h / 2.0)
            rel_area = ((x2 - x1) * (y2 - y1)) / (w * h)
            best_conf = target_det['confidence']

        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0] = num_dets / 5.0
        state[1] = avg_dets / 5.0
        state[2] = self.pan / 180.0
        state[3] = self.tilt / 80.0
        state[4] = self.zoom / 8.0
        state[5] = rel_x
        state[6] = rel_y
        state[7] = rel_area
        state[8] = best_conf
        state[9] = self.last_action / 7.0
        state[10] = reward_trend / 5.0
        state[11] = self.current_step / max(1, self.max_steps)

        # Scene-level features [12:24]: coverage + track registry context.
        if self.state_dim > 12:
            summary = self.scene.summary()
            state[12] = summary['stale_total']
            # Polar staleness super-bands (always 4 floats)
            for i, v in enumerate(summary['polar_staleness'][:4]):
                state[13 + i] = float(v)
            state[17] = summary['covered_frac']
            state[18] = min(len(self.tracks), 20) / 20.0     # seen-id count, capped
            state[19] = min(self.last_novel, 5) / 5.0        # novel ids this step
            state[20] = float(np.tanh(self.last_cov_reward)) # coverage gain this step
            # New: outer (peripheral) staleness — tells agent there's reward
            # to be had near the horizon.
            state[21] = summary['outer_stale']
            # 22-23 reserved
        return state

    # ----------------------------------------------------------- misc

    def render(self, mode='human'):
        frame = self._get_frame()
        view = self._get_ptz_view(frame)
        info = f"Pan {self.pan:+.1f}  Tilt {self.tilt:+.1f}  Zoom {self.zoom:.1f}x"
        cv2.putText(view, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Fisheye PTZ', view)
        cv2.waitKey(1)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()


if __name__ == '__main__':
    env = FisheyePTZEnvironment({'max_steps': 8})
    s = env.reset()
    print('init state shape:', s.shape, 'fish R:', env.fish_R)
    for i in range(8):
        a = env.action_space.sample()
        s, r, d, info = env.step(a, action_param=0.7)
        print(f'step {i}: action={a} reward={r:+.2f} dets={info["num_detections"]} '
              f'pan={info["pan"]:+.1f} tilt={info["tilt"]:+.1f} zoom={info["zoom"]:.1f}')
        if d:
            break
    env.close()
