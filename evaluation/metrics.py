"""Ground-truth metrics for the fisheye active-coverage task — evaluation signals
that are *independent of the training reward*.

Why
---
The episode reward is exactly what the policy was optimised for, so it is not a
neutral yardstick: "RL maximises the reward it was trained on" is not a result a
reviewer will credit. This module scores a policy against a **pseudo-ground-truth
oracle** instead.

The oracle is the *same* fine-tuned top-down OBB detector, but run on the **full
raw fisheye frame** every step. The raw fisheye sees the entire room at once,
whereas each agent camera only sees a narrow perspective crop — so the oracle's
detections are a reasonable stand-in for "every person present", and the agent's
job is to actually point a camera at them. Both the oracle people and the agent's
own detections are expressed in a common, camera-motion-cancelled
``(azimuth, polar)`` bearing space (via :func:`world_bearing`), so they can be
matched directly by angular separation.

Metrics (all reward-free)
-------------------------
* ``discovery_rate``       fraction of oracle people the agent detected at least once
* ``time_to_detect``       mean steps from a person appearing to the agent seeing them
* ``observed_time_frac``   SUSTAINED coverage -- mean over people of the fraction of their
                           present-time the agent actually observed them (discovery asks
                           only whether they were EVER seen; this asks how much of the time)
* ``max_unobserved_gap``   mean over people of the longest consecutive unobserved run
* ``coverage_pct``         fraction of the azimuth circle the cameras swept
* ``inter_camera_overlap`` fraction of steps the two cameras redundantly overlap
* ``dwell_ratio_moving``   fraction of agent detections spent on *moving* people (tests N8)
* ``time_to_detect_mover`` mean steps to (re)acquire a person once they start moving (tests N8)

The pure functions at the top take plain Python/NumPy data and are unit-tested by
``--selftest`` (no GPU or env needed). The :class:`Oracle` and
:func:`evaluate_policy_gt` runner wire them to the live env.

Run
---
    python evaluation/metrics.py --selftest          # geometry + metric unit checks

    python evaluation/metrics.py \
        --data_path data/test --detector_weights yolo26n_obb_topview_person_290526.pt \
        --steps 720 --episodes 3 --seed_list 0 --state_dim 29 \
        --tracker results/fisheye_dual_20260529_151337/tracker_best.pt \
        --explorer results/fisheye_dual_20260529_151337/explorer_best.pt \
        --policies rl,random,sweep,greedy \
        --out evaluation/results/gt_seed0.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ======================================================================
# Geometry — a bearing is (azimuth_deg, polar_deg) with polar measured
# from the nadir (0 = straight down), matching the env's tilt convention
# and the return of FisheyePTZEnvironment.world_bearing.
# ======================================================================

def bearing_to_vec(az_deg, polar_deg):
    """Unit 3-vector for a bearing. Convention is internal and only used for
    relative angles, so it just has to be consistent between oracle and agent."""
    a = math.radians(az_deg)
    t = math.radians(polar_deg)
    st = math.sin(t)
    return np.array([st * math.cos(a), st * math.sin(a), math.cos(t)], dtype=np.float64)


def angular_sep(b1, b2):
    """Great-circle angle (deg) between two (az, polar) bearings."""
    v1 = bearing_to_vec(*b1)
    v2 = bearing_to_vec(*b2)
    d = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(d))


def fisheye_pixel_to_bearing(px, py, cx, cy, fish_R, fish_fov_deg):
    """Inverse of the equidistant fisheye projection: a raw-fisheye pixel ->
    world bearing (az, polar). Mirrors ``project_view``:

        r_pix = f_fish * theta,   f_fish = R / (fov/2),   phi = atan2(dy, dx)

    so theta (polar from nadir) = r_pix / f_fish and azimuth = phi. This lands in
    the exact same space as ``world_bearing`` (verified in --selftest)."""
    dx = px - cx
    dy = py - cy
    r_pix = math.hypot(dx, dy)
    f_fish = fish_R / (math.radians(fish_fov_deg) / 2.0)
    polar = math.degrees(r_pix / f_fish) if f_fish > 1e-9 else 0.0
    az = math.degrees(math.atan2(dy, dx))
    return az, polar


# ======================================================================
# Pure metric functions. Inputs are per-step logs, so these are trivially
# unit-testable and carry no env/GPU dependency.
#
# Shared log shapes (one entry per step, length T):
#   gt[t]     : list of dicts {id, bearing:(az,pol), moving:bool}
#   agent[t]  : list of bearings (az, pol) the agent actually detected this step
#   axes[t]   : list of per-camera optical-axis bearings (az, pol)
#   zooms[t]  : list of per-camera zoom (for FOV width)
# ======================================================================

def _first_present(gt):
    first = {}
    for t, people in enumerate(gt):
        for p in people:
            first.setdefault(p['id'], t)
    return first


def _discovery_events(gt, agent, gate_deg):
    """For each gt id, the first step an agent detection fell within gate_deg."""
    seen = {}
    for t, (people, dets) in enumerate(zip(gt, agent)):
        if not dets:
            continue
        for p in people:
            if p['id'] in seen:
                continue
            if any(angular_sep(p['bearing'], d) <= gate_deg for d in dets):
                seen[p['id']] = t
    return seen


def discovery_rate(gt, agent, gate_deg=15.0):
    ids = set(_first_present(gt))
    if not ids:
        return float('nan')
    seen = _discovery_events(gt, agent, gate_deg)
    return len(seen) / len(ids)


def time_to_detect(gt, agent, gate_deg=15.0):
    """Mean steps from appearance to first agent detection, over *discovered*
    people (undetected people don't contribute an infinite latency; the
    discovery_rate already penalises them)."""
    first = _first_present(gt)
    seen = _discovery_events(gt, agent, gate_deg)
    lat = [seen[i] - first[i] for i in seen]
    return float(np.mean(lat)) if lat else float('nan')


def _observation_series(gt, agent, gate_deg):
    """Per person id: (n_present, n_observed, max_unobserved_run) over the steps
    that person is present. 'Observed' at step t means some agent detection at t
    fell within gate_deg of them -- the same test discovery uses, so the two
    metrics agree about what counts as seeing someone."""
    stats = {}
    for t, (people, dets) in enumerate(zip(gt, agent)):
        for p in people:
            st = stats.setdefault(p['id'], {'present': 0, 'seen': 0, 'run': 0, 'maxrun': 0})
            st['present'] += 1
            hit = bool(dets) and any(angular_sep(p['bearing'], d) <= gate_deg for d in dets)
            if hit:
                st['seen'] += 1
                st['run'] = 0
            else:
                st['run'] += 1
                if st['run'] > st['maxrun']:
                    st['maxrun'] = st['run']
    return stats


def observed_time_frac(gt, agent, gate_deg=15.0):
    """SUSTAINED coverage: mean over people of (steps observed / steps present).

    Complements discovery_rate, which only asks whether a person was EVER seen.
    A policy that finds everyone once and then looks away scores high on
    discovery and low here. Macro-averaged over people so one long-present
    person cannot dominate; people never observed contribute 0 rather than
    being dropped."""
    stats = _observation_series(gt, agent, gate_deg)
    if not stats:
        return float('nan')
    return float(np.mean([st['seen'] / st['present'] for st in stats.values()
                          if st['present'] > 0]))


def max_unobserved_gap(gt, agent, gate_deg=15.0):
    """Mean over people of their longest consecutive unobserved run, in steps,
    within their own presence window. The worst-case companion to
    observed_time_frac: a policy can observe 60% of the time and still leave a
    single very long blind interval."""
    stats = _observation_series(gt, agent, gate_deg)
    if not stats:
        return float('nan')
    return float(np.mean([st['maxrun'] for st in stats.values()]))


def coverage_pct(axes, n_bins=36):
    """Fraction of azimuth bins (default 10 deg) that any camera axis pointed at
    over the episode. Pure trajectory metric — no ground truth needed."""
    visited = set()
    for step_axes in axes:
        for az, _pol in step_axes:
            b = int(((az % 360.0) / 360.0) * n_bins) % n_bins
            visited.add(b)
    return len(visited) / n_bins


def inter_camera_overlap(axes, zooms, base_fov=90.0):
    """Fraction of steps where two cameras' fields of view overlap (their axis
    separation is less than the sum of their half-FOVs). Lower = the cameras
    divide the scene rather than double-covering it."""
    n = 0
    over = 0
    for step_axes, step_zoom in zip(axes, zooms):
        if len(step_axes) < 2:
            continue
        n += 1
        sep = angular_sep(step_axes[0], step_axes[1])
        half = 0.5 * (base_fov / max(step_zoom[0], 1e-6) + base_fov / max(step_zoom[1], 1e-6)) / 2.0
        # half is the mean half-FOV of the two cams; overlap if axes closer than 2*half.
        if sep <= 2.0 * half:
            over += 1
    return (over / n) if n else float('nan')


_SPHERE_GRID = None


def _sphere_grid(n_theta=720, n_phi=1440):
    """Cache a full-sphere direction grid + per-cell solid-angle weights, so the
    continuous overlap metric integrates over the sphere rather than testing a
    single yes/no intersection. Returns (unit_vectors [N,3], weights [N])."""
    global _SPHERE_GRID
    if _SPHERE_GRID is not None:
        return _SPHERE_GRID
    thetas = (np.arange(n_theta) + 0.5) * (math.pi / n_theta)      # 0..pi (from pole)
    phis = (np.arange(n_phi) + 0.5) * (2 * math.pi / n_phi)        # 0..2pi
    T, P = np.meshgrid(thetas, phis, indexing='ij')
    st = np.sin(T)
    V = np.stack([st * np.cos(P), st * np.sin(P), np.cos(T)], axis=-1).reshape(-1, 3)
    W = (st * (math.pi / n_theta) * (2 * math.pi / n_phi)).reshape(-1)  # sinθ dθ dφ
    _SPHERE_GRID = (V, W)
    return _SPHERE_GRID


def solid_angle_overlap(axes, zooms, base_fov=90.0):
    """Continuous redundancy: mean over steps of the solid-angle IoU of the two
    view cones, Omega(V1 ∩ V2) / Omega(V1 ∪ V2). Unlike ``inter_camera_overlap``
    (a 0/1 any-intersection test), a grazing touch scores near 0 and near-identical
    aim scores near 1. Cones are the circular FOV caps (half-angle = fov/2)."""
    V, W = _sphere_grid()
    ious = []
    for step_axes, step_zoom in zip(axes, zooms):
        if len(step_axes) < 2:
            continue
        a1 = bearing_to_vec(*step_axes[0]); a2 = bearing_to_vec(*step_axes[1])
        r1 = math.radians(0.5 * base_fov / max(step_zoom[0], 1e-6))
        r2 = math.radians(0.5 * base_fov / max(step_zoom[1], 1e-6))
        in1 = (V @ a1) >= math.cos(r1)
        in2 = (V @ a2) >= math.cos(r2)
        inter = float(W[in1 & in2].sum())
        union = float(W[in1 | in2].sum())
        ious.append((inter / union) if union > 0 else 0.0)
    return float(np.mean(ious)) if ious else float('nan')


# ----------------------------------------------------------------------
# EXACT frustum overlap.
#
# The two functions above model each virtual view as a circular cap of
# half-angle fov/2. The renderer does not produce a cap: project_view builds a
# W x H rectilinear image with f = (W/2)/tan(fov/2), so ``base_fov`` is the
# HORIZONTAL fov and the view is a rectangular pyramid -- 90.0 x 58.7 deg at
# the default 960x540, whose corners reach 48.9 deg but whose top/bottom edges
# reach only 29.4 deg. The cap is 1.30x too large in solid angle and, more
# importantly, is wrong in the direction that matters here: two cameras at
# opposite azimuth separate along the image's SHORT axis, so the cap test
# reports redundancy between tilt 29.4 and 45 deg where the rendered views are
# already disjoint. These functions test the actual image rectangle instead.
# ----------------------------------------------------------------------

def _cam_basis(pan_deg, tilt_deg):
    """(right, up, forward) world vectors, matching project_view's rotation:
    tilt about world X, then pan about world Z, optical axis -Z."""
    t, p = math.radians(tilt_deg), math.radians(pan_deg)
    ct, st_, cp, sp = math.cos(t), math.sin(t), math.cos(p), math.sin(p)

    def rot(v):
        x, y, z = v
        ry = y * ct - z * st_
        rz = y * st_ + z * ct
        return np.array([x * cp - ry * sp, x * sp + ry * cp, rz], dtype=np.float64)

    return rot((1.0, 0.0, 0.0)), rot((0.0, 1.0, 0.0)), rot((0.0, 0.0, -1.0))


def _half_angles(base_fov, zoom, out_w, out_h):
    """(horizontal, vertical, corner) half-angles in radians for one view."""
    hfov = math.radians(base_fov / max(zoom, 1e-6))
    f = (out_w / 2.0) / math.tan(hfov / 2.0)
    return (hfov / 2.0,
            math.atan((out_h / 2.0) / f),
            math.atan(math.hypot(out_w / 2.0, out_h / 2.0) / f))


def frustum_solid_angle(base_fov, zoom, out_w, out_h):
    """Exact solid angle of one rectilinear view: 4*asin(sin(a/2)sin(b/2))."""
    ha, va, _ = _half_angles(base_fov, zoom, out_w, out_h)
    return 4.0 * math.asin(math.sin(ha) * math.sin(va))


def _in_frustum(V, pan, tilt, zoom, base_fov, out_w, out_h):
    """Boolean mask: which unit vectors project inside the rendered image."""
    ha, va, _ = _half_angles(base_fov, zoom, out_w, out_h)
    r, u, fw = _cam_basis(pan, tilt)
    z = V @ fw
    ok = z > 1e-12
    zz = np.where(ok, z, 1.0)
    return ok & (np.abs((V @ r) / zz) <= math.tan(ha)) \
              & (np.abs((V @ u) / zz) <= math.tan(va))


def _frustum_pair_iou(p1, p2, base_fov, out_w, out_h):
    """Solid-angle IoU of two rendered views. Analytic areas + analytic
    disjointness test; the grid is only touched when the bounding caps of the
    two views actually intersect, which keeps this cheap for swept policies."""
    (pan1, tilt1, z1), (pan2, tilt2, z2) = p1, p2
    _, _, c1 = _half_angles(base_fov, z1, out_w, out_h)
    _, _, c2 = _half_angles(base_fov, z2, out_w, out_h)
    _, _, fw1 = _cam_basis(pan1, tilt1)
    _, _, fw2 = _cam_basis(pan2, tilt2)
    sep = math.acos(max(-1.0, min(1.0, float(fw1 @ fw2))))
    o1 = frustum_solid_angle(base_fov, z1, out_w, out_h)
    o2 = frustum_solid_angle(base_fov, z2, out_w, out_h)
    if sep >= c1 + c2:                       # bounding caps miss -> exactly disjoint
        return 0.0
    V, W = _sphere_grid()
    m1 = _in_frustum(V, pan1, tilt1, z1, base_fov, out_w, out_h)
    m2 = _in_frustum(V, pan2, tilt2, z2, base_fov, out_w, out_h)
    inter = float(W[m1 & m2].sum())
    union = o1 + o2 - inter
    return (inter / union) if union > 0 else 0.0


def inter_camera_overlap_frustum(poses, base_fov=90.0, out_w=960, out_h=540):
    """Fraction of steps where the two RENDERED views actually intersect.
    ``poses[t]`` is [(pan, tilt, zoom), (pan, tilt, zoom)]. Exact counterpart of
    ``inter_camera_overlap``."""
    n = over = 0
    for step in poses:
        if len(step) < 2:
            continue
        n += 1
        if _frustum_pair_iou(step[0], step[1], base_fov, out_w, out_h) > 0.0:
            over += 1
    return (over / n) if n else float('nan')


def solid_angle_overlap_frustum(poses, base_fov=90.0, out_w=960, out_h=540):
    """Mean solid-angle IoU of the two rendered views. Exact counterpart of
    ``solid_angle_overlap``."""
    ious = [_frustum_pair_iou(s[0], s[1], base_fov, out_w, out_h)
            for s in poses if len(s) >= 2]
    return float(np.mean(ious)) if ious else float('nan')


def dwell_ratio_moving(gt, agent, gate_deg=15.0):
    """Of the agent detections that match a known person, the fraction that land
    on a *moving* person. Higher means the policy spends its looking on movers —
    the behaviour the motion-priority mechanism (N8) is supposed to produce."""
    moving = 0
    total = 0
    for people, dets in zip(gt, agent):
        if not people or not dets:
            continue
        for d in dets:
            best = None
            best_sep = gate_deg
            for p in people:
                s = angular_sep(p['bearing'], d)
                if s <= best_sep:
                    best_sep = s
                    best = p
            if best is not None:
                total += 1
                if best['moving']:
                    moving += 1
    return (moving / total) if total else float('nan')


def time_to_detect_mover(gt, agent, gate_deg=15.0):
    """Mean steps from a person *starting to move* to the agent detecting them
    while they are moving. Directly tests the anti-fixation hand-off claim."""
    onset = {}
    for t, people in enumerate(gt):
        for p in people:
            if p['moving'] and p['id'] not in onset:
                onset[p['id']] = t
    lat = []
    for pid, t0 in onset.items():
        for t in range(t0, len(gt)):
            people = {p['id']: p for p in gt[t]}
            if pid not in people or not people[pid]['moving']:
                continue
            if any(angular_sep(people[pid]['bearing'], d) <= gate_deg for d in agent[t]):
                lat.append(t - t0)
                break
    return float(np.mean(lat)) if lat else float('nan')


def all_metrics(gt, agent, axes, zooms, gate_deg=15.0, n_bins=36, base_fov=90.0,
                frame_skip=5, fps=None, poses=None, out_w=960, out_h=540):
    m = {
        'n_gt_people': len(_first_present(gt)),
        'discovery_rate': discovery_rate(gt, agent, gate_deg),
        'time_to_detect_steps': time_to_detect(gt, agent, gate_deg),
        'observed_time_frac': observed_time_frac(gt, agent, gate_deg),
        'max_unobserved_gap': max_unobserved_gap(gt, agent, gate_deg),
        'coverage_pct': coverage_pct(axes, n_bins),
        'inter_camera_overlap': inter_camera_overlap(axes, zooms, base_fov),
        'solid_angle_overlap': solid_angle_overlap(axes, zooms, base_fov),
        'dwell_ratio_moving': dwell_ratio_moving(gt, agent, gate_deg),
        'time_to_detect_mover_steps': time_to_detect_mover(gt, agent, gate_deg),
    }
    if poses is not None:
        # Exact rendered-frustum overlap. The two keys above keep the circular-cone
        # approximation for continuity with earlier runs; these are the correct ones.
        m['inter_camera_overlap_frustum'] = inter_camera_overlap_frustum(
            poses, base_fov, out_w, out_h)
        m['solid_angle_overlap_frustum'] = solid_angle_overlap_frustum(
            poses, base_fov, out_w, out_h)
    if fps:
        sec = frame_skip / float(fps)
        for k in ('time_to_detect_steps', 'time_to_detect_mover_steps'):
            v = m[k]
            m[k.replace('_steps', '_sec')] = (v * sec) if v == v else float('nan')  # nan-safe
    return m


# ======================================================================
# Oracle — runs the fine-tuned OBB detector on the raw fisheye and tracks
# people by nearest bearing, labelling each as moving/static.
# ======================================================================

class Oracle:
    def __init__(self, host, gate_deg=12.0, vel_thresh_deg=2.0, vel_window=3, max_age=6):
        self.host = host
        self.gate = gate_deg
        self.vel_thresh = vel_thresh_deg
        self.vel_window = vel_window
        self.max_age = max_age
        self.tracks = {}      # id -> {bearing, last_step, hist:[bearings], vel}
        self.next_id = 1
        self.step = -1

    def _detect_raw(self, frame):
        det = self.host.detector
        if det is None or frame is None:
            return []
        results = det.predict(frame, conf=self.host.yolo_conf, verbose=False)
        out = []
        for r in results:
            obb = getattr(r, 'obb', None)
            if obb is None or len(obb) == 0:
                continue
            for i in range(len(obb)):
                try:
                    cx, cy = obb.xywhr[i][:2].detach().cpu().numpy().tolist()
                except Exception:
                    corners = obb.xyxyxyxy[i].detach().cpu().numpy().reshape(-1, 2)
                    cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
                out.append(fisheye_pixel_to_bearing(
                    cx, cy, self.host.fish_cx, self.host.fish_cy,
                    self.host.fish_R, self.host.fish_fov_deg))
        return out

    def update(self, frame, step):
        """Detect on the raw fisheye, associate to tracks, label motion. Returns
        the list of {id, bearing, moving} for this step."""
        self.step = step
        bearings = self._detect_raw(frame)

        # drop stale
        self.tracks = {i: t for i, t in self.tracks.items()
                       if step - t['last_step'] <= self.max_age}

        # greedy nearest-bearing association
        unused = set(self.tracks)
        assigned = {}
        for b in bearings:
            best, best_sep = None, self.gate
            for i in unused:
                s = angular_sep(self.tracks[i]['bearing'], b)
                if s <= best_sep:
                    best_sep, best = s, i
            if best is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'bearing': b, 'last_step': step, 'hist': [b], 'vel': 0.0}
            else:
                unused.discard(best)
                tid = best
                tk = self.tracks[tid]
                tk['hist'].append(b)
                tk['hist'] = tk['hist'][-(self.vel_window + 1):]
                if len(tk['hist']) >= 2:
                    seps = [angular_sep(tk['hist'][k], tk['hist'][k + 1])
                            for k in range(len(tk['hist']) - 1)]
                    tk['vel'] = float(np.mean(seps))
                tk['bearing'] = b
                tk['last_step'] = step
            assigned[tid] = self.tracks[tid]

        return [{'id': i, 'bearing': t['bearing'], 'moving': t['vel'] >= self.vel_thresh}
                for i, t in assigned.items()]


# ======================================================================
# Rollout: step a policy through the env, logging oracle GT + agent obs.
# ======================================================================

def _axis_bearing(host, pan, tilt, zoom):
    """Optical-axis bearing of a virtual camera, via world_bearing of the view
    centre pixel — reuses the env's own inverse so conventions can't drift."""
    w, h = host.ptz_out_w, host.ptz_out_h
    return host.world_bearing([w / 2.0, h / 2.0, w / 2.0, h / 2.0],
                              pan, tilt, zoom, w, h, host.base_fov)


class _SeqReader:
    """Sequential frame reader for the oracle — same grab-forward trick as the
    env's _get_frame, so it never pays the per-step keyframe-seek cost."""
    def __init__(self, path):
        import cv2
        self.cap = cv2.VideoCapture(path)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.next = 0

    def read(self, frame_idx):
        import cv2
        idx = int(frame_idx % self.count) if self.count > 0 else 0
        if idx < self.next or (idx - self.next) > 90:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.next = idx
        while self.next < idx:
            if not self.cap.grab():
                break
            self.next += 1
        ok, frame = self.cap.read()
        self.next = idx + 1 if ok else 0
        return frame if ok else None

    def close(self):
        self.cap.release()


def rollout_gt(env, policy, steps, gate_deg=15.0, oracle_kwargs=None):
    """One episode. Returns (metrics_dict, logs). Assumes a DualFisheyePTZEnvironment."""
    host = env._host
    oracle = Oracle(host, **(oracle_kwargs or {}))
    if hasattr(policy, 'reset'):
        policy.reset()
    states = env.reset()

    reader = _SeqReader(host.video_path)
    gt_log, agent_log, axes_log, zoom_log, pose_log = [], [], [], [], []

    for t in range(steps):
        cf = host.current_frame
        raw = reader.read(cf)
        gt_log.append(oracle.update(raw, t))

        # agent's actual detections this step -> bearings
        dets_bear = []
        for i in range(2):
            for d in env.last_detections[i]:
                dets_bear.append(host.world_bearing(
                    d['bbox'], float(env.pan[i]), float(env.tilt[i]), float(env.zoom[i]),
                    host.ptz_out_w, host.ptz_out_h, host.base_fov))
        agent_log.append(dets_bear)
        axes_log.append([_axis_bearing(host, float(env.pan[i]), float(env.tilt[i]),
                                       float(env.zoom[i])) for i in range(2)])
        zoom_log.append([float(env.zoom[0]), float(env.zoom[1])])
        # Raw (pan, tilt, zoom) as well as the axis bearing: the exact frustum
        # metric needs the camera's roll, which the bearing alone does not carry.
        pose_log.append([(float(env.pan[i]), float(env.tilt[i]), float(env.zoom[i]))
                         for i in range(2)])

        actions, params = policy.act(states, env)
        states, _r, done, _ = env.step(actions, params)
        if done:
            break

    reader.close()
    m = all_metrics(gt_log, agent_log, axes_log, zoom_log, gate_deg=gate_deg,
                    base_fov=host.base_fov, frame_skip=host.frame_skip,
                    fps=reader.fps or None, poses=pose_log,
                    out_w=host.ptz_out_w, out_h=host.ptz_out_h)
    return m, {'gt': gt_log, 'agent': agent_log, 'axes': axes_log,
               'zooms': zoom_log, 'poses': pose_log}


# ======================================================================
# Self-test — geometry round-trip + metric sanity, no GPU/env needed.
# ======================================================================

def _selftest():
    ok = True

    # 1) fisheye_pixel_to_bearing must invert the env's forward projection.
    #    Reproduce project_view's centre-pixel mapping for a few bearings and
    #    check the inverse recovers them.
    cx = cy = 1160.0
    R = 1160.0
    fov = 180.0
    f_fish = R / (math.radians(fov) / 2.0)
    for az, polar in [(0, 20), (90, 45), (-120, 60), (170, 10)]:
        r_pix = f_fish * math.radians(polar)
        px = cx + r_pix * math.cos(math.radians(az))
        py = cy + r_pix * math.sin(math.radians(az))
        raz, rpolar = fisheye_pixel_to_bearing(px, py, cx, cy, R, fov)
        if abs(((raz - az + 180) % 360) - 180) > 1e-6 or abs(rpolar - polar) > 1e-6:
            print(f'[FAIL] pixel<->bearing round-trip az={az} polar={polar} '
                  f'-> ({raz:.3f},{rpolar:.3f})')
            ok = False

    # 2) angular_sep sanity
    assert abs(angular_sep((0, 45), (0, 45))) < 1e-9
    assert abs(angular_sep((0, 90), (90, 90)) - 90.0) < 1e-6, angular_sep((0, 90), (90, 90))

    # 3) discovery / TTD on a synthetic 4-step episode, 2 people.
    #    p1 present all 4 steps, detected from step1. p2 present steps 2-3, never detected.
    gt = [
        [{'id': 1, 'bearing': (0, 30), 'moving': False}],
        [{'id': 1, 'bearing': (0, 30), 'moving': False}],
        [{'id': 1, 'bearing': (0, 30), 'moving': False}, {'id': 2, 'bearing': (100, 40), 'moving': True}],
        [{'id': 1, 'bearing': (0, 30), 'moving': False}, {'id': 2, 'bearing': (100, 40), 'moving': True}],
    ]
    agent = [[], [(0, 31)], [(0, 30)], [(1, 30)]]
    dr = discovery_rate(gt, agent)
    ttd = time_to_detect(gt, agent)
    if abs(dr - 0.5) > 1e-9:
        print(f'[FAIL] discovery_rate expected 0.5 got {dr}'); ok = False
    if abs(ttd - 1.0) > 1e-9:  # p1 first present t0, first seen t1 -> latency 1
        print(f'[FAIL] time_to_detect expected 1.0 got {ttd}'); ok = False

    # 4) coverage / overlap. Use polar=90 (horizon-looking) so that azimuths 180
    #    apart really are 180 deg of angular separation — at small polar both axes
    #    point near the nadir and are much closer than their azimuth gap suggests.
    axes = [[(0, 90), (180, 90)], [(90, 90), (270, 90)]]
    cov = coverage_pct(axes, n_bins=36)  # bins for 0,180,90,270 deg -> 4 bins
    if abs(cov - 4 / 36) > 1e-9:
        print(f'[FAIL] coverage_pct expected {4/36:.4f} got {cov}'); ok = False
    # --- exact rendered-frustum geometry ---------------------------------
    ha, va, co = _half_angles(90.0, 1.0, 960, 540)
    for nm, got, want in (('HFOV', math.degrees(2 * ha), 90.0),
                          ('VFOV', math.degrees(2 * va), 58.7155),
                          ('corner', math.degrees(co), 48.9254)):
        if abs(got - want) > 1e-3:
            print(f'[FAIL] {nm} expected {want} got {got}'); ok = False
    V_, W_ = _sphere_grid()
    a_an = frustum_solid_angle(90.0, 1.0, 960, 540)
    a_nu = float(W_[_in_frustum(V_, 0.0, 0.0, 1.0, 90.0, 960, 540)].sum())
    if abs(a_an - a_nu) > 2e-3:
        print(f'[FAIL] frustum solid angle analytic {a_an} vs grid {a_nu}'); ok = False
    if abs(_frustum_pair_iou((30., 40., 1.), (30., 40., 1.), 90., 960, 540) - 1.0) > 2e-3:
        print('[FAIL] identical views should have frustum IoU 1'); ok = False
    # Two cameras at opposite azimuth separate along the image's SHORT axis, so
    # the views go disjoint at tilt = VFOV/2 = 29.36 deg, NOT at fov/2 = 45 deg.
    if _frustum_pair_iou((0., 29., 1.), (180., 29., 1.), 90., 960, 540) <= 0.0:
        print('[FAIL] tilt 29 deg should still overlap'); ok = False
    if _frustum_pair_iou((0., 30., 1.), (180., 30., 1.), 90., 960, 540) != 0.0:
        print('[FAIL] tilt 30 deg should be disjoint'); ok = False
    if _frustum_pair_iou((0., 48., 1.), (180., 48., 1.), 90., 960, 540) != 0.0:
        print('[FAIL] tilt 48 deg should be disjoint'); ok = False

    ov = inter_camera_overlap(axes, [[1.0, 1.0], [1.0, 1.0]], base_fov=90.0)
    if abs(ov - 0.0) > 1e-9:  # cams 180 deg apart at the horizon, 90-deg FOV -> never overlap
        print(f'[FAIL] inter_camera_overlap expected 0 got {ov}'); ok = False
    # and a positive control: same axis -> always overlaps
    ov2 = inter_camera_overlap([[(45, 60), (50, 60)]], [[1.0, 1.0]], base_fov=90.0)
    if abs(ov2 - 1.0) > 1e-9:
        print(f'[FAIL] inter_camera_overlap (coincident) expected 1 got {ov2}'); ok = False

    # 5) motion metrics
    drm = dwell_ratio_moving(gt, agent)  # matched dets all on p1 (static) -> 0
    if abs(drm - 0.0) > 1e-9:
        print(f'[FAIL] dwell_ratio_moving expected 0 got {drm}'); ok = False

    print('SELFTEST PASSED' if ok else 'SELFTEST FAILED')
    return ok


# ======================================================================
# CLI
# ======================================================================

def _parse_args():
    ap = argparse.ArgumentParser('Ground-truth metrics for the fisheye coverage task')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--data_path', type=str)
    ap.add_argument('--detector_weights', type=str, default=None)
    ap.add_argument('--steps', type=int, default=720)
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed_list', type=str, default=None)
    ap.add_argument('--state_dim', type=int, default=29)
    ap.add_argument('--full_map', action='store_true',
                    help='Observation carries the full coverage grid (state_dim 29 + K*M). '
                         'Must match how the checkpoint was trained.')
    ap.add_argument('--policies', type=str, default='rl,random,sweep,greedy')
    ap.add_argument('--ppo_dir', type=str, default=None, help='dir with tracker_best.pt/explorer_best.pt from ppo_continuous.py')
    ap.add_argument('--tracker', type=str, default=None)
    ap.add_argument('--explorer', type=str, default=None)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--gate_deg', type=float, default=15.0)
    ap.add_argument('--ov_target_tilt', type=float, default=48.0,
                    help='polar angle (deg from nadir) for the overlap-optimized sweep; '
                         '>=45 makes the two FOV cones disjoint (near-zero overlap).')
    ap.add_argument('--loop_video', action='store_true',
                    help='Wrap the clip to fill --steps (legacy). Default: OFF — each '
                         'episode ends at the true clip end (no replay contamination).')
    ap.add_argument('--random_clips', action='store_true',
                    help='Draw clips randomly per episode (legacy). Default: OFF — episode '
                         'ep uses clip (ep mod n_clips) so all clips are covered equally.')
    ap.add_argument('--out', type=str, default=None)
    return ap.parse_args()


def _aggregate(values):
    a = np.array([v for v in values if v == v], dtype=float)  # drop nans
    if a.size == 0:
        return {'mean': float('nan'), 'std': float('nan'), 'n': 0}
    return {'mean': float(a.mean()),
            'std': float(a.std(ddof=1)) if a.size > 1 else 0.0,
            'n': int(a.size)}


def main():
    args = _parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if not args.data_path:
        print('error: --data_path required (or use --selftest)'); sys.exit(2)

    import torch
    from environments.dual_fisheye_env import DualFisheyePTZEnvironment, NUM_ACTIONS
    from models.agent.mpdqn_agent import MPDQNAgent
    from baselines.evaluate import (PPOContinuousPolicy, RandomPolicy, SweepPolicy, GreedyPolicy,
                                     HeuristicPolicy, CoordinatedSweepPolicy, RLPolicy,
                                     HybridPolicy, SmartHybridPolicy,
                                     OverlapOptimizedSweepPolicy)

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    env = DualFisheyePTZEnvironment({
        'data_path': args.data_path, 'max_steps': args.steps,
        'state_dim': args.state_dim, 'detector_weights': args.detector_weights,
        'loop_video': bool(args.loop_video),  # default False: end at true clip end
        'full_map': bool(args.full_map),
    })

    if args.seed_list:
        seeds = [int(s) for s in args.seed_list.split(',') if s.strip() != '']
    else:
        seeds = list(range(args.seeds))
    requested = [p.strip() for p in args.policies.split(',') if p.strip()]
    print(f'[cfg] clips={len(env._host.video_paths)} steps={args.steps} '
          f'episodes/seed={args.episodes} seeds={seeds} gate={args.gate_deg} '
          f'policies={requested}')

    def build(name):
        if name == 'random':
            return RandomPolicy(np.random.default_rng(0))  # rng reset per seed below
        if name == 'sweep':
            return SweepPolicy()
        if name == 'greedy':
            return GreedyPolicy()
        if name == 'heuristic':
            return HeuristicPolicy()
        if name == 'coordsweep':
            return CoordinatedSweepPolicy()
        if name == 'ovsweep':
            return OverlapOptimizedSweepPolicy(target_tilt=args.ov_target_tilt)
        if name == 'rl':
            if not (args.tracker and args.explorer):
                return None
            tr = MPDQNAgent(state_dim=args.state_dim, num_actions=NUM_ACTIONS,
                            hidden_layers=[256, 128, 64], device=args.device)
            ex = MPDQNAgent(state_dim=args.state_dim, num_actions=NUM_ACTIONS,
                            hidden_layers=[256, 128, 64], device=args.device)
            tr.load(args.tracker); ex.load(args.explorer)
            return RLPolicy(tr, ex)
        if name == 'ppo':
            if not args.ppo_dir:
                return None
            import torch as _t
            sys.path.insert(0, '/app')
            from scripts.ppo_continuous import ActorCritic
            ags = []
            for nm in ('tracker', 'explorer'):
                ac = ActorCritic().to(args.device)
                ac.load_state_dict(_t.load(f'{args.ppo_dir}/{nm}_best.pt', map_location=args.device))
                ac.eval(); ags.append(ac)
            return PPOContinuousPolicy(ags, np.array([env.pan_speed, env.tilt_speed,
                                        env.zoom_speed], dtype=np.float32),
                                       env.tilt_range, env.zoom_range, args.device)
        if name == 'hybrid':
            if not args.tracker:
                return None
            tr = MPDQNAgent(state_dim=args.state_dim, num_actions=NUM_ACTIONS,
                            hidden_layers=[256, 128, 64], device=args.device)
            tr.load(args.tracker)
            return HybridPolicy(tr)
        if name == 'smarthybrid':
            if not args.tracker:
                return None
            tr = MPDQNAgent(state_dim=args.state_dim, num_actions=NUM_ACTIONS,
                            hidden_layers=[256, 128, 64], device=args.device)
            tr.load(args.tracker)
            return SmartHybridPolicy(tr)
        return None

    # policy -> metric_name -> list of per-(seed,episode) values
    collected = {p: {} for p in requested}
    metric_keys = None

    for seed in seeds:
        print(f'\n===== seed {seed} =====')
        for name in requested:
            pol = build(name)
            if pol is None:
                continue
            if name == 'random':
                pol.rng = np.random.default_rng(1000 + seed)
            for ep in range(args.episodes):
                env._host._rng = np.random.default_rng(seed * 100 + ep)  # paired draw (fallback)
                # Deterministic clip-balanced eval: episode ep -> clip (ep mod n_clips),
                # so every policy/seed is scored on the same clips, each equally weighted,
                # instead of a random draw that (with few clips) dominates the metrics.
                env._host.forced_clip_idx = None if args.random_clips else ep
                m, _ = rollout_gt(env, pol, args.steps, gate_deg=args.gate_deg)
                if metric_keys is None:
                    metric_keys = [k for k, v in m.items()]
                for k, v in m.items():
                    collected[name].setdefault(k, []).append(v)
                print(f'  {name:8s} seed{seed} ep{ep}: '
                      f'disc={m["discovery_rate"]:.2f} ttd={m["time_to_detect_steps"]:.1f} '
                      f'cov={m["coverage_pct"]:.2f} overlap={m["inter_camera_overlap"]:.2f} '
                      f'dwell_mov={m["dwell_ratio_moving"]:.2f}')

    env.close()

    summary = {p: {k: _aggregate(v) for k, v in collected[p].items()} for p in requested
               if collected[p]}

    print('\n' + '=' * 92)
    print('GROUND-TRUTH METRICS (mean over seeds x episodes)')
    print('=' * 92)
    cols = ['discovery_rate', 'time_to_detect_steps', 'observed_time_frac',
            'max_unobserved_gap', 'coverage_pct', 'inter_camera_overlap',
            'dwell_ratio_moving', 'time_to_detect_mover_steps']
    _hdr = {'discovery_rate': 'discov', 'time_to_detect_steps': 'ttd',
            'observed_time_frac': 'obsfrac', 'max_unobserved_gap': 'maxgap',
            'coverage_pct': 'covpct', 'inter_camera_overlap': 'overlap',
            'dwell_ratio_moving': 'dwell', 'time_to_detect_mover_steps': 'ttdmove'}
    print(f'{"policy":9s} ' + ' '.join(f'{_hdr[c]:>10s}' for c in cols))
    for p in sorted(summary, key=lambda k: -summary[k].get('discovery_rate', {}).get('mean', 0)):
        row = summary[p]
        cells = []
        for c in cols:
            mv = row.get(c, {}).get('mean', float('nan'))
            cells.append(f'{mv:>10.2f}')
        star = '  <-- RL' if p == 'rl' else ''
        print(f'{p:9s} ' + ' '.join(cells) + star)
    print('=' * 92)
    print('cols: discovery_rate, time_to_detect(steps), observed_time_frac (SUSTAINED '
          'coverage: mean fraction of a person\'s present-time actually observed), '
          'max_unobserved_gap (steps), coverage_pct, inter_camera_overlap, '
          'dwell_ratio_moving, time_to_detect_mover(steps)')

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'config': vars(args), 'summary': summary}, f, indent=2)
        print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
