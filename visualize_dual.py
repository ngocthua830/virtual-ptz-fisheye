"""Visualize the dual-PTZ fisheye agents.

Output: three-pane horizontal MP4 — fisheye overview (with red TRACKER and blue
EXPLORER FOV polygons), tracker view (red boxes around its target id), explorer
view (blue boxes around novel detections).
"""

import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environments.dual_fisheye_env import DualFisheyePTZEnvironment, NUM_ACTIONS
from models.agent.mpdqn_agent import MPDQNAgent


ACTION_NAMES = ['STAY', 'PAN_LEFT', 'PAN_RIGHT', 'ZOOM_IN', 'ZOOM_OUT', 'TILT_UP', 'TILT_DOWN']


def _reason_tracker(env, action, dets, r, info=None):
    """One short sentence that explains the tracker's behaviour this step."""
    info = info or {}
    vel = float(info.get('target_vel', 0.0))
    static = int(info.get('static_steps', 0))
    released = bool(info.get('lock_released', False))
    pulse = bool(info.get('pulse_active', False))

    # Behaviour driven by the new motion / zoom-regulation logic takes priority.
    if released:
        return f'Target sat still {static}+ steps -> handing off to a mover (id {env.target_id})'
    if pulse and action == 4:
        return 'Context pulse: zooming out to scan surroundings'
    if pulse:
        return 'Context pulse due - should widen to reassess scene'

    if env.target_id not in (None, -1):
        seen = any(d.get('track_id') == env.target_id for d in dets)
        if seen:
            tgt = next(d for d in dets if d.get('track_id') == env.target_id)
            bx = (tgt['bbox'][0] + tgt['bbox'][2]) / 2.0
            off = (bx - env._host.ptz_out_w / 2) / (env._host.ptz_out_w / 2)
            move = f'moving {vel:.1f} deg/s' if vel >= env.v_static_deg else f'still {static} steps'
            if static > 0 and vel < env.v_static_deg:
                if action == 4:
                    return f'id {env.target_id} {move} -> widening, low priority'
                return f'id {env.target_id} {move} - deprioritizing, watching for movers'
            if action == 0:
                return f'Tracking id {env.target_id} ({move}), centered ({off:+.2f})'
            if action in (1, 2):
                dirn = 'left' if action == 1 else 'right'
                return f'Tracking id {env.target_id} ({move}), panning {dirn} to follow'
            if action == 3:
                return f'Tracking id {env.target_id} ({move}), zooming to frame'
            if action == 4:
                return f'Tracking id {env.target_id} ({move}), widening for context'
            if action == 5:
                return f'Tracking id {env.target_id} ({move}), tilting up'
            if action == 6:
                return f'Tracking id {env.target_id} ({move}), tilting down'
        return f'Lost id {env.target_id} - auto-widening to re-acquire ({ACTION_NAMES[action].lower()})'
    if dets:
        movers = sum(1 for d in dets if env._vel_norm(d.get('track_id', -1)) > env.mover_thresh)
        tag = f', {movers} moving' if movers else ''
        return f'Acquiring lock - {len(dets)} candidate(s){tag}, preferring movers'
    if action == 0:
        return 'No target - holding view'
    if action in (1, 2):
        return f'Searching: panning {"left" if action == 1 else "right"}'
    if action in (5, 6):
        return f'Searching: tilting {"up" if action == 5 else "down"}'
    return f'Searching: {ACTION_NAMES[action].lower().replace("_", " ")}'


def _reason_explorer(env, action, dets, r, info):
    """One short sentence that explains the explorer's behaviour this step."""
    sep = abs(((env.pan[1] - env.pan[0] + 180) % 360) - 180)
    novel = [d for d in dets
             if d.get('track_id', -1) not in (-1, env.target_id)
             and d.get('track_id') not in env.seen_track_ids]
    if novel:
        ids = ','.join(str(d['track_id']) for d in novel[:3])
        return f'Discovered NEW id {ids} ({len(novel)} novel det(s))'
    if sep < env.overlap_thresh_deg:
        return f'Too close to tracker (sep {sep:.0f}deg) - pulling away'
    cov_e = info.get('coverage_rewards', (0.0, 0.0))[1]
    if cov_e > 0.4:
        if action in (5, 6):
            band = 'periphery' if env.tilt[1] > 45 else 'nadir'
            return f'Sweeping {band} - high stale-coverage gain ({cov_e:.2f})'
        return f'Visiting stale sector (cov gain {cov_e:.2f})'
    if dets:
        return f'Seeing {len(dets)} already-known person(s); pressing on'
    if action == 0:
        return f'Idling at pan {env.pan[1]:+.0f}deg - nothing nearby'
    return f'Continuing sweep ({ACTION_NAMES[action].lower().replace("_", " ")})'


def _draw_fov_quad(overlay, env, cam_idx, color, thickness=4, samples_per_edge=24):
    """Draw the back-projected PTZ-view outline onto the fisheye overview.

    Densely samples each edge and clamps to disk so the polygon traces the
    *curve* of the FOV on the hemisphere even when the view extends past the
    rim (rays sampling outside the disk get clamped to the rim).
    """
    if env._host.fish_R is None:
        env._host._detect_fisheye_circle(overlay)
    cx, cy, R = env._host.fish_cx, env._host.fish_cy, env._host.fish_R
    out_h, out_w = env._host.ptz_out_h, env._host.ptz_out_w
    pan = float(env.pan[cam_idx])
    tilt = float(env.tilt[cam_idx])
    zoom = float(env.zoom[cam_idx])
    fov = env._host.base_fov / max(zoom, 1e-6)
    f = (out_w / 2.0) / np.tan(np.radians(fov) / 2.0)

    half_w, half_h = out_w / 2.0, out_h / 2.0
    s = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
    top    = np.stack([-half_w + s * out_w,  np.full_like(s, -half_h)], axis=1)
    right  = np.stack([np.full_like(s,  half_w), -half_h + s * out_h], axis=1)
    bottom = np.stack([ half_w - s * out_w,  np.full_like(s,  half_h)], axis=1)
    left   = np.stack([np.full_like(s, -half_w),  half_h - s * out_h], axis=1)
    edge_pts = np.concatenate([top, right, bottom, left], axis=0).astype(np.float32)

    t = np.radians(tilt)
    p = np.radians(pan)
    cos_t, sin_t = np.cos(t), np.sin(t)
    cos_p, sin_p = np.cos(p), np.sin(p)
    fish_fov_rad = np.radians(env._host.fish_fov_deg)
    f_fish = R / (fish_fov_rad / 2.0)

    dx = edge_pts[:, 0]
    dy = edge_pts[:, 1]
    dz = -np.full_like(dx, f)
    ry = dy * cos_t - dz * sin_t
    rz = dy * sin_t + dz * cos_t
    wx = dx * cos_p - ry * sin_p
    wy = dx * sin_p + ry * cos_p
    wz = rz
    r3 = np.sqrt(wx * wx + wy * wy + wz * wz) + 1e-9
    theta = np.arccos(np.clip(-wz / r3, -1.0, 1.0))
    phi = np.arctan2(wy, wx)
    r_pix = np.clip(f_fish * theta, 0.0, R)

    xs = (cx + r_pix * np.cos(phi)).astype(np.int32)
    ys = (cy + r_pix * np.sin(phi)).astype(np.int32)
    poly = np.stack([xs, ys], axis=1)
    cv2.polylines(overlay, [poly], True, color, thickness)


# Layout constants for the 2x2 visualisation canvas.
#   +-----------+-----------+
#   | fisheye   | tracker   |
#   | (square)  | view      |
#   +-----------+-----------+
#   | explorer  | stats /   |
#   | view      | coverage  |
#   +-----------+-----------+
CELL_W, CELL_H = 720, 540
LOG_STRIP_H = 220                     # bottom log strip (rolling reasoning history)
LOG_MAX_ENTRIES = 9                   # how many lines fit visually
CANVAS_W, CANVAS_H = CELL_W * 2, CELL_H * 2 + LOG_STRIP_H
TRACKER_COLOR = (60, 60, 240)         # red-ish (BGR)
EXPLORER_COLOR = (240, 140, 30)       # orange-ish (BGR)
FRESH_COLOR = (90, 210, 90)


def _text(img, txt, x, y, scale=0.55, color=(240, 240, 240), thick=1, bg=True):
    """putText with a translucent dark background strip for readability."""
    if bg:
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick + 1)
        cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _fit_square_into(img, side, pad_color=(0, 0, 0)):
    """Letterbox a (possibly non-square) image into a side×side cell."""
    h, w = img.shape[:2]
    scale = side / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((side, side, 3), pad_color, dtype=np.uint8)
    oy, ox = (side - new_h) // 2, (side - new_w) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas, (ox, oy, scale)


def _fit_view_into(img, w, h):
    """Resize a view image to fill w×h (no letterbox); returns (img, x_scale, y_scale)."""
    h0, w0 = img.shape[:2]
    return cv2.resize(img, (w, h)), w / w0, h / h0


def _draw_coverage_heatmap(env, dst_w, dst_h):
    """Render the shared CoverageMap freshness as a small (azimuth × polar) image.

    Brighter = staler (more reward available). Includes peripheral weighting
    visually so the user can see which sectors the policy is being pulled toward.
    """
    cm = env.scene
    fresh = cm.freshness()              # (K, M)
    stale = 1.0 - fresh
    # Apply peripheral weights so the user sees what the agent is being
    # *rewarded* to look at, not just where it's been.
    weighted = stale * cm._polar_w[None, :]
    norm = weighted / max(weighted.max(), 1e-6)
    # Reshape so polar varies vertically (top = nadir, bottom = horizon) and
    # azimuth horizontally (left = -180, right = +180).
    heat = (norm.T * 255).astype(np.uint8)              # (M, K)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    heat_color = cv2.resize(heat_color, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
    return heat_color


def _draw_stats_panel(env, info, step, max_steps, dst_w, dst_h,
                      a_t, a_e, p_t, p_e, r_t, r_e):
    """Bottom-right cell: aggregates rewards, action history, coverage heatmap."""
    panel = np.full((dst_h, dst_w, 3), 24, dtype=np.uint8)

    total_t = sum(env.reward_history[0])
    total_e = sum(env.reward_history[1])
    total = total_t + total_e

    # --- top text block ---
    _text(panel, f'Step {step:>4}/{max_steps}', 16, 30, scale=0.7,
          color=(255, 255, 255), thick=2)
    _text(panel, f'Total  T  {total_t:+8.2f}', 16, 70,
          scale=0.65, color=TRACKER_COLOR, thick=2)
    _text(panel, f'       E  {total_e:+8.2f}', 16, 100,
          scale=0.65, color=EXPLORER_COLOR, thick=2)
    _text(panel, f'       sum  {total:+8.2f}', 16, 130,
          scale=0.65, color=(255, 255, 60), thick=2)

    # Step reward bars (visual)
    bar_x, bar_y_t, bar_y_e = 360, 60, 100
    bar_w_max = dst_w - bar_x - 30
    def _bar(y, val, color):
        cv2.rectangle(panel, (bar_x, y - 18), (bar_x + bar_w_max, y + 4),
                      (60, 60, 60), 1)
        v = max(-5.0, min(5.0, val))
        width = int(bar_w_max * (abs(v) / 5.0))
        mid = bar_x + bar_w_max // 2
        if v >= 0:
            cv2.rectangle(panel, (mid, y - 18), (mid + width // 2, y + 4), color, -1)
        else:
            cv2.rectangle(panel, (mid - width // 2, y - 18), (mid, y + 4), color, -1)
        cv2.line(panel, (mid, y - 22), (mid, y + 8), (140, 140, 140), 1)
    _bar(bar_y_t, r_t, TRACKER_COLOR)
    _bar(bar_y_e, r_e, EXPLORER_COLOR)
    _text(panel, f'step r:  T {r_t:+.2f}', bar_x, 130, scale=0.55,
          color=TRACKER_COLOR, thick=1)
    _text(panel, f'         E {r_e:+.2f}', bar_x + 220, 130, scale=0.55,
          color=EXPLORER_COLOR, thick=1)

    # --- scene-state row ---
    cov_t, cov_e = info.get('coverage_rewards', (0.0, 0.0))
    seen = info.get('seen_ids', 0)
    stale = info.get('stale_total', 0.0)
    _text(panel, f'seen ids: {seen}', 16, 180, scale=0.6, color=(120, 220, 255), thick=2)
    _text(panel, f'scene stale: {stale:.2f}', 16, 210, scale=0.55, color=(180, 220, 255))
    _text(panel, f'cov gain T {cov_t:.2f}  E {cov_e:.2f}', 16, 235, scale=0.5,
          color=(180, 220, 255))

    # --- coverage heatmap ---
    heat_w, heat_h = dst_w - 40, 160
    heat = _draw_coverage_heatmap(env, heat_w, heat_h)
    panel[dst_h - heat_h - 30:dst_h - 30, 20:20 + heat_w] = heat
    _text(panel, 'Shared CoverageMap  (top=nadir, bottom=rim;  bright=stale, dark=fresh)',
          20, dst_h - heat_h - 40, scale=0.45, color=(220, 220, 220))

    # --- action labels ---
    _text(panel, f'T act: {ACTION_NAMES[a_t]:<10s} p={p_t:.2f}', 16, 290,
          scale=0.55, color=TRACKER_COLOR, thick=1)
    _text(panel, f'E act: {ACTION_NAMES[a_e]:<10s} p={p_e:.2f}', 16, 320,
          scale=0.55, color=EXPLORER_COLOR, thick=1)

    return panel


def _draw_reasoning_log(log, dst_w, dst_h, current_step):
    """Render the rolling reasoning log strip at the bottom of the canvas.

    `log` is a deque of (step, cam, reason) where cam ∈ {'T', 'E'}. We only
    push to the deque when the agent's reasoning *changes* from the previous
    step (see the caller), so the strip stays an event log, not a per-frame
    spam. The most recent entry gets a left arrow and is brightened.
    """
    strip = np.full((dst_h, dst_w, 3), 18, dtype=np.uint8)
    cv2.rectangle(strip, (0, 0), (dst_w, 30), (40, 40, 40), -1)
    _text(strip, f'Reasoning log  (showing last {LOG_MAX_ENTRIES} events)',
          14, 22, scale=0.55, color=(220, 220, 220), thick=1, bg=False)

    # Colour key
    cv2.rectangle(strip, (dst_w - 360, 6), (dst_w - 350, 24), TRACKER_COLOR, -1)
    _text(strip, 'TRACKER', dst_w - 344, 22, scale=0.5,
          color=TRACKER_COLOR, thick=1, bg=False)
    cv2.rectangle(strip, (dst_w - 240, 6), (dst_w - 230, 24), EXPLORER_COLOR, -1)
    _text(strip, 'EXPLORER', dst_w - 224, 22, scale=0.5,
          color=EXPLORER_COLOR, thick=1, bg=False)

    if not log:
        _text(strip, '(no events yet)', 30, 60, scale=0.55,
              color=(140, 140, 140), thick=1, bg=False)
        return strip

    entries = list(log)[-LOG_MAX_ENTRIES:]
    line_h = 20
    y0 = 56
    for i, (s, cam, text) in enumerate(entries):
        is_latest = (i == len(entries) - 1)
        color = TRACKER_COLOR if cam == 'T' else EXPLORER_COLOR
        # Dim older lines so the eye lands on the most recent.
        if not is_latest:
            color = tuple(int(c * 0.65) for c in color)
        prefix = '> ' if is_latest else '  '
        line = f'{prefix}step {s:>4}  [{cam}]  {text}'
        _text(strip, line, 14, y0 + i * line_h, scale=0.5,
              color=color, thick=2 if is_latest else 1, bg=False)
    return strip


def visualize_episode(env, tracker, explorer, max_steps, save_path, fps=15):
    states = env.reset()
    out = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (CANVAS_W, CANVAS_H)
    )
    pbar = tqdm(total=max_steps, desc=os.path.basename(save_path))

    # Rolling log of reasoning events. (step, 'T'/'E', sentence).
    # Append only when sentence changes from previous, so consecutive identical
    # captions don't fill the buffer.
    reason_log = deque(maxlen=128)
    prev_reason = {'T': None, 'E': None}

    for step in range(max_steps):
        a_t, p_t = tracker.choose_action(states[0], training=False)
        a_e, p_e = explorer.choose_action(states[1], training=False)
        next_states, (r_t, r_e), done, info = env.step((a_t, a_e), (p_t, p_e))
        states = next_states

        frame = info['frame']
        view_t, view_e = info['views']
        dets_t, dets_e = info['detections']

        # --- fisheye overview (top-left cell) ---
        overview = frame.copy()
        cv2.circle(overview, (int(env._host.fish_cx), int(env._host.fish_cy)),
                   int(env._host.fish_R), (180, 180, 180), 3)
        _draw_fov_quad(overview, env, 0, color=TRACKER_COLOR, thickness=6)
        _draw_fov_quad(overview, env, 1, color=EXPLORER_COLOR, thickness=6)
        # Square fisheye stays square — letterbox into the cell.
        # Use the shorter side so we don't over-shrink.
        side = min(CELL_W, CELL_H)
        fisheye_cell, _ = _fit_square_into(overview, side)
        # Centre the square in the cell.
        fisheye_cell_canvas = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        oy = (CELL_H - side) // 2
        ox = (CELL_W - side) // 2
        fisheye_cell_canvas[oy:oy + side, ox:ox + side] = fisheye_cell
        _text(fisheye_cell_canvas, 'Fisheye overview', 16, 32, scale=0.7,
              color=(255, 255, 255), thick=2)
        _text(fisheye_cell_canvas, 'red = TRACKER FOV', 16, 64, scale=0.5, color=TRACKER_COLOR)
        _text(fisheye_cell_canvas, 'orange = EXPLORER FOV', 16, 90, scale=0.5,
              color=EXPLORER_COLOR)
        _text(fisheye_cell_canvas,
              f'sep azimuth: {abs(((env.pan[1] - env.pan[0] + 180) % 360) - 180):.0f}deg',
              16, CELL_H - 18, scale=0.5, color=(220, 220, 255))

        # --- per-step reasoning captions ---
        reason_t = _reason_tracker(env, a_t, dets_t, r_t, info)
        reason_e = _reason_explorer(env, a_e, dets_e, r_e, info)

        # --- tracker pane (top-right cell) ---
        vt, sx_t, sy_t = _fit_view_into(view_t, CELL_W, CELL_H)
        # red header strip
        cv2.rectangle(vt, (0, 0), (CELL_W, 36), TRACKER_COLOR, -1)
        _text(vt, f'TRACKER   pan {env.pan[0]:+.0f}deg   tilt {env.tilt[0]:+.0f}deg   '
                  f'zoom {env.zoom[0]:.1f}x   ->  {ACTION_NAMES[a_t]} ({p_t:.2f})',
              12, 26, scale=0.55, color=(255, 255, 255), thick=2, bg=False)
        for d in dets_t:
            x1, y1, x2, y2 = d['bbox']
            x1, x2 = int(x1 * sx_t), int(x2 * sx_t)
            y1, y2 = int(y1 * sy_t), int(y2 * sy_t)
            tid = d.get('track_id', -1)
            is_tgt = tid == env.target_id and tid != -1
            color = TRACKER_COLOR if is_tgt else (90, 220, 90)
            cv2.rectangle(vt, (x1, y1), (x2, y2), color, 3)
            _text(vt, f'{"TARGET " if is_tgt else ""}id {tid}',
                  x1, max(y1 - 6, 20), scale=0.5, color=color, thick=1)
        # Reasoning caption strip at bottom (full-width, dark bg, white text)
        cv2.rectangle(vt, (0, CELL_H - 64), (CELL_W, CELL_H - 28), (0, 0, 0), -1)
        _text(vt, reason_t, 12, CELL_H - 40, scale=0.55,
              color=(240, 240, 240), thick=1, bg=False)
        _text(vt, f'target id: {env.target_id}   dets: {len(dets_t)}   r {r_t:+.2f}',
              12, CELL_H - 8, scale=0.5, color=(255, 255, 60), thick=1, bg=False)

        # --- explorer pane (bottom-left cell) ---
        ve, sx_e, sy_e = _fit_view_into(view_e, CELL_W, CELL_H)
        cv2.rectangle(ve, (0, 0), (CELL_W, 36), EXPLORER_COLOR, -1)
        _text(ve, f'EXPLORER  pan {env.pan[1]:+.0f}deg   tilt {env.tilt[1]:+.0f}deg   '
                  f'zoom {env.zoom[1]:.1f}x   ->  {ACTION_NAMES[a_e]} ({p_e:.2f})',
              12, 26, scale=0.55, color=(255, 255, 255), thick=2, bg=False)
        seen = env.seen_track_ids
        for d in dets_e:
            x1, y1, x2, y2 = d['bbox']
            x1, x2 = int(x1 * sx_e), int(x2 * sx_e)
            y1, y2 = int(y1 * sy_e), int(y2 * sy_e)
            tid = d.get('track_id', -1)
            novel = tid != -1 and tid not in seen and tid != env.target_id
            color = EXPLORER_COLOR if novel else (90, 220, 90)
            cv2.rectangle(ve, (x1, y1), (x2, y2), color, 3)
            _text(ve, f'{"NEW " if novel else ""}id {tid}',
                  x1, max(y1 - 6, 20), scale=0.5, color=color, thick=1)
        cv2.rectangle(ve, (0, CELL_H - 64), (CELL_W, CELL_H - 28), (0, 0, 0), -1)
        _text(ve, reason_e, 12, CELL_H - 40, scale=0.55,
              color=(240, 240, 240), thick=1, bg=False)
        _text(ve, f'dets: {len(dets_e)}   r {r_e:+.2f}',
              12, CELL_H - 8, scale=0.5, color=(255, 255, 60), thick=1, bg=False)

        # --- stats / coverage (bottom-right cell) ---
        stats_cell = _draw_stats_panel(env, info, step, max_steps, CELL_W, CELL_H,
                                       a_t, a_e, p_t, p_e, r_t, r_e)

        # --- rolling reasoning log: push only when text changes ---
        if reason_t != prev_reason['T']:
            reason_log.append((step, 'T', reason_t))
            prev_reason['T'] = reason_t
        if reason_e != prev_reason['E']:
            reason_log.append((step, 'E', reason_e))
            prev_reason['E'] = reason_e
        log_strip = _draw_reasoning_log(reason_log, CANVAS_W, LOG_STRIP_H, step)

        top = np.hstack([fisheye_cell_canvas, vt])
        bot = np.hstack([ve, stats_cell])
        canvas = np.vstack([top, bot, log_strip])
        out.write(canvas)

        pbar.update(1)
        pbar.set_postfix(T=f'{sum(env.reward_history[0]):+.1f}',
                         E=f'{sum(env.reward_history[1]):+.1f}')
        if done:
            break

    pbar.close()
    out.release()
    return sum(env.reward_history[0]), sum(env.reward_history[1])


def main():
    ap = argparse.ArgumentParser('Visualize dual fisheye PTZ agents')
    ap.add_argument('--tracker', type=str, required=True)
    ap.add_argument('--explorer', type=str, required=True)
    ap.add_argument('--data_path', type=str, default='./data/1773114828772.mp4')
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--frame_skip', type=int, default=1)
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--output', type=str, default='visualization_dual')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--state_dim', type=int, default=29)
    ap.add_argument('--detector_weights', type=str, default=None,
                    help='Path to detector .pt (default yolov8n.pt). '
                         'OBB models auto-detected via task tag.')
    ap.add_argument('--yolo_conf', type=float, default=None,
                    help='Detection confidence threshold (overrides env default).')
    args = ap.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    os.makedirs(args.output, exist_ok=True)
    env_cfg = {
        'data_path': args.data_path,
        'max_steps': args.steps,
        'state_dim': args.state_dim,
        'frame_skip': args.frame_skip,
        'detector_weights': args.detector_weights,
    }
    if args.yolo_conf is not None:
        env_cfg['yolo_conf'] = args.yolo_conf
    env = DualFisheyePTZEnvironment(env_cfg)
    tracker = MPDQNAgent(state_dim=args.state_dim, num_actions=NUM_ACTIONS,
                         hidden_layers=[256, 128, 64], device=args.device)
    explorer = MPDQNAgent(state_dim=args.state_dim, num_actions=NUM_ACTIONS,
                          hidden_layers=[256, 128, 64], device=args.device)
    tracker.load(args.tracker)
    explorer.load(args.explorer)

    for ep in range(args.episodes):
        save_path = os.path.join(args.output, f'episode_{ep+1}.mp4')
        rt, re = visualize_episode(env, tracker, explorer, args.steps, save_path, fps=args.fps)
        print(f'Episode {ep+1}: tracker {rt:+.2f}  explorer {re:+.2f}  total {rt+re:+.2f} -> {save_path}')

    env.close()


if __name__ == '__main__':
    main()
