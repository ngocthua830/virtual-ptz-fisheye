"""Render an episode of the fisheye PTZ agent to a side-by-side MP4."""

import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environments.fisheye_env import FisheyePTZEnvironment
from models.agent.mpdqn_agent import MPDQNAgent


ACTION_NAMES = ['STAY', 'PAN_LEFT', 'PAN_RIGHT', 'ZOOM_IN', 'ZOOM_OUT', 'TILT_UP', 'TILT_DOWN']


def _reason(env, action, dets, info):
    """One short sentence that explains the single-cam agent's behaviour this step."""
    novel = info.get('novel_ids', 0)
    cov = info.get('coverage_reward', 0.0)
    if dets:
        if env.target_id not in (None, -1):
            seen = any(d.get('track_id') == env.target_id for d in dets)
            if seen:
                if action == 0:
                    return f'Locked on id {env.target_id} - holding'
                if action in (1, 2):
                    dirn = 'left' if action == 1 else 'right'
                    return f'Locked on id {env.target_id}, panning {dirn} to recenter'
                if action == 3:
                    return f'Locked on id {env.target_id}, zooming in'
                if action == 4:
                    return f'Locked on id {env.target_id}, zooming out'
                if action == 5:
                    return f'Locked on id {env.target_id}, tilting up'
                if action == 6:
                    return f'Locked on id {env.target_id}, tilting down'
        if novel > 0:
            return f'Discovered {novel} new person id(s)'
        return f'Acquiring lock - {len(dets)} candidate(s) in view'
    # No detections.
    if cov > 0.4:
        if action in (5, 6):
            band = 'periphery' if env.tilt > 45 else 'nadir'
            return f'Sweeping {band} - high coverage gain ({cov:.2f})'
        return f'Visiting stale sector (cov gain {cov:.2f})'
    if action == 0:
        return f'No target - holding view at pan {env.pan:+.0f}deg'
    if action in (1, 2):
        return f'Searching: panning {"left" if action == 1 else "right"}'
    if action in (5, 6):
        return f'Searching: tilting {"up" if action == 5 else "down"}'
    return f'Searching: {ACTION_NAMES[action].lower().replace("_", " ")}'


def _draw_fisheye_overlay(fisheye, env, samples_per_edge=24):
    """Sketch the current PTZ FOV outline on the fisheye overview image.

    Densely samples each edge of the perspective rectangle so the polygon
    traces the *curve* its rays make on the disk, and clamps r_pix to the
    rim instead of dropping out-of-disk corners. With this, an agent looking
    past the horizon shows up as a polygon hugging the disk edge.
    """
    overlay = fisheye.copy()
    if env.fish_R is None:
        env._detect_fisheye_circle(overlay)
    cx, cy, R = env.fish_cx, env.fish_cy, env.fish_R

    cv2.circle(overlay, (int(cx), int(cy)), int(R), (200, 200, 200), 2)
    cv2.circle(overlay, (int(cx), int(cy)), 6, (255, 255, 255), -1)

    out_h, out_w = env.ptz_out_h, env.ptz_out_w
    fov = env.base_fov / max(env.zoom, 1e-6)
    f = (out_w / 2.0) / np.tan(np.radians(fov) / 2.0)

    # Walk the rectangle perimeter, sampling many points.
    half_w, half_h = out_w / 2.0, out_h / 2.0
    s = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
    top    = np.stack([-half_w + s * out_w,  np.full_like(s, -half_h)], axis=1)
    right  = np.stack([np.full_like(s,  half_w), -half_h + s * out_h], axis=1)
    bottom = np.stack([ half_w - s * out_w,  np.full_like(s,  half_h)], axis=1)
    left   = np.stack([np.full_like(s, -half_w),  half_h - s * out_h], axis=1)
    edge_pts = np.concatenate([top, right, bottom, left], axis=0).astype(np.float32)

    t = np.radians(env.tilt)
    p = np.radians(env.pan)
    cos_t, sin_t = np.cos(t), np.sin(t)
    cos_p, sin_p = np.cos(p), np.sin(p)
    fish_fov_rad = np.radians(env.fish_fov_deg)
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
    r_pix = np.clip(f_fish * theta, 0.0, R)         # clamp instead of drop

    xs = (cx + r_pix * np.cos(phi)).astype(np.int32)
    ys = (cy + r_pix * np.sin(phi)).astype(np.int32)
    poly = np.stack([xs, ys], axis=1)
    cv2.polylines(overlay, [poly], True, (0, 0, 255), 4)

    # Indicate when the FOV truly extends past the rim.
    if (f_fish * theta).max() > R + 1:
        cv2.putText(overlay, 'view extends past rim',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return overlay


LOG_STRIP_H = 200
LOG_MAX_ENTRIES = 8


def _draw_reason_log(log, dst_w, dst_h):
    """Bottom strip: rolling history of reasoning lines (latest at bottom)."""
    strip = np.full((dst_h, dst_w, 3), 18, dtype=np.uint8)
    cv2.rectangle(strip, (0, 0), (dst_w, 30), (40, 40, 40), -1)
    cv2.putText(strip, f'Reasoning log  (last {LOG_MAX_ENTRIES} events)',
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    if not log:
        cv2.putText(strip, '(no events yet)', (24, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1, cv2.LINE_AA)
        return strip
    entries = list(log)[-LOG_MAX_ENTRIES:]
    line_h = 20
    y0 = 56
    for i, (s, text) in enumerate(entries):
        is_latest = i == len(entries) - 1
        color = (90, 220, 90) if is_latest else (160, 160, 160)
        prefix = '> ' if is_latest else '  '
        cv2.putText(strip, f'{prefix}step {s:>4}  {text}', (14, y0 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                    2 if is_latest else 1, cv2.LINE_AA)
    return strip


def visualize_episode(env, agent, max_steps, save_path, fps=10):
    state = env.reset()
    pane_w, pane_h = 960, 540
    out = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
        (pane_w * 2, pane_h + LOG_STRIP_H),
    )
    pbar = tqdm(total=max_steps, desc=os.path.basename(save_path))
    reason_log = deque(maxlen=64)
    prev_reason = None

    for step in range(max_steps):
        action, action_param = agent.choose_action(state, training=False)
        next_state, reward, done, info = env.step(action, action_param=action_param)
        state = next_state

        ptz_view = info['ptz_view']
        fisheye = info['fisheye_frame']
        detections = info['detections']

        reason = _reason(env, action, detections, info)

        # PTZ pane annotations
        cv2.putText(ptz_view,
                    f'Step {step}  Pan {env.pan:+.1f}  Tilt {env.tilt:+.1f}  Zoom {env.zoom:.1f}x',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(ptz_view,
                    f'Action: {ACTION_NAMES[action]} ({action_param:.2f})',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(ptz_view, f'Detections: {len(detections)}',
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d['bbox']]
            tid = d.get('track_id', -1)
            is_tgt = hasattr(env, 'target_id') and tid == env.target_id and tid != -1
            color = (0, 0, 255) if is_tgt else (0, 255, 0)
            cv2.rectangle(ptz_view, (x1, y1), (x2, y2), color, 2)
            label = f'{"TARGET " if is_tgt else ""}ID:{tid} {d["confidence"]:.2f}'
            cv2.putText(ptz_view, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        overview = _draw_fisheye_overlay(fisheye, env)
        overview = cv2.resize(overview, (pane_w, pane_h))
        cv2.putText(overview, 'Fisheye (red = PTZ FOV)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        combined = np.hstack([overview, ptz_view])
        # Reasoning caption: full-width dark strip near bottom for at-a-glance "why"
        h2 = pane_h
        cv2.rectangle(combined, (0, h2 - 60), (combined.shape[1], h2 - 24),
                      (0, 0, 0), -1)
        cv2.putText(combined, reason, (10, h2 - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
        total_r = sum(env.reward_history)
        cv2.putText(combined, f'Total reward: {total_r:+.2f}',
                    (10, h2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Rolling log: push only when reason changes
        if reason != prev_reason:
            reason_log.append((step, reason))
            prev_reason = reason
        log_strip = _draw_reason_log(reason_log, combined.shape[1], LOG_STRIP_H)
        combined = np.vstack([combined, log_strip])
        out.write(combined)

        pbar.update(1)
        pbar.set_postfix(reward=f'{total_r:+.2f}')
        if done:
            break

    pbar.close()
    out.release()
    return sum(env.reward_history)


def main():
    ap = argparse.ArgumentParser('Visualize fisheye PTZ agent')
    ap.add_argument('--model', type=str, required=False, default=None)
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--steps', type=int, default=64)
    ap.add_argument('--output', type=str, default='visualization')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--data_path', type=str, default='./data/1767931881294.mp4')
    ap.add_argument('--state_dim', type=int, default=24)
    ap.add_argument('--frame_skip', type=int, default=5,
                    help='Frames advanced per env step (1 = every frame)')
    ap.add_argument('--fps', type=int, default=10, help='Output video FPS')
    ap.add_argument('--detector_weights', type=str, default=None,
                    help='Path to detector .pt (default yolov8n.pt). '
                         'OBB models auto-detected via task tag.')
    args = ap.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    os.makedirs(args.output, exist_ok=True)
    env = FisheyePTZEnvironment({
        'data_path': args.data_path,
        'max_steps': args.steps,
        'state_dim': args.state_dim,
        'frame_skip': args.frame_skip,
        'detector_weights': args.detector_weights,
    })
    agent = MPDQNAgent(
        state_dim=args.state_dim,
        num_actions=7,
        hidden_layers=[256, 128, 64],
        device=args.device,
    )
    if args.model and os.path.exists(args.model):
        agent.load(args.model)
    else:
        print('No model loaded — using random policy.')

    for ep in range(args.episodes):
        save_path = os.path.join(args.output, f'episode_{ep+1}.mp4')
        r = visualize_episode(env, agent, args.steps, save_path, fps=args.fps)
        print(f'Episode {ep+1} reward {r:+.2f} -> {save_path}')

    env.close()


if __name__ == '__main__':
    main()
