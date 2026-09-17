"""Evaluate controllers on LOAF against HUMAN annotations.

The only thing swapped relative to Table 1 is where pixels and ground truth come
from: frames come from a LOAF sequence instead of our clips, and the reference is
the dataset's human annotation instead of the detector-based pseudo-reference.
The renderer, the detector, the state vector and the policies are unchanged.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.loaf import LoafSequence, load_annotations
from evaluation.metrics import (discovery_rate, time_to_detect, observed_time_frac,
                                max_unobserved_gap, inter_camera_overlap_frustum,
                                solid_angle_overlap_frustum)

STAY, PAN_L, PAN_R, ZOOM_I, ZOOM_O, TILT_U, TILT_D = range(7)


class StaticTiles:
    """No control at all: two fixed viewports, re-pinned every step."""
    name = 'tiles2'
    stochastic = False

    def __init__(self, pans=(-180.0, 0.0), tilt=45.0):
        self.pans, self.tilt = pans, tilt

    def reset(self): pass

    def pin(self, env):
        for i, p in enumerate(self.pans):
            env.pan[i] = p; env.tilt[i] = self.tilt; env.zoom[i] = 1.0

    def act(self, states, env):
        self.pin(env)
        return (STAY, STAY), (1.0, 1.0)


def attach(host, seq, frame_skip):
    """Point the environment at a LOAF sequence instead of a video file."""
    host.fish_cx, host.fish_cy, host.fish_R = seq.cx, seq.cy, seq.R
    host.fish_fov_deg = 180.0
    host.frame_count = len(seq) * frame_skip
    host.video_path = f'loaf:{seq.seq}'

    def _get_frame():
        idx = int(host.current_frame // frame_skip)
        if idx >= len(seq):
            host._clip_ended = True
            idx = len(seq) - 1
        return seq.read(idx)

    host._get_frame = _get_frame


def run(env, policy, seq, gt, steps, gate):
    host = env._host
    if hasattr(policy, 'reset'):
        policy.reset()
    states = env.reset()
    # reset() opens one of the built-in clips and overwrites the capture state;
    # re-point at the LOAF sequence afterwards so frames really come from it.
    attach(host, seq, host.frame_skip)
    if isinstance(policy, StaticTiles):
        policy.pin(env)
    agent, poses = [], []
    n = min(steps, len(gt))
    for t in range(n):
        dets = []
        for i in range(2):
            for d in env.last_detections[i]:
                dets.append(host.world_bearing(
                    d['bbox'], float(env.pan[i]), float(env.tilt[i]), float(env.zoom[i]),
                    host.ptz_out_w, host.ptz_out_h, host.base_fov))
        agent.append(dets)
        poses.append([(float(env.pan[i]), float(env.tilt[i]), float(env.zoom[i]))
                      for i in range(2)])
        a, p = policy.act(states, env)
        states, _r, done, _ = env.step(a, p)
        if done:
            break
    g = gt[:len(agent)]
    return {
        'n_ref_tracks': len({p['id'] for f in g for p in f}),
        'discovery': discovery_rate(g, agent, gate),
        'ttd': time_to_detect(g, agent, gate),
        'obsfrac': observed_time_frac(g, agent, gate),
        'maxgap': max_unobserved_gap(g, agent, gate),
        'overlap': inter_camera_overlap_frustum(poses, host.base_fov,
                                                host.ptz_out_w, host.ptz_out_h),
        'overlap_omega': solid_angle_overlap_frustum(poses, host.base_fov,
                                                     host.ptz_out_w, host.ptz_out_h),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--loaf_root', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--sequences', required=True)
    ap.add_argument('--policies', default='ovsweep,tiles2,random,rl')
    ap.add_argument('--detector_weights', required=True)
    ap.add_argument('--tracker'); ap.add_argument('--explorer')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--gate_deg', type=float, default=15.0)
    ap.add_argument('--state_dim', type=int, default=29)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    from environments.dual_fisheye_env import DualFisheyePTZEnvironment, NUM_ACTIONS
    from models.agent.mpdqn_agent import MPDQNAgent
    from baselines.evaluate import (RandomPolicy, OverlapOptimizedSweepPolicy,
                                    CoordinatedSweepPolicy, RLPolicy)

    env = DualFisheyePTZEnvironment({
        'data_path': 'data/test', 'max_steps': a.steps, 'state_dim': a.state_dim,
        'detector_weights': a.detector_weights, 'loop_video': False})
    host = env._host
    if host.detector is None:
        print('FATAL: detector unavailable'); sys.exit(2)

    ann = load_annotations(a.loaf_root, a.split, sequences=a.sequences.split(','))
    # resume: keep sequences already completed in a previous (possibly killed) run
    results = {}
    if os.path.exists(a.out):
        try:
            results = json.load(open(a.out))
            print('resuming; already done:', ','.join(sorted(results)))
        except Exception:
            results = {}
    for seq_id in a.sequences.split(','):
        if seq_id in results and all(n in results[seq_id] for n in a.policies.split(',')):
            print(f'  {seq_id} already complete, skipping')
            continue
        seq = LoafSequence(a.loaf_root, a.split, seq_id, ann=ann)
        seq.read(0)
        gt = seq.reference(max_frames=a.steps)
        attach(host, seq, host.frame_skip)
        results.setdefault(seq_id, {})
        for name in a.policies.split(','):
            if name == 'ovsweep':   pol = OverlapOptimizedSweepPolicy(48.0)
            elif name == 'tiles2':  pol = StaticTiles()
            elif name == 'random':  pol = RandomPolicy(np.random.default_rng(0))
            elif name == 'coordsweep': pol = CoordinatedSweepPolicy()
            elif name == 'rl':
                if not (a.tracker and a.explorer): continue
                tr = MPDQNAgent(state_dim=a.state_dim, num_actions=NUM_ACTIONS,
                                hidden_layers=[256, 128, 64], device=a.device)
                ex = MPDQNAgent(state_dim=a.state_dim, num_actions=NUM_ACTIONS,
                                hidden_layers=[256, 128, 64], device=a.device)
                tr.load(a.tracker); ex.load(a.explorer)
                pol = RLPolicy(tr, ex)
            else:
                continue
            r = run(env, pol, seq, gt, a.steps, a.gate_deg)
            results[seq_id][name] = r
            print(f"  {seq_id} {name:<10} disc {r['discovery']:.3f}  ttd {r['ttd']:6.2f}  "
                  f"obs {r['obsfrac']:.3f}  gap {r['maxgap']:6.2f}  ovlp {r['overlap']:.2f}"
                  f"  (ref tracks {r['n_ref_tracks']})", flush=True)
        json.dump(results, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
