"""Run the virtual-PTZ policies on FRIDA, whose person IDs are GROUND TRUTH.

Mirrors ``run_loaf.py`` exactly -- same renderer, detector, state vector, policies
and metrics -- with one difference that is the entire point: the reference tracks
come from the annotated ``person_id`` instead of from our greedy association. Any
identity-based number here is therefore free of the association gate that every
LOAF number inherits.

Usage:
  python -m evaluation.run_frida --frida_root /frida/FRIDA --cameras 1:1,1:2,1:3 \
     --policies ovsweep,tiles2,random,rl --detector_weights DET.pt \
     --tracker t.pt --explorer e.pt --steps 300 --out results/frida/eval.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.frida import FridaSequence, attach
from evaluation.metrics import (discovery_rate, time_to_detect, observed_time_frac,
                                max_unobserved_gap, inter_camera_overlap_frustum,
                                solid_angle_overlap_frustum, per_frame_pr)
from evaluation.run_loaf import StaticTiles


def run(env, policy, seq, gt, steps, gate):
    host = env._host
    if hasattr(policy, 'reset'):
        policy.reset()
    states = env.reset()
    # reset() opens one of the built-in clips and overwrites the capture state;
    # re-point at the FRIDA sequence afterwards so frames really come from it.
    attach(host, seq, host.frame_skip)
    if isinstance(policy, StaticTiles):
        policy.pin(env)
    agent, poses = [], []
    n = min(steps, len(gt))
    for _t in range(n):
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
    out = {
        'n_ref_tracks': len({p['id'] for f in g for p in f}),
        'gt_ids_are_annotated': True,          # the whole reason for this run
        'discovery': discovery_rate(g, agent, gate),
        'ttd': time_to_detect(g, agent, gate),
        'obsfrac': observed_time_frac(g, agent, gate),
        'maxgap': max_unobserved_gap(g, agent, gate),
        'overlap': inter_camera_overlap_frustum(poses, host.base_fov,
                                                host.ptz_out_w, host.ptz_out_h),
        'overlap_omega': solid_angle_overlap_frustum(poses, host.base_fov,
                                                     host.ptz_out_w, host.ptz_out_h),
    }
    out.update({('pr_' + k): v for k, v in per_frame_pr(
        g, agent, poses, gate, host.base_fov,
        host.ptz_out_w, host.ptz_out_h).items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frida_root', required=True, help='.../FRIDA')
    ap.add_argument('--cameras', required=True,
                    help='comma list of SEGMENT:CAMERA, e.g. 1:1,1:2,1:3')
    ap.add_argument('--policies', default='ovsweep,tiles2,random,rl')
    ap.add_argument('--detector_weights', required=True)
    ap.add_argument('--tracker', default=None)
    ap.add_argument('--explorer', default=None)
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
        print('FATAL: detector unavailable')
        sys.exit(2)

    results = {}
    if os.path.exists(a.out):
        try:
            results = json.load(open(a.out))
            print('resuming; already done:', ','.join(sorted(results)))
        except Exception:
            results = {}

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    for spec in a.cameras.split(','):
        segment, camera = (int(x) for x in spec.split(':'))
        key = f'S{segment}C{camera}'
        if key in results and all(n in results[key] for n in a.policies.split(',')):
            print(f'  {key} already complete, skipping')
            continue
        seq = FridaSequence(a.frida_root, segment, camera)
        gt = seq.reference(max_frames=a.steps)
        attach(host, seq, host.frame_skip)
        results.setdefault(key, {})
        for name in a.policies.split(','):
            if name == 'ovsweep':
                pol = OverlapOptimizedSweepPolicy(48.0)
            elif name == 'tiles2':
                pol = StaticTiles()
            elif name == 'random':
                pol = RandomPolicy(np.random.default_rng(0))
            elif name == 'coordsweep':
                pol = CoordinatedSweepPolicy()
            elif name == 'rl':
                if not (a.tracker and a.explorer):
                    continue
                tr = MPDQNAgent(state_dim=a.state_dim, num_actions=NUM_ACTIONS,
                                hidden_layers=[256, 128, 64], device=a.device)
                ex = MPDQNAgent(state_dim=a.state_dim, num_actions=NUM_ACTIONS,
                                hidden_layers=[256, 128, 64], device=a.device)
                tr.load(a.tracker)
                ex.load(a.explorer)
                pol = RLPolicy(tr, ex)
            else:
                continue
            r = run(env, pol, seq, gt, a.steps, a.gate_deg)
            results[key][name] = r
            print(f"  {key} {name:<10} disc {r['discovery']:.3f}  ttd {r['ttd']:6.2f}  "
                  f"obs {r['obsfrac']:.3f}  gap {r['maxgap']:6.2f}  "
                  f"P {r['pr_precision']:.3f} R {r['pr_recall']:.3f}  "
                  f"(GT tracks {r['n_ref_tracks']})", flush=True)
        json.dump(results, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
