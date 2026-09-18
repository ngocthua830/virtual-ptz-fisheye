"""Corrected horizon experiment (addresses review #1,#2,#4,#5).

The old horizon eval recomputed discovery at each H from a *separate* rollout whose
oracle cohort (denominator) grew with H, so discovery could fall as H rose -- different
horizons scored different people. Here we instead run ONE full 300-step deterministic
rollout per (policy, clip, seed) and derive every horizon from it against a FIXED cohort:

  * cohort_full   = every identity the oracle sees anywhere in the full clip.
  * discovery@H   = |{ids first detected by the agent at step <= H}| / |cohort_full|
                    -> monotone non-decreasing in H (the property the old metric lacked).
  * cum_found@H   = absolute count of identities discovered by step H (denominator-free).
  * ttd@H         = mean steps-to-first-detection over ids discovered by H.

Each of the 3 held-out test clips is rolled out once (no clip is double-counted -> fixes
the "6 clips" mislabel). H is a *maximum* budget: a rollout ends at the true clip end if
that is < 300 steps (we also record the real clip length).

Usage (one model, or baselines):
  python -m evaluation.horizon_curve --data_path data/test --detector_weights DET.pt \
     --policies rl --tracker t.pt --explorer e.pt --out out.json
  python -m evaluation.horizon_curve --data_path data/test --detector_weights DET.pt \
     --policies random,sweep,coordsweep --random_seeds 0,1,2,3,4 --out base.json
"""
import argparse, json, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import (rollout_gt, _discovery_events, _first_present)

HORIZONS = [5, 10, 15, 20, 30, 60, 90, 150, 300]


def _curve_from_logs(logs, gate_deg):
    """Given one full rollout's logs, return per-H fixed-cohort stats."""
    gt, agent = logs['gt'], logs['agent']
    n = len(gt)
    cohort_full = _first_present(gt)                 # id -> first appearance step (full clip)
    denom = len(cohort_full)
    out = {}
    for H in HORIZONS:
        Hc = min(H, n)
        seen = _discovery_events(gt[:Hc], agent[:Hc], gate_deg)   # id -> first-detect step (<=H)
        found = len(seen)
        lat = [seen[i] - cohort_full[i] for i in seen]
        out[str(H)] = {
            'disc_fixed': (found / denom) if denom else float('nan'),  # / full-clip cohort
            'cum_found': found,                                        # absolute count
            'appeared_by_H': sum(1 for f in cohort_full.values() if f < Hc),
            'ttd': float(np.mean(lat)) if lat else float('nan'),
            'real_steps': n,
        }
    # Per-person survival data for time-to-detection (review #14, right-censoring):
    # for every cohort person, latency from appearance to first detection; undetected
    # people are right-censored at their observation window (clip end - appearance).
    seen_full = _discovery_events(gt, agent, gate_deg)
    surv = []
    for pid, appear in cohort_full.items():
        w = n - appear                        # steps this person was observable
        if w <= 0:
            continue
        if pid in seen_full:
            surv.append([seen_full[pid] - appear, 1])   # [latency, event=detected]
        else:
            surv.append([w, 0])                          # [censoring time, event=0]
    out['cohort_full'] = denom
    out['survival'] = surv
    return out


def main():
    ap = argparse.ArgumentParser('corrected horizon curve')
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--detector_weights', required=True)
    ap.add_argument('--policies', default='rl')
    ap.add_argument('--tracker'); ap.add_argument('--explorer')
    ap.add_argument('--state_dim', type=int, default=29)
    ap.add_argument('--gate_deg', type=float, default=15.0)
    ap.add_argument('--random_seeds', default='0,1,2,3,4',
                    help='seeds for the stochastic random policy (multiple -> mean+/-std)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import torch
    from environments.dual_fisheye_env import DualFisheyePTZEnvironment, NUM_ACTIONS
    from models.agent.mpdqn_agent import MPDQNAgent
    from baselines.evaluate import (RandomPolicy, SweepPolicy, GreedyPolicy,
                                     HeuristicPolicy, CoordinatedSweepPolicy, RLPolicy,
                                     HybridPolicy, SmartHybridPolicy,
                                     OverlapOptimizedSweepPolicy)
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    env = DualFisheyePTZEnvironment({
        'data_path': args.data_path, 'max_steps': 300, 'state_dim': args.state_dim,
        'detector_weights': args.detector_weights, 'loop_video': False,
    })
    n_clips = len(env._host.video_paths)

    def build_rl():
        tr = MPDQNAgent(state_dim=args.state_dim, num_actions=NUM_ACTIONS,
                        hidden_layers=[256, 128, 64], device=args.device)
        ex = MPDQNAgent(state_dim=args.state_dim, num_actions=NUM_ACTIONS,
                        hidden_layers=[256, 128, 64], device=args.device)
        tr.load(args.tracker); ex.load(args.explorer)
        return RLPolicy(tr, ex)

    def run_policy(make_pol, clip):
        env._host.forced_clip_idx = clip
        env._host._rng = np.random.default_rng(1234 + clip)
        _m, logs = rollout_gt(env, make_pol(), 300, gate_deg=args.gate_deg)
        return _curve_from_logs(logs, args.gate_deg)

    result = {'config': vars(args), 'n_clips': n_clips, 'horizons': HORIZONS, 'policies': {}}
    for name in [p.strip() for p in args.policies.split(',') if p.strip()]:
        print(f'=== policy {name} ===', flush=True)
        if name == 'rl':
            if not (args.tracker and args.explorer):
                print('  [skip] rl needs --tracker/--explorer'); continue
            runs = [run_policy(build_rl, c) for c in range(n_clips)]           # 1 pass / clip
        elif name == 'hybrid':
            if not args.tracker:
                print('  [skip] hybrid needs --tracker'); continue
            def build_hyb():
                tr = MPDQNAgent(state_dim=args.state_dim, num_actions=NUM_ACTIONS,
                                hidden_layers=[256, 128, 64], device=args.device)
                tr.load(args.tracker)
                return HybridPolicy(tr)
            runs = [run_policy(build_hyb, c) for c in range(n_clips)]
        elif name == 'smarthybrid':
            if not args.tracker:
                print('  [skip] smarthybrid needs --tracker'); continue
            def build_shyb():
                tr = MPDQNAgent(state_dim=args.state_dim, num_actions=NUM_ACTIONS,
                                hidden_layers=[256, 128, 64], device=args.device)
                tr.load(args.tracker)
                return SmartHybridPolicy(tr)
            runs = [run_policy(build_shyb, c) for c in range(n_clips)]
        elif name == 'random':
            seeds = [int(s) for s in args.random_seeds.split(',') if s.strip()]
            runs = []
            for c in range(n_clips):
                for s in seeds:
                    runs.append(run_policy(lambda s=s: RandomPolicy(np.random.default_rng(1000 + s)), c))
        else:
            ctor = {'sweep': SweepPolicy, 'coordsweep': CoordinatedSweepPolicy,
                    'ovsweep': OverlapOptimizedSweepPolicy,
                    'greedy': GreedyPolicy, 'heuristic': HeuristicPolicy}[name]
            runs = [run_policy(ctor, c) for c in range(n_clips)]
        result['policies'][name] = runs
        for H in HORIZONS:
            d = [r[str(H)]['disc_fixed'] for r in runs]
            print(f'  H={H:3d}  disc_fixed={np.nanmean(d):.3f}+/-{np.nanstd(d):.3f} '
                  f'(n={len(d)})', flush=True)

    env.close()
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
