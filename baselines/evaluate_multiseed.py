"""Multi-seed, multi-episode evaluation of the trained MP-DQN policy vs the
non-learned baselines — the statistically-sound version of ``evaluate.py``.

Why this exists
---------------
``evaluate.py`` caps every *deterministic* policy (RL, sweep, greedy) at a single
episode, so on a fixed clip the RL headline number has ``std = 0``. That is the #1
rigor gap for a paper. Here we instead:

  * point ``--data_path`` at a *directory* of held-out clips, so each episode
    samples a clip via the env's seeded RNG (``fisheye_env.py`` reset);
  * run *every* policy for ``--episodes`` episodes under ``--seeds`` different
    seeds;
  * make the comparison **paired** — before each policy runs, the host clip-RNG is
    reset to the same seed, so policy A episode e and policy B episode e see the
    *same* clip;
  * treat each seed's mean-over-episodes as one sample (n = num seeds) and report
    mean +/- std across seeds, plus a paired t-test of RL vs each baseline.

Usage
-----
    python baselines/evaluate_multiseed.py \
        --data_path data/test \
        --detector_weights yolo26n_obb_topview_person_290526.pt \
        --steps 720 --episodes 3 --seeds 5 --state_dim 29 \
        --tracker results/fisheye_dual_20260529_151337/tracker_best.pt \
        --explorer results/fisheye_dual_20260529_151337/explorer_best.pt \
        --out baselines/results/multiseed_290526.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.dual_fisheye_env import DualFisheyePTZEnvironment, NUM_ACTIONS
from models.agent.mpdqn_agent import MPDQNAgent
from baselines.evaluate import (
    RandomPolicy, SweepPolicy, GreedyPolicy, RLPolicy, run_policy,
)

try:
    from scipy import stats as _scipy_stats
except Exception:
    _scipy_stats = None


def parse_args():
    ap = argparse.ArgumentParser('Multi-seed dual fisheye PTZ baselines vs RL')
    ap.add_argument('--data_path', type=str, required=True,
                    help='Directory of clips (each episode samples one) or a single file.')
    ap.add_argument('--detector_weights', type=str, default=None)
    ap.add_argument('--steps', type=int, default=720)
    ap.add_argument('--episodes', type=int, default=3, help='Episodes per seed.')
    ap.add_argument('--seeds', type=int, default=5, help='Number of seeds (0..seeds-1).')
    ap.add_argument('--seed_list', type=str, default=None,
                    help='Explicit comma-separated seeds (overrides --seeds). '
                         'Used to shard one seed per GPU across parallel containers.')
    ap.add_argument('--state_dim', type=int, default=29)
    ap.add_argument('--policies', type=str, default='random,sweep,greedy,rl')
    ap.add_argument('--tracker', type=str, default=None)
    ap.add_argument('--explorer', type=str, default=None)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--out', type=str, default=None)
    return ap.parse_args()


def make_policies(requested, args):
    """Factories are re-invoked per seed so stateful policies reset cleanly."""
    def build(name, seed):
        if name == 'random':
            return RandomPolicy(np.random.default_rng(1000 + seed))
        if name == 'sweep':
            return SweepPolicy()
        if name == 'greedy':
            return GreedyPolicy()
        if name == 'rl':
            if not (args.tracker and args.explorer):
                return None
            tr = MPDQNAgent(state_dim=args.state_dim, num_actions=NUM_ACTIONS,
                            hidden_layers=[256, 128, 64], device=args.device)
            ex = MPDQNAgent(state_dim=args.state_dim, num_actions=NUM_ACTIONS,
                            hidden_layers=[256, 128, 64], device=args.device)
            tr.load(args.tracker)
            ex.load(args.explorer)
            return RLPolicy(tr, ex)
        return None
    return build


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    env = DualFisheyePTZEnvironment({
        'data_path': args.data_path,
        'max_steps': args.steps,
        'state_dim': args.state_dim,
        'detector_weights': args.detector_weights,
    })

    if args.seed_list:
        seed_values = [int(s) for s in args.seed_list.split(',') if s.strip() != '']
    else:
        seed_values = list(range(args.seeds))

    n_clips = len(getattr(env._host, 'video_paths', []) or [])
    print(f'[cfg] clips={n_clips} steps={args.steps} '
          f'episodes/seed={args.episodes} seeds={seed_values} '
          f'detector={os.path.basename(args.detector_weights or "default")}')
    if n_clips <= 1:
        print('[warn] <=1 clip found: deterministic policies will be identical '
              'across episodes. Point --data_path at a directory of clips.')

    requested = [p.strip() for p in args.policies.split(',') if p.strip()]
    build = make_policies(requested, args)

    # per-policy: episode-level totals, and per-seed mean-over-episodes
    ep_totals = {p: [] for p in requested}          # flat list of every episode total
    seed_means = {p: [] for p in requested}         # one mean per seed (unit of analysis)
    seed_means_t = {p: [] for p in requested}       # tracker-only per-seed mean
    seed_means_e = {p: [] for p in requested}       # explorer-only per-seed mean

    for seed in seed_values:
        print(f'\n===== seed {seed} =====')
        for name in requested:
            pol = build(name, seed)
            if pol is None:
                if seed == 0:
                    print(f'[skip] {name}: not available (missing --tracker/--explorer?)')
                continue
            # Paired design: reset the host clip-RNG so every policy sees the same
            # clip sequence for this seed.
            env._host._rng = np.random.default_rng(seed)
            rt, re = run_policy(env, pol, args.episodes, args.steps)
            tot = [a + b for a, b in zip(rt, re)]
            ep_totals[name].extend(tot)
            seed_means[name].append(float(np.mean(tot)))
            seed_means_t[name].append(float(np.mean(rt)))
            seed_means_e[name].append(float(np.mean(re)))
            print(f'  {name:8s} eps={args.episodes} '
                  f'seed_mean_total={np.mean(tot):+.1f} (episodes: '
                  f'{", ".join(f"{x:+.0f}" for x in tot)})')

    env.close()

    # ---------- aggregate (unit of analysis = seed) ----------
    present = [p for p in requested if seed_means[p]]
    summary = {}
    for p in present:
        sm = np.array(seed_means[p])
        summary[p] = {
            'n_seeds': int(sm.size),
            'episodes_per_seed': args.episodes,
            'total_mean': float(sm.mean()),
            'total_std': float(sm.std(ddof=1)) if sm.size > 1 else 0.0,
            'total_sem': float(sm.std(ddof=1) / np.sqrt(sm.size)) if sm.size > 1 else 0.0,
            'tracker_mean': float(np.mean(seed_means_t[p])),
            'explorer_mean': float(np.mean(seed_means_e[p])),
            'per_seed_totals': [float(x) for x in sm],
            'all_episode_totals': [float(x) for x in ep_totals[p]],
        }

    print('\n' + '=' * 78)
    print(f'MULTI-SEED comparison | data={os.path.basename(args.data_path.rstrip("/"))} '
          f'clips={n_clips} steps={args.steps} seeds={args.seeds} x {args.episodes} eps')
    print('=' * 78)
    print(f'{"policy":10s} {"tracker":>10s} {"explorer":>10s} '
          f'{"TOTAL mean":>12s} {"std":>8s} {"sem":>8s}')
    print('-' * 78)
    for p in sorted(present, key=lambda k: summary[k]['total_mean']):
        s = summary[p]
        star = '  <-- RL' if p == 'rl' else ''
        print(f'{p:10s} {s["tracker_mean"]:>+10.1f} {s["explorer_mean"]:>+10.1f} '
              f'{s["total_mean"]:>+12.1f} {s["total_std"]:>8.1f} {s["total_sem"]:>8.1f}{star}')
    print('=' * 78)

    # ---------- paired RL vs baseline ----------
    if 'rl' in summary:
        rl_seed = np.array(seed_means['rl'])
        print('\nPaired RL vs baseline (per-seed, n={}):'.format(rl_seed.size))
        summary['_comparisons'] = {}
        for p in present:
            if p == 'rl':
                continue
            base_seed = np.array(seed_means[p])
            diff = rl_seed - base_seed
            pct = (summary['rl']['total_mean'] - summary[p]['total_mean']) / \
                  abs(summary[p]['total_mean']) * 100 if summary[p]['total_mean'] else float('nan')
            line = (f'  RL vs {p:8s}: {pct:+.0f}%  '
                    f'(RL {summary["rl"]["total_mean"]:+.0f} vs {summary[p]["total_mean"]:+.0f}), '
                    f'mean_diff={diff.mean():+.1f} +/- {diff.std(ddof=1) if diff.size>1 else 0.0:.1f}')
            comp = {'pct': float(pct), 'mean_diff': float(diff.mean()),
                    'diff_std': float(diff.std(ddof=1)) if diff.size > 1 else 0.0}
            if _scipy_stats is not None and diff.size > 1:
                t, pv = _scipy_stats.ttest_rel(rl_seed, base_seed)
                comp['t_stat'] = float(t)
                comp['p_value'] = float(pv)
                line += f', paired t={t:+.2f} p={pv:.4f}'
                if pv < 0.05:
                    line += ' *'
            summary['_comparisons'][p] = comp
            print(line)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'config': vars(args), 'clips': n_clips, 'summary': summary}, f, indent=2)
        print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
