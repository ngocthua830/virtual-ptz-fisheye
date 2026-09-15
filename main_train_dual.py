"""Joint training of two MP-DQN agents (tracker + explorer) on the dual fisheye env."""

import argparse
import datetime
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environments.dual_fisheye_env import DualFisheyePTZEnvironment, NUM_ACTIONS
from models.agent.mpdqn_agent import MPDQNAgent


def set_seed(seed):
    """Anchor every RNG so a run is reproducible from a single seed.

    Covers the agent's epsilon-greedy/replay sampling (python `random`),
    its action-param noise (`numpy`), network init/optim (`torch` + cuda),
    and the env's per-episode clip sampling (host env reads config['seed']).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser('Dual fisheye PTZ training (tracker + explorer)')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--max_steps', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--actor_lr', type=float, default=1e-4)
    p.add_argument('--param_lr', type=float, default=1e-5)
    p.add_argument('--epsilon_start', type=float, default=1.0)
    p.add_argument('--epsilon_end', type=float, default=0.02)
    p.add_argument('--epsilon_decay', type=int, default=5000)
    p.add_argument('--replay_memory', type=int, default=100000)
    p.add_argument('--target_update_freq', type=int, default=100)
    p.add_argument('--save_freq', type=int, default=50)
    p.add_argument('--data_path', type=str, default='./data/train/')
    p.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--results_dir', type=str, default='results')
    p.add_argument('--detector_weights', type=str, default=None,
                   help='Path to detector .pt (OBB models auto-detected).')
    p.add_argument('--seed', type=int, default=42,
                   help='Anchor seed for reproducible runs (RNGs + env clip sampling).')
    # ---- ablation knobs (defaults reproduce the full model) ----
    p.add_argument('--coverage_weight', type=float, default=0.5,
                   help='Shared coverage-map credit weight. 0 = no map coordination.')
    p.add_argument('--novelty_weight', type=float, default=2.0,
                   help='Explorer new-identity bonus. 0 = no novelty reward.')
    p.add_argument('--overlap_thresh_deg', type=float, default=40.0,
                   help='Anti-overlap penalty threshold (deg). 0 = no overlap penalty.')
    p.add_argument('--peripheral_weight', type=float, default=3.0,
                   help='Coverage peripheral weighting (rim). 1.0 = uniform weighting.')
    p.add_argument('--tag', type=str, default='',
                   help='Optional suffix appended to the run directory name.')
    return p.parse_args()


def make_agent(env, args):
    return MPDQNAgent(
        state_dim=env.state_dim,
        num_actions=NUM_ACTIONS,
        hidden_layers=[256, 128, 64],
        learning_rate=args.actor_lr,
        param_lr=args.param_lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        replay_memory_size=args.replay_memory,
        target_update_freq=args.target_update_freq,
        device=args.device,
    )


def train(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f'_{args.tag}' if args.tag else ''
    out_dir = os.path.join(args.results_dir, f'fisheye_dual_{timestamp}{suffix}')
    os.makedirs(out_dir, exist_ok=True)

    env = DualFisheyePTZEnvironment({
        'data_path': args.data_path,
        'max_steps': args.max_steps,
        'detector_weights': args.detector_weights,
        'seed': args.seed,
        'coverage_weight': args.coverage_weight,
        'novelty_weight': args.novelty_weight,
        'overlap_thresh_deg': args.overlap_thresh_deg,
        'peripheral_weight': args.peripheral_weight,
    })
    tracker = make_agent(env, args)
    explorer = make_agent(env, args)

    rewards_t, rewards_e = [], []
    best_total = -float('inf')

    print('=' * 60)
    print(f'Dual fisheye PTZ training | episodes={args.episodes} steps={args.max_steps}')
    print(f'seed={args.seed} | detector={os.path.basename(args.detector_weights or "default")}')
    print(f'Results dir: {out_dir}')
    print('=' * 60)

    try:
        for ep in range(args.episodes):
            states = env.reset()
            ep_r_t, ep_r_e = 0.0, 0.0

            for step in range(args.max_steps):
                a_t, p_t = tracker.choose_action(states[0], training=True)
                a_e, p_e = explorer.choose_action(states[1], training=True)

                next_states, (r_t, r_e), done, info = env.step((a_t, a_e), (p_t, p_e))

                tracker.store_transition(states[0], a_t, p_t, r_t, next_states[0], done)
                explorer.store_transition(states[1], a_e, p_e, r_e, next_states[1], done)

                if len(tracker.memory) >= args.batch_size:
                    tracker.update()
                    explorer.update()

                ep_r_t += r_t
                ep_r_e += r_e
                states = next_states

                print(
                    f'  Step {step+1}/{args.max_steps} | '
                    f'T:act={a_t} r={r_t:+.2f} dets={len(info["detections"][0])} '
                    f'pan={info["pan"][0]:+.1f} tilt={info["tilt"][0]:+.1f} z={info["zoom"][0]:.1f} | '
                    f'E:act={a_e} r={r_e:+.2f} dets={len(info["detections"][1])} '
                    f'pan={info["pan"][1]:+.1f} tilt={info["tilt"][1]:+.1f} z={info["zoom"][1]:.1f} | '
                    f'seen={info["seen_ids"]}'
                )
                if done:
                    break

            rewards_t.append(ep_r_t)
            rewards_e.append(ep_r_e)
            total = ep_r_t + ep_r_e

            print(
                f'Episode {ep+1}/{args.episodes} | tracker={ep_r_t:+.2f} '
                f'explorer={ep_r_e:+.2f} total={total:+.2f} | '
                f'epsT={tracker.epsilon:.3f} epsE={explorer.epsilon:.3f}'
            )

            if total > best_total:
                best_total = total
                tracker.save(os.path.join(out_dir, 'tracker_best.pt'))
                explorer.save(os.path.join(out_dir, 'explorer_best.pt'))
                print(f'  new best total {best_total:+.2f}')

            if (ep + 1) % args.save_freq == 0:
                tracker.save(os.path.join(out_dir, f'tracker_ep{ep+1}.pt'))
                explorer.save(os.path.join(out_dir, f'explorer_ep{ep+1}.pt'))
    except KeyboardInterrupt:
        print('\nInterrupted — saving final models.')
    finally:
        tracker.save(os.path.join(out_dir, 'tracker_final.pt'))
        explorer.save(os.path.join(out_dir, 'explorer_final.pt'))
        np.savez(os.path.join(out_dir, 'training_stats.npz'),
                 tracker_rewards=rewards_t, explorer_rewards=rewards_e)
        print('=' * 60)
        print(f'Done. best_total={best_total:+.2f} | results={out_dir}')
        print('=' * 60)
        env.close()


if __name__ == '__main__':
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    set_seed(args.seed)
    train(args)
