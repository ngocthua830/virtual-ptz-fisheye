"""Single-agent (shared-policy) ablation of the dual tracker+explorer design.

ONE MP-DQN controls BOTH virtual cameras: at each step it selects actions for cam-0 and
cam-1 from their respective per-camera states, and BOTH transitions are pooled into its
single replay buffer. This is the fair "single-agent" comparison a reviewer asked for --
it keeps the two-camera setup (so discovery/coverage/overlap remain comparable to the
dual system) but removes the tracker/explorer division of labor. The saved checkpoint is
evaluated with the existing harness as RLPolicy(single, single) (same weights on both
cams): pass --tracker single_best.pt --explorer single_best.pt to evaluation/metrics.py.

Usage mirrors main_train_dual.py (same reward-shaping knobs so the budget/reward match):
  python main_train_single.py --episodes 100 --max_steps 80 --seed 42 --tag single_s42 ...
"""

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser('Single (shared-policy) fisheye PTZ training')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--max_steps', type=int, default=80)
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
    p.add_argument('--detector_weights', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    # keep the same reward knobs as the dual full model so budget/reward are matched
    p.add_argument('--coverage_weight', type=float, default=0.5)
    p.add_argument('--novelty_weight', type=float, default=2.0)
    p.add_argument('--overlap_thresh_deg', type=float, default=40.0)
    p.add_argument('--peripheral_weight', type=float, default=3.0)
    p.add_argument('--tag', type=str, default='')
    return p.parse_args()


def make_agent(env, args):
    return MPDQNAgent(
        state_dim=env.state_dim,
        num_actions=7,
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
    out_dir = os.path.join(args.results_dir, f'fisheye_single_{timestamp}{suffix}')
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
    agent = make_agent(env, args)  # ONE shared policy for both cameras

    rewards = []
    best_total = -float('inf')

    print('=' * 60)
    print(f'Single (shared-policy) PTZ training | episodes={args.episodes} '
          f'steps={args.max_steps}')
    print(f'seed={args.seed} | detector={os.path.basename(args.detector_weights or "default")}')
    print(f'Results dir: {out_dir}')
    print('=' * 60)

    try:
        for ep in range(args.episodes):
            states = env.reset()
            ep_r = 0.0

            for step in range(args.max_steps):
                # same shared policy selects for both cameras
                a0, p0 = agent.choose_action(states[0], training=True)
                a1, p1 = agent.choose_action(states[1], training=True)

                next_states, (r0, r1), done, info = env.step((a0, a1), (p0, p1))

                # pool both cameras' experience into the one replay buffer
                agent.store_transition(states[0], a0, p0, r0, next_states[0], done)
                agent.store_transition(states[1], a1, p1, r1, next_states[1], done)

                if len(agent.memory) >= args.batch_size:
                    agent.update()
                    agent.update()  # two transitions added per step -> two updates

                ep_r += r0 + r1
                states = next_states

                print(
                    f'  Step {step+1}/{args.max_steps} | '
                    f'C0:act={a0} r={r0:+.2f} dets={len(info["detections"][0])} | '
                    f'C1:act={a1} r={r1:+.2f} dets={len(info["detections"][1])} | '
                    f'seen={info["seen_ids"]}'
                )
                if done:
                    break

            rewards.append(ep_r)
            print(f'Episode {ep+1}/{args.episodes} | total={ep_r:+.2f} | eps={agent.epsilon:.3f}')

            if ep_r > best_total:
                best_total = ep_r
                agent.save(os.path.join(out_dir, 'single_best.pt'))
                print(f'  new best total {best_total:+.2f}')

            if (ep + 1) % args.save_freq == 0:
                agent.save(os.path.join(out_dir, f'single_ep{ep+1}.pt'))
    except KeyboardInterrupt:
        print('\nInterrupted — saving final model.')
    finally:
        agent.save(os.path.join(out_dir, 'single_final.pt'))
        np.savez(os.path.join(out_dir, 'training_stats.npz'), rewards=rewards)
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
