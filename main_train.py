"""Train MP-DQN on the single-fisheye PTZ environment.

Train and eval both use the same fisheye clip (data/1767931881294.mp4).
"""

import argparse
import datetime
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environments.fisheye_env import FisheyePTZEnvironment
from models.agent.mpdqn_agent import MPDQNAgent


def parse_args():
    p = argparse.ArgumentParser('Fisheye PTZ MP-DQN training')
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--max_steps', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--actor_lr', type=float, default=1e-4)
    p.add_argument('--param_lr', type=float, default=1e-5)
    p.add_argument('--epsilon_start', type=float, default=1.0)
    p.add_argument('--epsilon_end', type=float, default=0.02)
    p.add_argument('--epsilon_decay', type=int, default=5000)
    p.add_argument('--replay_memory', type=int, default=100000)
    p.add_argument('--target_update_freq', type=int, default=100)
    p.add_argument('--save_freq', type=int, default=50)
    p.add_argument('--log_freq', type=int, default=1)
    p.add_argument('--data_path', type=str, default='./data/train/')
    p.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--results_dir', type=str, default='results')
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--load_model', type=str, default=None)
    p.add_argument('--start_episode', type=int, default=0)
    p.add_argument('--detector_weights', type=str, default=None,
                   help='Path to detector .pt (OBB models auto-detected).')
    return p.parse_args()


def make_env(args):
    return FisheyePTZEnvironment({
        'data_path': args.data_path,
        'max_steps': args.max_steps,
        'detector_weights': args.detector_weights,
    })


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
    out_dir = os.path.join(args.results_dir, f'fisheye_ptz_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)

    env = make_env(args)
    agent = make_agent(env, args)

    if args.load_model and os.path.exists(args.load_model):
        agent.load(args.load_model)

    episode_rewards, episode_lengths, losses = [], [], []
    best_reward = -float('inf')

    print('=' * 60)
    print(f'Fisheye PTZ training | episodes={args.episodes} steps={args.max_steps} '
          f'device={args.device}')
    print(f'Results dir: {out_dir}')
    print('=' * 60)

    try:
        for episode in range(args.start_episode, args.episodes):
            state = env.reset()
            ep_reward, ep_loss, update_count = 0.0, 0.0, 0

            for step in range(args.max_steps):
                action, action_param = agent.choose_action(state, training=True)
                next_state, reward, done, info = env.step(action, action_param=action_param)

                print(f'  Step {step+1}/{args.max_steps} | act={action} '
                      f'param={action_param:.2f} | r={reward:+.2f} | '
                      f'dets={info["num_detections"]} | '
                      f'pan={info["pan"]:+.1f} tilt={info["tilt"]:+.1f} z={info["zoom"]:.1f}')

                agent.store_transition(state, action, action_param, reward, next_state, done)
                if len(agent.memory) >= args.batch_size:
                    stats = agent.update()
                    ep_loss += stats.get('loss_actor', 0.0)
                    update_count += 1

                ep_reward += reward
                state = next_state
                if done:
                    break

            episode_rewards.append(ep_reward)
            episode_lengths.append(step + 1)
            if update_count:
                losses.append(ep_loss / update_count)

            if (episode + 1) % args.log_freq == 0:
                avg_r = float(np.mean(episode_rewards[-args.log_freq:]))
                print(f'Episode {episode+1}/{args.episodes} | reward={ep_reward:+.2f} | '
                      f'avg({args.log_freq})={avg_r:+.2f} | eps={agent.epsilon:.4f}')

            if ep_reward > best_reward:
                best_reward = ep_reward
                agent.save(os.path.join(out_dir, 'agent_best.pt'))
                print(f'  new best {best_reward:+.2f}')

            if (episode + 1) % args.save_freq == 0:
                agent.save(os.path.join(out_dir, f'agent_ep{episode+1}.pt'))
    except KeyboardInterrupt:
        print('\nInterrupted — saving final model.')
    finally:
        agent.save(os.path.join(out_dir, 'agent_final.pt'))
        np.savez(os.path.join(out_dir, 'training_stats.npz'),
                 episode_rewards=episode_rewards,
                 episode_lengths=episode_lengths,
                 losses=losses)
        print('=' * 60)
        print(f'Done. best_reward={best_reward:+.2f} | results={out_dir}')
        print('=' * 60)
        env.close()


def evaluate(args):
    env = make_env(args)
    agent = make_agent(env, args)

    model_path = args.load_model or os.path.join(args.results_dir, 'agent_best.pt')
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f'No model at {model_path}, using random policy.')

    rewards = []
    for ep in range(10):
        state = env.reset()
        ep_r = 0.0
        for step in range(args.max_steps):
            action, action_param = agent.choose_action(state, training=False)
            state, r, done, _ = env.step(action, action_param=action_param)
            ep_r += r
            if done:
                break
        rewards.append(ep_r)
        print(f'eval ep {ep+1}: reward={ep_r:+.2f}')
    print('=' * 60)
    print(f'Average eval reward: {np.mean(rewards):+.2f}')
    env.close()


if __name__ == '__main__':
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA unavailable, falling back to CPU.')
        args.device = 'cpu'
    (evaluate if args.eval_only else train)(args)
