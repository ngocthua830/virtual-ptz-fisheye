"""K=1 training: ONE MP-DQN agent, one rendered+detected crop per source frame.

The paper's headline setting is K=2 (tracker + explorer). This script trains the
same agent family under the tighter sensing budget K=1, to test whether the value
of learned control rises as the view budget shrinks.

Everything except the budget is held fixed: same 29-D observation layout (the two
pair-geometry features are zeroed by the env, see `n_cams`), same MP-DQN
hyper-parameters, same reward family -- the single agent receives BOTH the tracker
and explorer reward terms on its one view, since it must follow *and* discover.
The anti-overlap penalty is inactive at K=1 (no partner to be redundant with).
"""
import argparse, datetime, os, random, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environments.dual_fisheye_env import DualFisheyePTZEnvironment, NUM_ACTIONS
from models.agent.mpdqn_agent import MPDQNAgent
from main_train_dual import set_seed, make_agent, parse_args as _dual_args


def parse_args():
    # Reuse the dual parser verbatim so every hyper-parameter matches K=2.
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    return _dual_args()


def train(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f'_{args.tag}' if args.tag else ''
    out_dir = os.path.join(args.results_dir, f'fisheye_solo_{timestamp}{suffix}')
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
        'state_dim': args.state_dim,
        'full_map': args.full_map,
        'n_cams': 1,                      # <-- the only difference from K=2
    })
    agent = make_agent(env, args)
    rewards, best = [], -float('inf')

    print('=' * 60)
    print(f'SOLO (K=1) fisheye PTZ training | episodes={args.episodes} steps={args.max_steps}')
    print(f'seed={args.seed} | results={out_dir}')
    print('=' * 60)
    try:
        for ep in range(args.episodes):
            states = env.reset()
            ep_r = 0.0
            for step in range(args.max_steps):
                a, p = agent.choose_action(states[0], training=True)
                next_states, (r,), done, info = env.step((a,), (p,))
                agent.store_transition(states[0], a, p, r, next_states[0], done)
                if len(agent.memory) >= args.batch_size:
                    agent.update()
                ep_r += r
                states = next_states
                if done:
                    break
            rewards.append(ep_r)
            print(f'Episode {ep+1}/{args.episodes} | solo={ep_r:+.2f} | eps={agent.epsilon:.3f}')
            if ep_r > best:
                best = ep_r
                agent.save(os.path.join(out_dir, 'solo_best.pt'))
                print(f'  new best {best:+.2f}')
            if (ep + 1) % args.save_freq == 0:
                agent.save(os.path.join(out_dir, f'solo_ep{ep+1}.pt'))
    except KeyboardInterrupt:
        print('\nInterrupted -- saving final model.')
    finally:
        agent.save(os.path.join(out_dir, 'solo_final.pt'))
        np.savez(os.path.join(out_dir, 'training_stats.npz'), solo_rewards=rewards)
        print(f'Done. best={best:+.2f} | results={out_dir}')
        env.close()


if __name__ == '__main__':
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    set_seed(args.seed)
    train(args)
