"""Non-learned PTZ baselines for the dual fisheye env, evaluated head-to-head
with the trained MP-DQN policy under *identical* conditions.

Every policy is stepped through the same `DualFisheyePTZEnvironment` with the
same detector, clip, step budget and reward function, so the per-episode reward
is an apples-to-apples comparison on the env's own objective (coverage +
motion-aware tracking + zoom regulation).

Policies
--------
- random    : uniform random action + param for both cameras (the requested baseline)
- sweep     : constant azimuth sweep (round-robin), periodic tilt nudge
- greedy    : reactive controller — pan toward the in-view target, zoom to frame,
              sweep when nothing is detected
- rl        : the trained tracker/explorer MP-DQN agents (if --tracker/--explorer given)

Usage
-----
    python baselines/evaluate.py \
        --data_path data/test/1779795064904.mp4 \
        --detector_weights yolo26n_obb_topview_person_270526.pt \
        --steps 720 --episodes 3 --state_dim 29 \
        --policies random,sweep,greedy,rl \
        --tracker results/<run>/tracker_best.pt \
        --explorer results/<run>/explorer_best.pt
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

# Action ids (mirrors dual_fisheye_env._apply_action)
STAY, PAN_L, PAN_R, ZOOM_IN, ZOOM_OUT, TILT_UP, TILT_DOWN = range(7)


class RandomPolicy:
    """Uniform random action + continuous param for both cameras."""
    name = 'random'
    stochastic = True

    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        pass

    def act(self, states, env):
        K = len(states)
        a = tuple(int(self.rng.integers(0, env.num_actions)) for _ in range(K))
        p = tuple(float(self.rng.random()) for _ in range(K))
        return a, p


class SweepPolicy:
    """Round-robin sweep: pan continuously, nudge tilt every `period` steps.

    The two cameras start 180 deg apart (env default), so a constant sweep keeps
    them on opposite sides of the scene — a sensible non-learned coverage policy.
    """
    name = 'sweep'
    stochastic = False

    def __init__(self, period=40):
        self.period = period
        self.t = 0

    def reset(self):
        self.t = 0

    def act(self, states, env):
        self.t += 1
        if self.t % self.period == 0:
            up = (self.t // self.period) % 2 == 0
            a = (TILT_UP, TILT_UP) if up else (TILT_DOWN, TILT_DOWN)
            return a, (1.0, 1.0)
        return (PAN_R, PAN_R), (1.0, 1.0)


class CoordinatedSweepPolicy:
    """Non-learned sweep with explicit division of labor to minimize redundancy:
    camera 0 rasters the LEFT azimuth half at a LOW tilt band (room centre), camera 1
    the RIGHT azimuth half at a HIGH tilt band (periphery). The two frustums are kept in
    different azimuth halves and different polar bands, so together they still cover the
    scene but avoid pointing at the same region. This is the strong baseline that tests
    whether RL's only advantage (low inter-camera overlap) survives against a *non-learned*
    policy that also avoids redundancy.
    """
    name = 'coordsweep'
    stochastic = False

    def __init__(self):
        self.t = 0
        self.dir = [1, -1]                 # azimuth sweep direction per camera
        self.lo = [-178.0, 2.0]            # cam0 left half, cam1 right half
        self.hi = [-2.0, 178.0]
        self.tilt_lo = [8.0, 42.0]         # cam0 low (centre) band, cam1 high (periphery)
        self.tilt_hi = [40.0, 74.0]
        self.tdir = [1, 1]

    def reset(self):
        self.t = 0
        self.dir = [1, -1]
        self.tdir = [1, 1]

    def act(self, states, env):
        self.t += 1
        acts = [STAY, STAY]
        for i in range(2):
            pan = float(env.pan[i])
            tilt = float(env.tilt[i])
            if self.t % 3 == 0:            # 1/3 of steps: scan tilt within the band
                if tilt >= self.tilt_hi[i]:
                    self.tdir[i] = -1
                elif tilt <= self.tilt_lo[i]:
                    self.tdir[i] = 1
                acts[i] = TILT_UP if self.tdir[i] > 0 else TILT_DOWN
            else:                          # otherwise: sweep azimuth within the half
                if pan >= self.hi[i]:
                    self.dir[i] = -1
                elif pan <= self.lo[i]:
                    self.dir[i] = 1
                acts[i] = PAN_R if self.dir[i] > 0 else PAN_L
        return tuple(acts), (1.0, 1.0)


class OverlapOptimizedSweepPolicy:
    """Systematic sweep that *directly minimizes* inter-camera redundancy by construction
    (review #3.4 / Q7). The two virtual cameras are held at antipodal azimuth (180 deg
    apart) and an equal, high polar angle `target_tilt`; their optical-axis separation is
    then 2*target_tilt. With FOV half-angle 45 deg (base_fov=90, zoom=1), any separation
    above 90 deg makes the two FOV cones DISJOINT, so the solid-angle overlap is ~0 by
    geometry, not by learning. Setting target_tilt>=48 deg (separation>=96 deg) guarantees
    this while a wide 90-deg cone at that axis still spans polar ~3..90 deg (nadir to
    horizon). The antipodal pair co-rotates in azimuth (both PAN_R, equal speed) to raster
    the whole hemisphere. This is the strong non-learned baseline that tests whether RL's
    one metric-robust advantage (lowest overlap) survives a controller built to beat it.

    Trade-off it exposes: forcing high tilt for separation sacrifices dedicated low-polar
    (directly-underneath) coverage, so discovery is the quantity to watch.
    """
    name = 'ovsweep'
    stochastic = False

    def __init__(self, target_tilt=48.0):
        self.target_tilt = float(target_tilt)
        self.t = 0

    def reset(self):
        self.t = 0

    def act(self, states, env):
        self.t += 1
        K = len(states)
        acts = [STAY] * K
        params = [1.0] * K
        for i in range(K):
            tilt = float(env.tilt[i])
            if abs(tilt - self.target_tilt) > 3.0:          # drive both axes into the high band
                acts[i] = TILT_UP if tilt < self.target_tilt else TILT_DOWN
                params[i] = float(min(1.0, abs(tilt - self.target_tilt) / max(env.tilt_speed, 1e-6)))
            else:                                           # then co-rotate azimuth to raster
                acts[i] = PAN_R                             # both same dir+speed -> stay 180 deg apart
                params[i] = 0.4                             # ~10 deg/step for a finer raster
        return tuple(acts), tuple(params)


class GreedyPolicy:
    """Reactive greedy-to-detection controller, per camera, from state features.

    State layout (see dual_fisheye_env._cam_state):
      s[0]=num_dets/5  s[5]=rel_x  s[6]=rel_y  s[7]=rel_area  s[8]=best_conf
    Pan toward the target until centered, then zoom to frame it; sweep when the
    view is empty.
    """
    name = 'greedy'
    stochastic = False

    def reset(self):
        pass

    def _act_one(self, s):
        num_dets, rel_x, rel_area, conf = s[0], s[5], s[7], s[8]
        if num_dets <= 0 or conf <= 0:
            return PAN_R, 1.0                       # nothing seen -> sweep
        if abs(rel_x) > 0.20:                        # off-center -> pan toward it
            return (PAN_R if rel_x > 0 else PAN_L), float(min(1.0, abs(rel_x)))
        if rel_area < 0.10:                          # centered but small -> zoom in
            return ZOOM_IN, 0.6
        if rel_area > 0.35:                          # too big -> zoom out
            return ZOOM_OUT, 0.6
        return STAY, 1.0                             # well framed

    def act(self, states, env):
        out = [self._act_one(s) for s in states]
        return tuple(a for a, _ in out), tuple(p for _, p in out)


class HeuristicPolicy:
    """Rule-based tracker+explorer that uses the SAME shared coverage map as the
    RL agent — the strong non-learned baseline the ablation needs. The tracker
    greedily follows the in-view target; the explorer steers toward the stalest
    azimuth sector of the coverage map while staying clear of the tracker, and
    alternates a tilt scan once aligned. If either camera already sees a person it
    frames them. This isolates "does learning beat a sensible hand-coded policy
    with the same information?".
    """
    name = 'heuristic'
    stochastic = False

    def __init__(self, sep_keep=40.0):
        self.sep_keep = float(sep_keep)
        self.t = 0

    def reset(self):
        self.t = 0

    def _track_one(self, s):
        num_dets, rel_x, rel_area, conf = s[0], s[5], s[7], s[8]
        if num_dets <= 0 or conf <= 0:
            return PAN_R, 1.0                       # nothing seen -> sweep
        if abs(rel_x) > 0.20:
            return (PAN_R if rel_x > 0 else PAN_L), float(min(1.0, abs(rel_x)))
        if rel_area < 0.10:
            return ZOOM_IN, 0.6
        if rel_area > 0.35:
            return ZOOM_OUT, 0.6
        return STAY, 1.0

    def _explore(self, s_expl, env):
        if s_expl[0] > 0 and s_expl[8] > 0:         # already sees someone -> frame
            return self._track_one(s_expl)
        try:                                        # steer to stalest azimuth sector
            summ = env.scene.summary()
            az_stale = np.asarray(summ['azimuth_staleness'], dtype=float)
            centres = np.asarray(env.scene._az_centres, dtype=float)
        except Exception:
            return PAN_R, 1.0
        cur = float(env.pan[1]); trk = float(env.pan[0])
        for i, c in enumerate(centres):             # avoid sectors near the tracker
            if abs(((c - trk + 180.0) % 360.0) - 180.0) < self.sep_keep:
                az_stale[i] = -1.0
        target = float(centres[int(np.argmax(az_stale))])
        d = ((target - cur + 180.0) % 360.0) - 180.0
        speed = float(getattr(env._host, 'pan_speed', 25.0))
        if abs(d) > 0.5 * speed:
            return (PAN_R if d > 0 else PAN_L), float(min(1.0, abs(d) / max(speed, 1e-6)))
        return (TILT_UP if (self.t // 6) % 2 == 0 else TILT_DOWN), 0.7  # aligned -> scan polar

    def act(self, states, env):
        self.t += 1
        a0, p0 = self._track_one(states[0])
        a1, p1 = self._explore(states[1], env)
        return (a0, a1), (p0, p1)


class AbsoluteViewPolicy:
    """Centralized ABSOLUTE-view controller (scripts/ppo_absolute.py).

    One policy sees both cameras (58-D) and selects an ordered pair of poses
    from the 12x4 view bank, setting pan/tilt/zoom OUTRIGHT -- the same
    instantaneous re-pointing freedom FSAC has, with no incremental slew cap.
    Returns STAY so the environment applies no further motion.

    Deterministic at evaluation: argmax of each categorical head rather than a
    sample, matching RLPolicy's and PPOContinuousPolicy's greedy behaviour. The
    second head is masked so the two cameras cannot land on the same view.
    """
    name = 'absppo'
    stochastic = False

    def __init__(self, policy, bank, device='cuda'):
        self.pol, self.bank, self.device = policy, bank, device
        self.h = None

    def reset(self):
        self.h = None

    def act(self, states, env):
        import numpy as _np, torch as _t
        o = _t.as_tensor(_np.concatenate([_np.asarray(states[0], dtype=_np.float32),
                                          _np.asarray(states[1], dtype=_np.float32)]),
                         device=self.device)
        with _t.no_grad():
            z, self.h = self.pol.feat(o, self.h)
            v1 = int(_t.argmax(self.pol.h1(z)))
            oh = _t.zeros(self.pol.n, device=o.device); oh[v1] = 1.0
            lg = self.pol.h2(_t.cat([z, oh], -1)).clone()
            # honour the SAME action mask the policy was trained under. A
            # mask-trained policy evaluated without its mask could emit pairs it
            # never saw in training, which would not be the policy we trained.
            legal = getattr(self.pol, 'disjoint', None)
            if legal is not None:
                lg[~legal[v1]] = -1e9
            else:
                lg[v1] = -1e9
            v2 = int(_t.argmax(lg))
        for i, vi in enumerate((v1, v2)):
            pan, tilt, zoom = self.bank[vi]
            env.pan[i], env.tilt[i], env.zoom[i] = pan, tilt, zoom
        return (STAY, STAY), (0.0, 0.0)


class PPOContinuousPolicy:
    """Continuous-action PPO controller (scripts/ppo_continuous.py).

    Emits (d_pan, d_tilt, d_zoom) per camera, scaled by the same per-step speed
    limits the discrete vocabulary uses, applies them directly, then returns STAY
    so the environment's step() performs no further action. Deterministic at
    evaluation (distribution mean), matching RLPolicy's greedy behaviour.
    """
    name = 'ppo'
    stochastic = False

    def __init__(self, agents, speeds, tilt_range, zoom_range, device='cuda'):
        self.agents, self.sp = agents, speeds
        self.tr, self.zr, self.device = tilt_range, zoom_range, device

    def reset(self):
        pass

    def act(self, states, env):
        import numpy as _np, torch as _t
        for i in range(2):
            o = _t.as_tensor(_np.asarray(states[i], dtype=_np.float32), device=self.device)
            with _t.no_grad():
                a = _t.tanh(self.agents[i].pi(o)).cpu().numpy()      # mean action
            d = _np.clip(a, -1.0, 1.0) * self.sp
            env.pan[i]  = float(((env.pan[i] + d[0] + 180.0) % 360.0) - 180.0)
            env.tilt[i] = float(_np.clip(env.tilt[i] + d[1], self.tr[0], self.tr[1]))
            env.zoom[i] = float(_np.clip(env.zoom[i] + d[2], self.zr[0], self.zr[1]))
        return (0, 0), (0.0, 0.0)


class RLPolicy:
    """Wraps the trained MP-DQN tracker/explorer agents (greedy, no exploration)."""
    name = 'rl'
    stochastic = False

    def __init__(self, tracker, explorer):
        self.tracker = tracker
        self.explorer = explorer

    def reset(self):
        pass

    def act(self, states, env):
        a_t, p_t = self.tracker.choose_action(states[0], training=False)
        a_e, p_e = self.explorer.choose_action(states[1], training=False)
        return (int(a_t), int(a_e)), (float(p_t), float(p_e))


class SoloPolicy:
    """K=1: a single MP-DQN agent driving the one available crop.

    Trained by ``main_train_solo.py`` in the same environment with ``n_cams=1``;
    the observation layout is the shared 29-D vector with the two pair-geometry
    features held at zero (there is no partner).
    """
    name = 'solo'
    stochastic = False

    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        pass

    def act(self, states, env):
        a, p = self.agent.choose_action(states[0], training=False)
        return (int(a),), (float(p),)


class HybridPolicy:
    """RL tracker on camera 0 + a full systematic raster on camera 1. The learned
    tracker gives fast, low-latency detection of active/moving targets; the raster gives
    a guaranteed coverage floor so the system stops missing people the pure-RL explorer
    left uncovered. Aims to combine the coordinated sweep's completeness (discovery) with
    the learned policy's redundancy/latency behaviour---no retraining required.
    """
    name = 'hybrid'
    stochastic = False

    def __init__(self, tracker, tilt_lo=8.0, tilt_hi=74.0):
        self.tracker = tracker
        self.tilt_lo, self.tilt_hi = float(tilt_lo), float(tilt_hi)
        self.t = 0
        self.pdir = 1
        self.tdir = 1

    def reset(self):
        self.t = 0
        self.pdir = 1
        self.tdir = 1

    def act(self, states, env):
        a_t, p_t = self.tracker.choose_action(states[0], training=False)   # cam0: learned
        self.t += 1
        pan1, tilt1 = float(env.pan[1]), float(env.tilt[1])
        if self.t % 4 == 0:                       # 1/4 steps: scan the full tilt band
            if tilt1 >= self.tilt_hi:
                self.tdir = -1
            elif tilt1 <= self.tilt_lo:
                self.tdir = 1
            a1, p1 = (TILT_UP if self.tdir > 0 else TILT_DOWN), 0.7
        else:                                     # otherwise: sweep the full azimuth
            if pan1 >= 178.0:
                self.pdir = -1
            elif pan1 <= -178.0:
                self.pdir = 1
            a1, p1 = (PAN_R if self.pdir > 0 else PAN_L), 1.0
        return (int(a_t), a1), (float(p_t), p1)


class SmartHybridPolicy:
    """Smarter hybrid: camera 0 runs the RL tracker only while a target is in view, and
    otherwise falls back to sweeping the stalest coverage-map sector (staying clear of
    camera 1) so it \emph{also} contributes coverage when idle; camera 1 runs a full
    systematic raster. The goal is the coordinated sweep's completeness (both cameras cover
    when nobody is being tracked) plus learned prioritization of movers.
    """
    name = 'smarthybrid'
    stochastic = False

    def __init__(self, tracker, sep_keep=40.0, tilt_lo=8.0, tilt_hi=74.0):
        self.tracker = tracker
        self.sep_keep = float(sep_keep)
        self.tilt_lo, self.tilt_hi = float(tilt_lo), float(tilt_hi)
        self.t = 0
        self.pdir = 1
        self.tdir = 1

    def reset(self):
        self.t = 0
        self.pdir = 1
        self.tdir = 1

    def _sweep_stalest(self, env):
        try:
            az = np.asarray(env.scene.summary()['azimuth_staleness'], float)
            centres = np.asarray(env.scene._az_centres, float)
        except Exception:
            return PAN_R, 1.0
        cur, other = float(env.pan[0]), float(env.pan[1])
        for i, c in enumerate(centres):                 # avoid camera 1's current sector
            if abs(((c - other + 180.0) % 360.0) - 180.0) < self.sep_keep:
                az[i] = -1.0
        target = float(centres[int(np.argmax(az))])
        d = ((target - cur + 180.0) % 360.0) - 180.0
        speed = float(getattr(env._host, 'pan_speed', 25.0))
        if abs(d) > 0.5 * speed:
            return (PAN_R if d > 0 else PAN_L), float(min(1.0, abs(d) / max(speed, 1e-6)))
        return (TILT_UP if (self.t // 6) % 2 == 0 else TILT_DOWN), 0.7

    def act(self, states, env):
        self.t += 1
        s0 = states[0]
        if s0[0] > 0 and s0[8] > 0:                      # cam0 sees someone -> learned track
            a0_, p0_ = self.tracker.choose_action(s0, training=False)
            a0, p0 = int(a0_), float(p0_)
        else:                                            # idle -> contribute coverage
            a0, p0 = self._sweep_stalest(env)
        pan1, tilt1 = float(env.pan[1]), float(env.tilt[1])   # cam1 full raster
        if self.t % 4 == 0:
            if tilt1 >= self.tilt_hi:
                self.tdir = -1
            elif tilt1 <= self.tilt_lo:
                self.tdir = 1
            a1, p1 = (TILT_UP if self.tdir > 0 else TILT_DOWN), 0.7
        else:
            if pan1 >= 178.0:
                self.pdir = -1
            elif pan1 <= -178.0:
                self.pdir = 1
            a1, p1 = (PAN_R if self.pdir > 0 else PAN_L), 1.0
        return (a0, a1), (p0, p1)


def run_policy(env, policy, episodes, steps):
    """Return per-episode (tracker, explorer, total) reward lists."""
    rt_list, re_list = [], []
    for _ in range(episodes):
        policy.reset()
        states = env.reset()
        rt = re = 0.0
        for _ in range(steps):
            actions, params = policy.act(states, env)
            states, (r_t, r_e), done, _ = env.step(actions, params)
            rt += r_t
            re += r_e
            if done:
                break
        rt_list.append(rt)
        re_list.append(re)
    return rt_list, re_list


def parse_args():
    ap = argparse.ArgumentParser('Dual fisheye PTZ baselines vs RL')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--detector_weights', type=str, default=None)
    ap.add_argument('--steps', type=int, default=720)
    ap.add_argument('--episodes', type=int, default=3,
                    help='Episodes per policy (deterministic policies repeat identically).')
    ap.add_argument('--state_dim', type=int, default=29)
    ap.add_argument('--policies', type=str, default='random,sweep,greedy,rl')
    ap.add_argument('--tracker', type=str, default=None)
    ap.add_argument('--explorer', type=str, default=None)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default=None, help='Optional JSON output path.')
    return ap.parse_args()


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

    rng = np.random.default_rng(args.seed)
    requested = [p.strip() for p in args.policies.split(',') if p.strip()]
    factory = {
        'random': lambda: RandomPolicy(rng),
        'sweep': lambda: SweepPolicy(),
        'greedy': lambda: GreedyPolicy(),
    }

    policies = []
    for name in requested:
        if name in factory:
            policies.append(factory[name]())
        elif name == 'rl':
            if args.tracker and args.explorer:
                tr = MPDQNAgent(state_dim=args.state_dim, num_actions=NUM_ACTIONS,
                                hidden_layers=[256, 128, 64], device=args.device)
                ex = MPDQNAgent(state_dim=args.state_dim, num_actions=NUM_ACTIONS,
                                hidden_layers=[256, 128, 64], device=args.device)
                tr.load(args.tracker)
                ex.load(args.explorer)
                policies.append(RLPolicy(tr, ex))
            else:
                print('[skip] rl policy requested but --tracker/--explorer not given')
        else:
            print(f'[skip] unknown policy: {name}')

    results = {}
    for pol in policies:
        eps = args.episodes if pol.stochastic else 1   # deterministic -> 1 episode
        rt, re = run_policy(env, pol, eps, args.steps)
        tot = [a + b for a, b in zip(rt, re)]
        results[pol.name] = {
            'episodes': eps,
            'tracker_mean': float(np.mean(rt)), 'tracker_std': float(np.std(rt)),
            'explorer_mean': float(np.mean(re)), 'explorer_std': float(np.std(re)),
            'total_mean': float(np.mean(tot)), 'total_std': float(np.std(tot)),
        }
        print(f'[done] {pol.name:8s} eps={eps} total={np.mean(tot):+.2f} +/- {np.std(tot):.2f}')

    env.close()

    # ---- comparison table ----
    print('\n' + '=' * 72)
    print(f'Baseline comparison | clip={os.path.basename(args.data_path)} '
          f'steps={args.steps} detector={os.path.basename(args.detector_weights or "default")}')
    print('=' * 72)
    print(f'{"policy":10s} {"eps":>4s} {"tracker":>12s} {"explorer":>12s} {"TOTAL":>14s}')
    print('-' * 72)
    order = sorted(results, key=lambda k: results[k]['total_mean'])
    for k in order:
        r = results[k]
        star = '  <-- RL' if k == 'rl' else ''
        print(f'{k:10s} {r["episodes"]:>4d} '
              f'{r["tracker_mean"]:>+12.1f} {r["explorer_mean"]:>+12.1f} '
              f'{r["total_mean"]:>+10.1f}+-{r["total_std"]:<.0f}{star}')
    print('=' * 72)

    if 'rl' in results:
        rl_tot = results['rl']['total_mean']
        for k in results:
            if k == 'rl':
                continue
            base = results[k]['total_mean']
            if abs(base) > 1e-6:
                print(f'RL vs {k}: {(rl_tot - base) / abs(base) * 100:+.0f}% '
                      f'({rl_tot:+.0f} vs {base:+.0f})')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'config': vars(args), 'results': results}, f, indent=2)
        print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
