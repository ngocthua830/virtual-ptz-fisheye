"""Continuous-action PPO baseline for the dual virtual-PTZ controller.

Addresses the standing limitation "one parameterized-action family (MP-DQN);
recurrent or continuous-action controllers might differ".

Instead of MP-DQN's 7 discrete primitives x magnitude, each agent emits a
continuous (d_pan, d_tilt, d_zoom) in [-1,1]^3, scaled by the SAME per-step speed
limits the discrete vocabulary uses, so neither controller can move faster than the
other. Everything downstream -- rendering, detector, reward, coverage map, the
tracker's zoom hygiene -- is the unmodified environment, so the only difference is
the action parameterisation.

Budget is matched to MP-DQN exactly: 100 episodes x 128 steps.
"""
import sys, os, json, argparse, time
sys.path.insert(0, '/app')
import numpy as np, torch, torch.nn as nn
from torch.distributions import Normal
from environments.dual_fisheye_env import DualFisheyePTZEnvironment

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=29, act_dim=3, hidden=(256,128,64)):
        super().__init__()
        def mlp(out):
            layers, d = [], obs_dim
            for h in hidden: layers += [nn.Linear(d,h), nn.Tanh()]; d = h
            return nn.Sequential(*layers, nn.Linear(d,out))
        self.pi, self.v = mlp(act_dim), mlp(1)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
    def dist(self, o):
        return Normal(torch.tanh(self.pi(o)), self.log_std.exp())
    def act(self, o):
        d = self.dist(o); a = d.sample()
        return a, d.log_prob(a).sum(-1), self.v(o).squeeze(-1)
    def evaluate(self, o, a):
        d = self.dist(o)
        return d.log_prob(a).sum(-1), d.entropy().sum(-1), self.v(o).squeeze(-1)

def gae(rew, val, last_v, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rew, dtype=np.float32); run = 0.0
    for t in reversed(range(len(rew))):
        nxt = last_v if t == len(rew)-1 else val[t+1]
        delta = rew[t] + gamma*nxt - val[t]
        run = delta + gamma*lam*run
        adv[t] = run
    return adv, adv + val

def update(ac, opt, obs, act, logp_old, adv, ret, clip=0.2, epochs=10, mb=64):
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(obs)
    for _ in range(epochs):
        for i in torch.randperm(n).split(mb):
            lp, ent, v = ac.evaluate(obs[i], act[i])
            ratio = (lp - logp_old[i]).exp()
            l_pi = -torch.min(ratio*adv[i], ratio.clamp(1-clip,1+clip)*adv[i]).mean()
            loss = l_pi + 0.5*((v-ret[i])**2).mean() - 0.01*ent.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--max_steps', type=int, default=128)
    ap.add_argument('--data_path', default='./data/train/')
    ap.add_argument('--detector_weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = a.device if torch.cuda.is_available() else 'cpu'

    env = DualFisheyePTZEnvironment({'data_path': a.data_path, 'max_steps': a.max_steps,
        'state_dim': 29, 'detector_weights': a.detector_weights, 'device': dev})
    # same per-step limits the discrete vocabulary uses -> neither can move faster
    SP = np.array([env.pan_speed, env.tilt_speed, env.zoom_speed], dtype=np.float32)
    TR, ZR = env.tilt_range, env.zoom_range

    agents = [ActorCritic().to(dev) for _ in range(2)]
    opts = [torch.optim.Adam(m.parameters(), lr=3e-4) for m in agents]
    os.makedirs(a.out, exist_ok=True)
    hist, best = [], -1e9
    for ep in range(a.episodes):
        s = env.reset()
        buf = [{k: [] for k in ('o','a','lp','r','v')} for _ in range(2)]
        for t in range(a.max_steps):
            acts = []
            for i in range(2):
                o = torch.as_tensor(np.asarray(s[i], dtype=np.float32), device=dev)
                with torch.no_grad(): ac_, lp, v = agents[i].act(o)
                d = np.clip(ac_.cpu().numpy(), -1, 1) * SP
                env.pan[i]  = float(((env.pan[i] + d[0] + 180.0) % 360.0) - 180.0)
                env.tilt[i] = float(np.clip(env.tilt[i] + d[1], TR[0], TR[1]))
                env.zoom[i] = float(np.clip(env.zoom[i] + d[2], ZR[0], ZR[1]))
                buf[i]['o'].append(o.cpu().numpy()); buf[i]['a'].append(ac_.cpu().numpy())
                buf[i]['lp'].append(float(lp)); buf[i]['v'].append(float(v))
            s2, rew, done, _ = env.step((0,0), (0.0,0.0))   # STAY: deltas already applied
            for i in range(2): buf[i]['r'].append(float(rew[i]))
            s = s2
            if done: break
        ret_ep = sum(sum(buf[i]['r']) for i in range(2))
        for i in range(2):
            o = torch.as_tensor(np.asarray(s[i], dtype=np.float32), device=dev)
            with torch.no_grad(): last_v = float(agents[i].v(o).squeeze(-1))
            r = np.array(buf[i]['r'], np.float32); v = np.array(buf[i]['v'], np.float32)
            adv, ret = gae(r, v, last_v)
            update(agents[i], opts[i],
                   torch.as_tensor(np.array(buf[i]['o']), device=dev),
                   torch.as_tensor(np.array(buf[i]['a']), device=dev),
                   torch.as_tensor(np.array(buf[i]['lp'], np.float32), device=dev),
                   torch.as_tensor(adv, device=dev), torch.as_tensor(ret, device=dev))
        hist.append(ret_ep)
        if ret_ep > best:
            best = ret_ep
            for i, nm in enumerate(('tracker','explorer')):
                torch.save(agents[i].state_dict(), f'{a.out}/{nm}_best.pt')
        if ep % 10 == 0 or ep == a.episodes-1:
            print(f'  ep {ep:3d}  return {ret_ep:8.1f}  best {best:8.1f}  '
                  f'last20 {np.mean(hist[-20:]):8.1f}', flush=True)
    json.dump({'returns': hist, 'best': best}, open(f'{a.out}/train.json','w'))
    print(f'DONE seed={a.seed} best={best:.1f} last20={np.mean(hist[-20:]):.1f}')

if __name__ == '__main__':
    main()
