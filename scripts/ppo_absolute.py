"""Centralized ABSOLUTE-view PPO: the learned counterpart of FSAC.

Review #18's sharpest objection: a virtual PTZ has no motor, yet our learned
policies are capped at 25 deg/step of pan while FSAC re-points instantly and
scores 0.98 discovery. So the comparison is budget-matched but NOT
action-space-matched, and the negative result could be an artefact of that.

This removes three confounds at once against the dual MP-DQN:
  * CENTRALIZED  - one policy sees both cameras and emits both poses, so there is
                   no tracker/explorer role split, no inter-agent credit
                   assignment and no tracker-first update order.
  * ABSOLUTE     - it selects poses directly from a view bank, exactly the freedom
                   FSAC has. No incremental slew limit.
  * GLOBAL REWARD- one scalar (sum over both cameras), not two local rewards.

Action space: a bank of |AZ| x |TILT| views at zoom 1, chosen as an ordered pair
with two categorical heads, pi(v1|s) then pi(v2|s,v1), the second masked so the
two cameras cannot occupy the same view. That is |AZ|*|TILT| outputs per head
rather than the C(48,2)=1128 of a joint head.

--recurrent adds a GRU over the observation, which is the other open limitation
(the 29-D observation is not a sufficient statistic and both current families are
feed-forward).

FAIRNESS INVARIANT, checked by --selftest and asserted at startup: FSAC's own
schedule must be REACHABLE inside this action space. If it were not, a loss here
would say nothing. FSAC uses pan 0/90/180/270 at tilt 45, zoom 1.
"""
import sys, os, json, argparse
sys.path.insert(0, '/app')
import numpy as np, torch, torch.nn as nn
from torch.distributions import Categorical

AZ    = [-180.0 + 30.0*i for i in range(12)]      # 12 azimuths, 30 deg apart
TILTS = [30.0, 45.0, 60.0, 75.0]
BANK  = [(a, t, 1.0) for t in TILTS for a in AZ]  # 48 candidate views


def disjoint_matrix(bank, base_fov=90.0, out_w=960, out_h=540, n_theta=90, n_phi=180):
    """(i,j) -> True iff views i and j share no direction on the sphere.

    Uses the paper's renderer-consistent frustum test (Sec. 4.5), not a cone, so
    "disjoint" means what Table 1's Ovlp_Omega column means. Masking the second
    head to this set injects GeoSweep's defining property as a hard structural
    prior instead of hoping the reward discovers it: 3/5 unmasked seeds intersect
    at essentially every step.

    Each view's membership mask is computed ONCE on a fixed sphere grid and the
    pairwise test is a bitwise AND -- 48 frustum tests rather than 48*48
    solid-angle integrations, which took minutes.
    """
    import numpy as _np, os as _os
    for _d in ('/app', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))):
        if _d not in sys.path:
            sys.path.insert(0, _d)
    from evaluation.metrics import _in_frustum, bearing_to_vec
    th = _np.linspace(0.5, 179.5, n_theta)
    ph = _np.linspace(-179.5, 179.5, n_phi)
    P, T = _np.meshgrid(ph, th)
    V = _np.stack([bearing_to_vec(float(p_), float(t_))
                   for t_, p_ in zip(T.ravel(), P.ravel())])
    masks = [_in_frustum(V, pan, tilt, zoom, base_fov, out_w, out_h)
             for (pan, tilt, zoom) in bank]
    n = len(bank)
    M = _np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i, j] = not bool((masks[i] & masks[j]).any())
    return M


def fsac_reachable(bank):
    """FSAC pins pan 0/90/180/270 at tilt 45. Every one of those poses must exist
    in the bank, or the learned selector is handicapped relative to the baseline
    it is compared against and the experiment is void."""
    need = [(0.0, 45.0), (90.0, 45.0), (180.0, 45.0), (270.0, 45.0)]
    have = {(round(((a + 180.0) % 360.0) - 180.0, 3), round(t, 3)) for a, t, _ in bank}
    miss = [p for p in need
            if (round(((p[0] + 180.0) % 360.0) - 180.0, 3), round(p[1], 3)) not in have]
    return (not miss), miss


class Policy(nn.Module):
    """Centralized actor-critic over an ordered pair of bank indices."""
    def __init__(self, obs_dim, n_views, hidden=256, recurrent=False, disjoint=None):
        super().__init__()
        self.recurrent, self.n = recurrent, n_views
        # disjoint[i] = boolean mask of legal partners for view i (None = no constraint)
        self.register_buffer('disjoint',
                             torch.as_tensor(disjoint) if disjoint is not None
                             else torch.ones(n_views, n_views, dtype=torch.bool),
                             persistent=False)   # keep out of state_dict: old ckpts must load
        self.enc = nn.Sequential(nn.Linear(obs_dim,hidden), nn.Tanh(),
                                 nn.Linear(hidden,hidden), nn.Tanh())
        self.gru = nn.GRUCell(hidden, hidden) if recurrent else None
        self.h1  = nn.Linear(hidden, n_views)                 # pi(v1 | s)
        self.h2  = nn.Linear(hidden + n_views, n_views)       # pi(v2 | s, v1)
        self.v   = nn.Linear(hidden, 1)

    def feat(self, o, h=None):
        z = self.enc(o)
        if self.recurrent:
            h = self.gru(z, h); return h, h
        return z, None

    def act(self, o, h=None):
        z, h2 = self.feat(o, h)
        d1 = Categorical(logits=self.h1(z)); v1 = d1.sample()
        oh = torch.zeros(self.n, device=o.device); oh[v1] = 1.0
        lg = self.h2(torch.cat([z, oh], -1)).clone()
        lg[~self.disjoint[v1]] = -1e9                  # illegal partners (incl. self)
        d2 = Categorical(logits=lg); v2 = d2.sample()
        return (v1, v2), d1.log_prob(v1) + d2.log_prob(v2), self.v(z).squeeze(-1), h2

    def evaluate(self, o, a):
        z, _ = self.feat(o)
        d1 = Categorical(logits=self.h1(z))
        oh = torch.zeros(len(o), self.n, device=o.device)
        oh.scatter_(1, a[:,0:1], 1.0)
        lg = self.h2(torch.cat([z, oh], -1)).clone()
        lg = lg.masked_fill(~self.disjoint[a[:,0]], -1e9)
        d2 = Categorical(logits=lg)
        lp  = d1.log_prob(a[:,0]) + d2.log_prob(a[:,1])
        ent = d1.entropy() + d2.entropy()
        return lp, ent, self.v(z).squeeze(-1)


def gae(rew, val, last_v, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rew, dtype=np.float32); run = 0.0
    for t in reversed(range(len(rew))):
        nxt = last_v if t == len(rew)-1 else val[t+1]
        run = (rew[t] + gamma*nxt - val[t]) + gamma*lam*run
        adv[t] = run
    return adv, adv + val


def update(pol, opt, obs, act, logp_old, adv, ret, clip=0.2, epochs=10, mb=64):
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(epochs):
        for i in torch.randperm(len(obs)).split(mb):
            lp, ent, v = pol.evaluate(obs[i], act[i])
            ratio = (lp - logp_old[i]).exp()
            l_pi = -torch.min(ratio*adv[i], ratio.clamp(1-clip,1+clip)*adv[i]).mean()
            loss = l_pi + 0.5*((v-ret[i])**2).mean() - 0.01*ent.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--max_steps', type=int, default=128)
    ap.add_argument('--data_path', default='./data/train/')
    ap.add_argument('--detector_weights')
    ap.add_argument('--out')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--recurrent', action='store_true')
    ap.add_argument('--disjoint', action='store_true',
                    help='mask the second head to views sharing zero solid angle with the first')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()

    DIS = disjoint_matrix(BANK) if a.disjoint else None
    if a.disjoint:
        import numpy as _np
        legal = DIS.sum(1)
        print('disjoint mask: %d/%d legal ordered pairs; min legal partners per view %d'
              % (int(DIS.sum()), len(BANK)**2, int(legal.min())))
        assert legal.min() > 0, 'some view has no disjoint partner -- policy would have no action'
        # FSAC's own pairs must survive the mask, or the comparison is void
        idx = {(round(((p0+180.0)%360.0)-180.0,3), round(t,3)): k
               for k,(p0,t,_z) in enumerate(BANK)}
        for p1,p2 in ((0.0,180.0),(90.0,270.0)):
            i = idx[(round(((p1+180.0)%360.0)-180.0,3),45.0)]
            j = idx[(round(((p2+180.0)%360.0)-180.0,3),45.0)]
            assert DIS[i,j], 'FSAC pair (%g,%g)@45 blocked by mask'%(p1,p2)
        print('FSAC antipodal pairs survive the mask: True')
    ok, miss = fsac_reachable(BANK)
    print('view bank: %d views (%d az x %d tilt, zoom 1)' % (len(BANK), len(AZ), len(TILTS)))
    print('FSAC schedule reachable in bank: %s%s' % (ok, '' if ok else '  MISSING %s'%miss))
    assert ok, 'FSAC poses not in bank -- comparison would be unfair, refusing to train'
    if a.selftest:
        p = Policy(58, len(BANK), recurrent=a.recurrent, disjoint=DIS)
        o = torch.zeros(58); (v1,v2), lp, v, h = p.act(o)
        assert v1 != v2, 'duplicate-view mask failed'
        A = torch.tensor([[int(v1), int(v2)]])
        lp2, ent, val = p.evaluate(o.unsqueeze(0), A)
        assert torch.isfinite(lp2).all() and torch.isfinite(ent).all()
        print('act -> views %d,%d  distinct=%s  logp=%.3f' % (v1, v2, v1!=v2, float(lp)))
        print('evaluate finite: True | recurrent=%s' % a.recurrent)
        print('SELFTEST PASS'); return

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = a.device if torch.cuda.is_available() else 'cpu'
    from environments.dual_fisheye_env import DualFisheyePTZEnvironment
    env = DualFisheyePTZEnvironment({'data_path': a.data_path, 'max_steps': a.max_steps,
        'state_dim': 29, 'detector_weights': a.detector_weights, 'device': dev})

    pol = Policy(58, len(BANK), recurrent=a.recurrent, disjoint=DIS).to(dev)
    opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    os.makedirs(a.out, exist_ok=True)
    hist, best = [], -1e9
    for ep in range(a.episodes):
        s = env.reset(); h = None
        O,A,LP,R,V = [],[],[],[],[]
        for t in range(a.max_steps):
            o = torch.as_tensor(np.concatenate([np.asarray(s[0],np.float32),
                                                np.asarray(s[1],np.float32)]), device=dev)
            with torch.no_grad(): (v1,v2), lp, val, h = pol.act(o, h)
            for i, vi in enumerate((int(v1), int(v2))):     # ABSOLUTE: set pose outright
                pan, tilt, zoom = BANK[vi]
                env.pan[i], env.tilt[i], env.zoom[i] = pan, tilt, zoom
            s2, rew, done, _ = env.step((0,0), (0.0,0.0))   # STAY: pose already set
            O.append(o.cpu().numpy()); A.append([int(v1),int(v2)])
            LP.append(float(lp)); V.append(float(val)); R.append(float(sum(rew)))  # GLOBAL reward
            s = s2
            if done: break
        o = torch.as_tensor(np.concatenate([np.asarray(s[0],np.float32),
                                            np.asarray(s[1],np.float32)]), device=dev)
        with torch.no_grad():
            z,_ = pol.feat(o, h); last_v = float(pol.v(z).squeeze(-1))
        adv, ret = gae(np.array(R,np.float32), np.array(V,np.float32), last_v)
        update(pol, opt, torch.as_tensor(np.array(O),device=dev),
               torch.as_tensor(np.array(A),device=dev),
               torch.as_tensor(np.array(LP,np.float32),device=dev),
               torch.as_tensor(adv,device=dev), torch.as_tensor(ret,device=dev))
        ret_ep = float(sum(R)); hist.append(ret_ep)
        if ret_ep > best:
            best = ret_ep; torch.save(pol.state_dict(), f'{a.out}/policy_best.pt')
        if ep % 10 == 0 or ep == a.episodes-1:
            print(f'  ep {ep:3d}  return {ret_ep:8.1f}  best {best:8.1f}  '
                  f'last20 {np.mean(hist[-20:]):8.1f}', flush=True)
    json.dump({'returns':hist,'best':best,'bank':len(BANK),'recurrent':a.recurrent,
               'disjoint_mask':bool(a.disjoint)},
              open(f'{a.out}/train.json','w'))
    print(f'DONE seed={a.seed} best={best:.1f} last20={np.mean(hist[-20:]):.1f}')

if __name__ == '__main__':
    main()
