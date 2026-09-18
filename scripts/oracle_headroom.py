"""Hindsight-oracle headroom, with a FIXED reference cohort.

Two passes, so every policy is scored against exactly the same people:
  pass 1  build the reference once over the whole clip -> people[t], frozen id set
  pass 2  replay each policy; at each step test which reference people its two
          views geometrically contain; the oracle grid-searches the best pair.

Isolates POINTING from perception: nothing here depends on the detector finding a
person inside a crop, only on whether the crop was aimed at them.

Reusing the fixed-cohort discipline of evaluation/horizon_curve.py -- an earlier
version accumulated the reference inside the policy loop, so a policy that ended a
step early was scored against a different id set.
"""
import sys, os, json
sys.path.insert(0, '/app')
import numpy as np
from evaluation.metrics import Oracle, _SeqReader, _in_frustum, bearing_to_vec
from environments.dual_fisheye_env import DualFisheyePTZEnvironment, NUM_ACTIONS
from baselines.evaluate import (OverlapOptimizedSweepPolicy, RLPolicy, RandomPolicy,
                                SweepPolicy, CoordinatedSweepPolicy)
from models.agent.mpdqn_agent import MPDQNAgent

BASE_FOV, OUT_W, OUT_H, STEPS = 90.0, 960, 540, 300
PAN_GRID  = list(range(-180, 180, 15))
TILT_GRID = [30, 40, 45, 50, 55, 60, 65, 70]

def wb_vec(az_w, polar_w):
    """world_bearing (az, polar; 0 = nadir) -> the space _in_frustum uses.
    Validated: a person at (az=pan, polar=tilt) is the centre of view (pan, tilt)."""
    return bearing_to_vec(az_w + 90.0, 180.0 - polar_w)

def covered(V, ids, pan, tilt, zoom=1.0):
    m = _in_frustum(V, pan, tilt, zoom, BASE_FOV, OUT_W, OUT_H)
    return {i for i, ok in zip(ids, m) if ok}

env = DualFisheyePTZEnvironment({'data_path': 'data/test', 'max_steps': STEPS, 'state_dim': 29,
    'detector_weights': 'yolo26n_obb_topview_person_290526.pt', 'loop_video': False})
host = env._host

def make(name):
    if name == 'rl':
        tr = MPDQNAgent(state_dim=29, num_actions=NUM_ACTIONS, hidden_layers=[256,128,64], device='cuda')
        ex = MPDQNAgent(state_dim=29, num_actions=NUM_ACTIONS, hidden_layers=[256,128,64], device='cuda')
        tr.load(os.environ['TRACKER']); ex.load(os.environ['EXPLORER']); return RLPolicy(tr, ex)
    if name == 'ovsweep':    return OverlapOptimizedSweepPolicy(48.0)
    if name == 'sweep':      return SweepPolicy()
    if name == 'coordsweep': return CoordinatedSweepPolicy()
    if name == 'random':     return RandomPolicy(np.random.default_rng(0))
    raise ValueError(name)

POLICIES = os.environ.get('POLICIES', 'ovsweep,rl,random').split(',')
totals = {p: 0 for p in POLICIES}; totals['oracle'] = 0
n_total = 0
per_clip = {}

for clip in host.video_paths:
    host.video_path = clip
    # ---- pass 1: reference over the FULL clip, independent of any policy ----
    rdr = _SeqReader(clip); orc = Oracle(host)
    people = []
    for t in range(STEPS):
        f = rdr.read(t * host.frame_skip)
        if f is None: break
        if host.fish_R is None: host._detect_fisheye_circle(f)
        people.append(orc.update(f, t))
    rdr.close()
    all_ids = {p['id'] for step in people for p in step}
    n = max(len(all_ids), 1); n_total += len(all_ids)
    name = clip.split('/')[-1]; per_clip[name] = {'n_ids': len(all_ids)}

    # ---- oracle on the frozen per-step reference ----
    seen = set()
    for step in people:
        if not step: continue
        ids = [p['id'] for p in step]
        V = np.array([wb_vec(*p['bearing']) for p in step], dtype=np.float64)
        cands = [(pa, ti, covered(V, ids, pa, ti)) for pa in PAN_GRID for ti in TILT_GRID]
        b1 = max(cands, key=lambda c: len(c[2]))
        b2 = max(cands, key=lambda c: len(c[2] - b1[2]))
        seen |= b1[2] | b2[2]
    per_clip[name]['oracle'] = len(seen)/n; totals['oracle'] += len(seen)

    # ---- pass 2: each policy against the SAME frozen reference ----
    for pname in POLICIES:
        pol = make(pname)
        if hasattr(pol, 'reset'): pol.reset()
        states = env.reset(); host.video_path = clip
        seen = set()
        for t in range(len(people)):
            step = people[t]
            if step:
                ids = [p['id'] for p in step]
                V = np.array([wb_vec(*p['bearing']) for p in step], dtype=np.float64)
                for i in range(2):
                    seen |= covered(V, ids, float(env.pan[i]), float(env.tilt[i]), float(env.zoom[i]))
            a, p = pol.act(states, env); states, _r, done, _ = env.step(a, p)
            if done: break
        per_clip[name][pname] = len(seen)/n; totals[pname] += len(seen)
    print(f"  {name}: ids={len(all_ids)}  " +
          "  ".join(f"{k}={per_clip[name][k]:.3f}" for k in ['oracle']+POLICIES), flush=True)

print("\nPOOLED (fixed cohort, n=%d ids)" % n_total)
pooled = {k: totals[k]/max(n_total,1) for k in totals}
for k in ['oracle']+POLICIES:
    print(f"  {k:10s} {pooled[k]:.3f}" + (f"   headroom vs oracle {pooled['oracle']-pooled[k]:+.3f}" if k!='oracle' else ""))
json.dump({'per_clip': per_clip, 'pooled': pooled, 'n_ids': n_total},
          open('/app/results/full128_bearing/oracle_ladder.json','w'), indent=1)
