"""Is the LOAF crossover real, or an artefact? Three specific threats:

T1 FRAGMENTATION. The human reference is tracked by us, not by the dataset. On
   0054 there are 381 "tracks" from ~31 people/frame. If one person fragments
   into several ids, a policy that STARES at a crowd discovers many fragments and
   is rewarded for it. Test: recompute discovery using only long-lived reference
   tracks, which are the least fragmented.
T2 DEGENERATE POLICY. RL might simply stop moving and point at the busiest
   region. Test: log what it actually does -- motion per step, zoom, coverage.
T3 GATE. With 31 people/frame a 15 deg gate may match a detection to the wrong
   person. Test: sweep the gate.
"""
import sys, math, json, collections, statistics as st
sys.path.insert(0,'/app')
import numpy as np
from evaluation.loaf import LoafSequence, load_annotations
from evaluation.metrics import discovery_rate, angular_sep
from evaluation.run_loaf import attach, StaticTiles
from environments.dual_fisheye_env import DualFisheyePTZEnvironment, NUM_ACTIONS
from models.agent.mpdqn_agent import MPDQNAgent
from baselines.evaluate import OverlapOptimizedSweepPolicy, RLPolicy

SEQ='0054'; STEPS=300
env=DualFisheyePTZEnvironment({'data_path':'data/test','max_steps':STEPS,'state_dim':29,
    'detector_weights':'yolo26n_obb_topview_person_290526.pt','loop_video':False})
host=env._host
ann=load_annotations('/loaf','val',sequences=[SEQ])
seq=LoafSequence('/loaf','val',SEQ,ann=ann); seq.read(0)
gt=seq.reference(max_frames=STEPS)

D=sys.argv[1]
def mk(name):
    if name=='ovsweep': return OverlapOptimizedSweepPolicy(48.0)
    tr=MPDQNAgent(state_dim=29,num_actions=NUM_ACTIONS,hidden_layers=[256,128,64],device='cuda')
    ex=MPDQNAgent(state_dim=29,num_actions=NUM_ACTIONS,hidden_layers=[256,128,64],device='cuda')
    tr.load(f'{D}/tracker_best.pt'); ex.load(f'{D}/explorer_best.pt')
    return RLPolicy(tr,ex)

runs={}
for name in ('ovsweep','rl'):
    pol=mk(name); pol.reset()
    states=env.reset(); attach(host,seq,host.frame_skip)
    agent,poses=[],[]
    for t in range(min(STEPS,len(gt))):
        dets=[]
        for i in range(2):
            for d in env.last_detections[i]:
                dets.append(host.world_bearing(d['bbox'],float(env.pan[i]),float(env.tilt[i]),
                            float(env.zoom[i]),host.ptz_out_w,host.ptz_out_h,host.base_fov))
        agent.append(dets); poses.append([(float(env.pan[i]),float(env.tilt[i]),float(env.zoom[i])) for i in (0,1)])
        a,p=pol.act(states,env); states,_,done,_=env.step(a,p)
        if done: break
    runs[name]=(agent,poses)

g=gt[:len(runs['rl'][0])]
life=collections.Counter()
for f in g:
    for p in f: life[p['id']]+=1

print("\n--- T1: discovery restricted to long-lived reference tracks ---")
print(f"{'min lifetime':>13} {'#tracks':>8} {'ovsweep':>9} {'rl':>9} {'RL-sweep':>10}")
for minlife in (1,3,5,10,20):
    keep={i for i,c in life.items() if c>=minlife}
    gf=[[p for p in f if p['id'] in keep] for f in g]
    if not keep: continue
    d_s=discovery_rate(gf,runs['ovsweep'][0],15.0); d_r=discovery_rate(gf,runs['rl'][0],15.0)
    print(f"{minlife:>13} {len(keep):>8} {d_s:>9.3f} {d_r:>9.3f} {d_r-d_s:>+10.3f}")

print("\n--- T2: what does each policy actually DO? ---")
for name,(agent,poses) in runs.items():
    pans=np.array([[p[0] for p in s] for s in poses]); tilts=np.array([[p[1] for p in s] for s in poses])
    zooms=np.array([[p[2] for p in s] for s in poses])
    dpan=np.abs(np.diff(pans,axis=0)); dpan=np.minimum(dpan,360-dpan)
    az_bins=set()
    for s in poses:
        for (pa,ti,zo) in s: az_bins.add(int(((pa+180)%360)//10))
    print(f"  {name:<8} |dpan|/step mean {dpan.mean():5.2f} deg, frac steps still {100*(dpan<0.5).mean():4.0f}% | "
          f"tilt {tilts.mean():5.1f}+-{tilts.std():4.1f} | zoom {zooms.mean():4.2f}+-{zooms.std():4.2f} | "
          f"azimuth bins visited {len(az_bins)}/36")

print("\n--- T3: gate sensitivity ---")
print(f"{'gate':>6} {'ovsweep':>9} {'rl':>9} {'RL-sweep':>10}")
for gate in (5.0,10.0,15.0,20.0):
    d_s=discovery_rate(g,runs['ovsweep'][0],gate); d_r=discovery_rate(g,runs['rl'][0],gate)
    print(f"{gate:>6.0f} {d_s:>9.3f} {d_r:>9.3f} {d_r-d_s:>+10.3f}")
