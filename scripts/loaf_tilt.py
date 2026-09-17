"""Re-tune the geometric baseline for LOAF, exactly as the paper tunes it for our
room. Comparing RL (which adapts tilt) against a sweep pinned at a tilt chosen
for a different scene is not a test of learning vs geometry."""
import sys, json
sys.path.insert(0,'/app')
from evaluation.loaf import LoafSequence, load_annotations
from evaluation.metrics import discovery_rate, observed_time_frac, time_to_detect
from evaluation.run_loaf import attach
from environments.dual_fisheye_env import DualFisheyePTZEnvironment
from baselines.evaluate import OverlapOptimizedSweepPolicy

SEQ=sys.argv[1]; STEPS=300
env=DualFisheyePTZEnvironment({'data_path':'data/test','max_steps':STEPS,'state_dim':29,
  'detector_weights':'yolo26n_obb_topview_person_290526.pt','loop_video':False})
host=env._host
ann=load_annotations('/loaf','val',sequences=[SEQ])
seq=LoafSequence('/loaf','val',SEQ,ann=ann); seq.read(0)
gt=seq.reference(max_frames=STEPS)
print(f"sequence {SEQ}")
print(f"{'target tilt':>12} {'discovery':>10} {'ttd':>7} {'obsfrac':>8}")
best=None
for tt in (45.0,55.0,65.0,75.0):
    pol=OverlapOptimizedSweepPolicy(tt); pol.reset()
    states=env.reset(); attach(host,seq,host.frame_skip)
    agent=[]
    for t in range(min(STEPS,len(gt))):
        dets=[]
        for i in range(2):
            for d in env.last_detections[i]:
                dets.append(host.world_bearing(d['bbox'],float(env.pan[i]),float(env.tilt[i]),
                            float(env.zoom[i]),host.ptz_out_w,host.ptz_out_h,host.base_fov))
        agent.append(dets)
        a,p=pol.act(states,env); states,_,done,_=env.step(a,p)
        if done: break
    g=gt[:len(agent)]
    d=discovery_rate(g,agent,15.0); o=observed_time_frac(g,agent,15.0); tt_=time_to_detect(g,agent,15.0)
    print(f"{tt:>12.0f} {d:>10.3f} {tt_:>7.2f} {o:>8.3f}", flush=True)
    if best is None or d>best[1]: best=(tt,d,o)
print(f"  best geometric: tilt {best[0]:.0f} -> discovery {best[1]:.3f}, obsfrac {best[2]:.3f}")
