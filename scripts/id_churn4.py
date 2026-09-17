"""Pick (gate, max_age) for world-bearing association.

Target: identity count close to the real people present and, above all, FLAT in
pan rate -- the failure of the old signal was that it scaled with self-motion.
"""
import sys, itertools
sys.path.insert(0, '/app')
from environments.dual_fisheye_env import DualFisheyePTZEnvironment as E
from evaluation.metrics import Oracle, _SeqReader
STAY, PAN_R = 0, 2
STEPS = 120
dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda:2'

def probe(gate, age):
    res = []
    for label, act, prm in (("stationary",STAY,1.0),("slow",PAN_R,0.2),
                            ("sweep",PAN_R,0.4),("fast",PAN_R,1.0)):
        env = E({'data_path':'data/_oneclip',
                 'detector_weights':'yolo26n_obb_topview_person_290526.pt',
                 'device':dev,'assoc_gate_deg':gate,'assoc_max_age':age})
        h=env._host; env.reset()
        rdr=_SeqReader(h.video_path); orc=Oracle(h)
        ids,ref,dets=set(),set(),0
        for t in range(STEPS):
            for p in orc.update(rdr.read(h.current_frame), t): ref.add(p['id'])
            for i in (0,1):
                for d in env.last_detections[i]:
                    ids.add(d['track_id']); dets+=1
            env.step((act,act),(prm,prm))
        rdr.close()
        res.append((label, 0.0 if act==STAY else h.pan_speed*prm, dets, len(ids), len(ref)))
    return res

for gate, age in itertools.product((10.0,), (8, 40, 120)):
    rows = probe(gate, age)
    ratios = [n/max(r,1) for _,_,_,n,r in rows[1:]]
    spread = max(ratios)-min(ratios)
    print(f"\n=== gate {gate} deg, max_age {age} steps "
          f"(moving-row ids/real spread {spread:.2f}) ===")
    for label,deg,dets,nid,nref in rows:
        print(f"  {label:<11}{deg:>7.1f} deg/step  dets {dets:>4}  ids {nid:>4}  "
              f"real {nref:>3}  ids/real {nid/max(nref,1):>5.2f}  new-id {nid/max(dets,1):>4.0%}")
