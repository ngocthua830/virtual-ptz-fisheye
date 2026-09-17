"""Does the novelty reward measure 'new person' or 'camera moved'?

The OBB path assigns identities by axis-aligned image-space IoU>=0.4 between
consecutive detector CALLS, with no ego-motion compensation and a pool shared
by both virtual cameras. Hold the scene fixed and vary only camera motion: if
the id count tracks pan speed rather than the number of people, the novelty
signal is measuring self-motion.
"""
import sys, numpy as np
sys.path.insert(0, '/app')
from environments.dual_fisheye_env import DualFisheyePTZEnvironment as E
from evaluation.metrics import Oracle, _SeqReader

STAY, PAN_L, PAN_R = 0, 1, 2
CFG = dict(data_path='data/test',
           detector_weights='yolo26n_obb_topview_person_290526.pt',
           device=sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
STEPS = 120

def run(label, action, param):
    env = E(dict(CFG)); h = env._host
    env.reset()
    rdr = _SeqReader(h.video_path); orc = Oracle(h)
    ids, gt_ids, dets_total = set(), set(), 0
    for t in range(STEPS):
        frame = rdr.read(h.current_frame)
        for p in orc.update(frame, t):
            gt_ids.add(p['id'])
        for i in (0, 1):
            for d in env.last_detections[i]:
                ids.add(d['track_id']); dets_total += 1
        env.step((action, action), (param, param))
    rdr.close()
    deg = {STAY: 0.0, PAN_R: h.pan_speed * param, PAN_L: h.pan_speed * param}[action]
    print(f"  {label:<22} pan {deg:5.1f} deg/step | agent ids {len(ids):5d} | "
          f"pseudo-ref ids {len(gt_ids):4d} | ratio {len(ids)/max(len(gt_ids),1):6.2f} | "
          f"dets {dets_total}")
    return len(ids), len(gt_ids)

print(f"clip-fixed id-churn probe, {STEPS} steps, same video for every row")
run("stationary (STAY)",   STAY,  1.0)
run("slow pan (p=0.2)",    PAN_R, 0.2)
run("sweep pan (p=0.4)",   PAN_R, 0.4)
run("fast pan (p=1.0)",    PAN_R, 1.0)
