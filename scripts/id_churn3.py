"""Does world-bearing association remove the camera-motion dependence?

Same probe as before: one clip, identical scene and detector, only the pan rate
varies. A correct identity signal should give a count near the real people and a
ratio flat in pan rate.
"""
import sys
sys.path.insert(0, '/app')
from environments.dual_fisheye_env import DualFisheyePTZEnvironment as E
from evaluation.metrics import Oracle, _SeqReader

STAY, PAN_R = 0, 2
STEPS = 120
dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda:2'

def probe(gate):
    out = []
    for label, act, prm in (("stationary", STAY, 1.0), ("slow", PAN_R, 0.2),
                            ("sweep", PAN_R, 0.4), ("fast", PAN_R, 1.0)):
        env = E({'data_path': 'data/_oneclip',
                 'detector_weights': 'yolo26n_obb_topview_person_290526.pt',
                 'device': dev, 'assoc_gate_deg': gate})
        h = env._host; env.reset()
        rdr = _SeqReader(h.video_path); orc = Oracle(h)
        ids, ref, dets = set(), set(), 0
        for t in range(STEPS):
            for p in orc.update(rdr.read(h.current_frame), t):
                ref.add(p['id'])
            for i in (0, 1):
                for d in env.last_detections[i]:
                    ids.add(d['track_id']); dets += 1
            env.step((act, act), (prm, prm))
        rdr.close()
        out.append((label, 0.0 if act == STAY else h.pan_speed*prm, dets, len(ids), len(ref)))
    return out

for gate in (6.0, 10.0, 15.0):
    print(f"\n=== assoc_gate_deg = {gate} ===")
    print(f"  {'motion':<12}{'deg/step':>9}{'dets':>7}{'ids':>7}{'real':>6}{'ids/real':>10}{'new-id':>8}")
    for label, deg, dets, nid, nref in probe(gate):
        print(f"  {label:<12}{deg:>9.1f}{dets:>7d}{nid:>7d}{nref:>6d}"
              f"{nid/max(nref,1):>10.2f}{nid/max(dets,1):>8.0%}")
