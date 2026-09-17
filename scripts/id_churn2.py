"""Same clip, same scene, same detector -- only the camera's own motion changes.

Identities come from axis-aligned image-space IoU>=0.4 between consecutive
detector calls, with no ego-motion compensation and one pool shared by both
virtual views. If the id count scales with pan speed while the number of real
people does not, the explorer's novelty bonus is paid for self-motion.
"""
import sys
sys.path.insert(0, '/app')
from environments.dual_fisheye_env import DualFisheyePTZEnvironment as E
from evaluation.metrics import Oracle, _SeqReader

STAY, PAN_R = 0, 2
CFG = dict(data_path='data/_oneclip',
           detector_weights='yolo26n_obb_topview_person_290526.pt',
           device=sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
STEPS = 120

print(f"one clip (1779692917083), {STEPS} steps, identical scene in every row\n")
print(f"  {'camera motion':<20} {'deg/step':>9} {'dets':>6} {'agent ids':>10} "
      f"{'ref ids':>8} {'ids/ref':>8} {'new-id rate':>12}")
for label, act, prm in (("stationary", STAY, 1.0), ("slow pan", PAN_R, 0.2),
                        ("sweep pan", PAN_R, 0.4), ("fast pan", PAN_R, 1.0)):
    env = E(dict(CFG)); h = env._host; env.reset()
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
    deg = 0.0 if act == STAY else h.pan_speed * prm
    print(f"  {label:<20} {deg:>9.1f} {dets:>6d} {len(ids):>10d} {len(ref):>8d} "
          f"{len(ids)/max(len(ref),1):>8.2f} {len(ids)/max(dets,1):>11.0%}")
