"""Where do people actually appear in polar angle, and does Ovlp-Sweep ever look there?

A view at tilt 45 covers polar 15.9-78.1 deg, so the co-rotating pair never images
the cone within ~16 deg of nadir (directly beneath the camera) nor the last ~12 deg
toward the horizon. This asks how much of the reference cohort lives there.
"""
import sys, numpy as np
sys.path.insert(0,'/app')
from environments.dual_fisheye_env import DualFisheyePTZEnvironment as E
from evaluation.metrics import Oracle, _SeqReader

env = E({'data_path':'data/test','max_steps':300,
         'detector_weights':'yolo26n_obb_topview_person_290526.pt','device':'cuda'})
h = env._host
LO, HI = 15.9, 78.1
allpol, first = [], {}
for clip in h.video_paths:
    h.video_path = clip
    rdr = _SeqReader(clip); orc = Oracle(h)
    for t in range(300):
        f = rdr.read(t*h.frame_skip)
        if f is None: break
        if h.fish_R is None or h.fish_cx is None:
            h._detect_fisheye_circle(f)
        for p in orc.update(f, t):
            allpol.append(p['bearing'][1])
            first.setdefault((clip,p['id']), p['bearing'][1])
    rdr.close()
a = np.array(allpol); fp = np.array(list(first.values()))
print(f"reference detections: {len(a)}   distinct reference tracks: {len(fp)}")
for name, arr in (("all detections", a), ("track first-appearance", fp)):
    print(f"  {name}:")
    print(f"     polar  min {arr.min():5.1f}  median {np.median(arr):5.1f}  max {arr.max():5.1f}")
    print(f"     inside nadir blind cone (<{LO} deg): {100*(arr<LO).mean():5.1f}%")
    print(f"     beyond outer edge      (>{HI} deg): {100*(arr>HI).mean():5.1f}%")
    print(f"     inside the swept band          : {100*((arr>=LO)&(arr<=HI)).mean():5.1f}%")
