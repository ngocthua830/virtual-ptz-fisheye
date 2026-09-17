import sys, math
sys.path.insert(0,'/app')
import numpy as np
from evaluation.loaf import LoafSequence, load_annotations, sequences, pixel_to_bearing
ann=load_annotations('/loaf','val')
print(f"{'seq':>6} {'med':>7} {'p75':>7} {'p90':>8} {'radius med':>11}  (cm)")
for seq in sequences('/loaf','val'):
    s=LoafSequence('/loaf','val',seq,ann=ann); s.read(0)
    def ground(t):
        out=[]
        for a in s._by_frame.get(s.frame_key(t),[]):
            if a.get('ignore') or a.get('iscrowd'): continue
            x,y,w,h=a['bbox']
            b=pixel_to_bearing(x+w/2,y+h/2,s.cx,s.cy,s.R,180.0)
            r=float(a['world_location'][0]); phi=math.radians(b[0])
            out.append((r*math.cos(phi), r*math.sin(phi), r))
        return out
    d=[]; rad=[]
    for t in range(1,80):
        A,B=ground(t),ground(t-1)
        rad += [p[2] for p in A]
        for p in A:
            if B: d.append(min(math.dist(p[:2],q[:2]) for q in B))
    d=np.array(d); rad=np.array(rad)
    print(f"{seq:>6} {np.median(d):>7.0f} {np.percentile(d,75):>7.0f} {np.percentile(d,90):>8.0f} {np.median(rad):>11.0f}")
