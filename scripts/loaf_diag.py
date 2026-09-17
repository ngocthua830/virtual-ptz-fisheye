import sys, math, glob, os, statistics as st
sys.path.insert(0,'/app')
import numpy as np
from evaluation.loaf import LoafSequence, load_annotations, pixel_to_bearing
ann=load_annotations('/loaf','val')
for seq in ('0051','0052','0054'):
    files=sorted(glob.glob(f'/loaf/images/val/{seq}_*.jpg'))
    nums=[int(os.path.basename(f).split('_')[1].split('.')[0]) for f in files]
    stride=st.median(np.diff(nums)) if len(nums)>1 else 0
    s=LoafSequence('/loaf','val',seq,ann=ann); s.read(0)
    # nearest-neighbour displacement between consecutive frames (ground, cm)
    def ground(t):
        out=[]
        for a in s._by_frame.get(s.frame_key(t),[]):
            if a.get('ignore') or a.get('iscrowd'): continue
            x,y,w,h=a['bbox']
            b=pixel_to_bearing(x+w/2,y+h/2,s.cx,s.cy,s.R,180.0)
            r=float(a['world_location'][0]); phi=math.radians(b[0])
            out.append((r*math.cos(phi), r*math.sin(phi)))
        return out
    d=[]
    for t in range(1,60):
        A,B=ground(t),ground(t-1)
        for p in A:
            if B: d.append(min(math.dist(p,q) for q in B))
    d=np.array(d) if d else np.array([0])
    # how many people-instances per frame vs distinct ground radii spread
    print(f"{seq}: stride {stride:.0f} frames | nn-displacement cm: "
          f"median {np.median(d):6.1f}  p90 {np.percentile(d,90):7.1f}  p99 {np.percentile(d,99):8.1f}  max {d.max():8.1f}")
