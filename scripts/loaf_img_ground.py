"""Compare two ways of getting a ground position for association:
   (a) world_location radius  (dataset-provided, noisy in some sequences)
   (b) image polar angle + camera height  (r = (cam_h - h_person) * tan(polar))"""
import sys, math
sys.path.insert(0,'/app')
import numpy as np
from evaluation.loaf import LoafSequence, load_annotations, sequences, pixel_to_bearing
ann=load_annotations('/loaf','val')
NOMINAL_H=170.0
print(f"{'seq':>6} | {'world_location p90':>18} | {'image-derived p90':>18}")
for seq in sequences('/loaf','val'):
    s=LoafSequence('/loaf','val',seq,ann=ann); s.read(0)
    def grounds(t):
        A=[];B=[]
        for a in s._by_frame.get(s.frame_key(t),[]):
            if a.get('ignore') or a.get('iscrowd'): continue
            x,y,w,h=a['bbox']
            b=pixel_to_bearing(x+w/2,y+h/2,s.cx,s.cy,s.R,180.0)
            phi=math.radians(b[0]); pol=math.radians(b[1])
            camh=float(a.get('camera_height') or 300.0)
            r_w=float(a['world_location'][0])
            r_i=max(camh-NOMINAL_H,1.0)*math.tan(min(pol, math.radians(88.0)))
            A.append((r_w*math.cos(phi), r_w*math.sin(phi)))
            B.append((r_i*math.cos(phi), r_i*math.sin(phi)))
        return A,B
    dw=[];di=[]
    for t in range(1,80):
        (Aw,Ai),(Bw,Bi)=grounds(t),grounds(t-1)
        for p in Aw:
            if Bw: dw.append(min(math.dist(p,q) for q in Bw))
        for p in Ai:
            if Bi: di.append(min(math.dist(p,q) for q in Bi))
    print(f"{seq:>6} | {np.percentile(dw,90):>18.0f} | {np.percentile(di,90):>18.0f}")
