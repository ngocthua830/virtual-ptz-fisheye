"""Sanity-check the static-tile result: what do 2 fixed tiles actually cover,
and what fraction of reference detections fall inside them?"""
import sys, math
sys.path.insert(0,'/app')
import numpy as np
from environments.dual_fisheye_env import DualFisheyePTZEnvironment as E
from evaluation.metrics import Oracle, _SeqReader, _in_frustum, _sphere_grid, frustum_solid_angle

V,W = _sphere_grid()
def union_mask(pans, tilt=45.0):
    m = np.zeros(len(V), bool)
    for p in pans: m |= _in_frustum(V, p, tilt, 1.0, 90.0, 960, 540)
    return m
for K in (2,4):
    pans=[(-180.0+360.0*i/K) for i in range(K)]
    m=union_mask(pans)
    frac_hemi = W[m].sum()/(2*math.pi)          # fraction of the LOWER hemisphere
    print(f"K={K} tiles at tilt 45: pans {['%.0f'%p for p in pans]}"
          f" -> covers {W[m].sum():.3f} sr = {100*frac_hemi:.1f}% of the hemisphere")

env = E({'data_path':'data/test','max_steps':300,
         'detector_weights':'yolo26n_obb_topview_person_290526.pt','device':'cuda'})
h=env._host
def inside(bearing, pans, tilt=45.0):
    az,pol=math.radians(bearing[0]),math.radians(bearing[1])
    v=np.array([[math.sin(pol)*math.cos(az), math.sin(pol)*math.sin(az), -math.cos(pol)]])
    return any(_in_frustum(v,p,tilt,1.0,90.0,960,540)[0] for p in pans)

for K in (2,4):
    pans=[(-180.0+360.0*i/K) for i in range(K)]
    tot=ins=0; ids=set(); ids_in=set()
    for clip in h.video_paths:
        h.video_path=clip
        rdr=_SeqReader(clip); orc=Oracle(h)
        for t in range(300):
            f=rdr.read(t*h.frame_skip)
            if f is None: break
            if h.fish_R is None or h.fish_cx is None: h._detect_fisheye_circle(f)
            for p in orc.update(f,t):
                tot+=1; ids.add((clip,p['id']))
                if inside(p['bearing'],pans):
                    ins+=1; ids_in.add((clip,p['id']))
        rdr.close()
    print(f"K={K}: {100*ins/tot:5.1f}% of reference detections inside the tiles | "
          f"{100*len(ids_in)/len(ids):5.1f}% of reference tracks enter them at least once")
