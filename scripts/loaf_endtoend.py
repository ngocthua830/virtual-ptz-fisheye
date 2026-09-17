"""Aim a virtual view at a human-annotated bearing; the person must be there."""
import sys, math
sys.path.insert(0,'/app')
import cv2, numpy as np
from evaluation.loaf import LoafSequence, load_annotations
from environments.dual_fisheye_env import DualFisheyePTZEnvironment as E

env=E({'data_path':'data/test','detector_weights':'yolo26n_obb_topview_person_290526.pt','device':'cuda'})
h=env._host
ann=load_annotations('/loaf','val')
s=LoafSequence('/loaf','val','0051',ann=ann)
gt=s.reference(max_frames=3)
frame=s.read(0)
h.fish_cx, h.fish_cy, h.fish_R = s.cx, s.cy, s.R
h.fish_fov_deg = 180.0

people=gt[0]
print(f"frame 0 has {len(people)} annotated people")
# the renderer's azimuth convention is pan+90 (see world_bearing derivation)
tiles=[]
for p in people[:4]:
    az,pol = p['bearing']
    pan = az - 90.0
    view = h.project_view(frame, pan, pol, 1.0)
    det = h._detect_objects(view, pose=(pan,pol,1.0), step=0)
    # how close is the nearest detection to the view centre?
    best=None
    for d in det:
        b=h.world_bearing(d['bbox'],pan,pol,1.0,h.ptz_out_w,h.ptz_out_h,h.base_fov)
        v1=np.array([math.sin(math.radians(b[1]))*math.cos(math.radians(b[0])),
                     math.sin(math.radians(b[1]))*math.sin(math.radians(b[0])),
                     -math.cos(math.radians(b[1]))])
        v2=np.array([math.sin(math.radians(pol))*math.cos(math.radians(az)),
                     math.sin(math.radians(pol))*math.sin(math.radians(az)),
                     -math.cos(math.radians(pol))])
        sep=math.degrees(math.acos(max(-1,min(1,float(v1@v2)))))
        if best is None or sep<best: best=sep
    print(f"  id {p['id']:>3} bearing=({az:7.1f},{pol:5.1f}) -> {len(det):2d} detections in view,"
          f" nearest {('%.1f deg'%best) if best is not None else 'NONE'}")
    tiles.append(view)
if tiles:
    grid=np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])]) if len(tiles)>=4 else np.hstack(tiles)
    grid=cv2.resize(grid,(min(900,grid.shape[1]//2), grid.shape[0]*min(900,grid.shape[1]//2)//grid.shape[1]))
    cv2.imwrite('loaf_views.jpg', grid); print('wrote loaf_views.jpg')
