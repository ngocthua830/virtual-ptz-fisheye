"""Does the detector fire more in a ZOOMED crop? If so, a policy that zooms wins
on 'discovery' partly by finding the detector's operating point, not by better
coverage -- which must be reported as such."""
import sys, statistics as st
sys.path.insert(0,'/app')
from evaluation.loaf import LoafSequence, load_annotations
from environments.dual_fisheye_env import DualFisheyePTZEnvironment as E
env=E({'data_path':'data/test','detector_weights':'yolo26n_obb_topview_person_290526.pt','device':'cuda'})
h=env._host
SEQS=['0055','0052','0054','0051']
ann=load_annotations('/loaf','val',sequences=SEQS)
print(f"{'seq':>6} {'tilt':>5} " + " ".join(f"z={z:<4}" for z in (1.0,1.5,2.0,3.0)))
for sq in SEQS:
    s=LoafSequence('/loaf','val',sq,ann=ann); s.read(0)
    h.fish_cx,h.fish_cy,h.fish_R=s.cx,s.cy,s.R; h.fish_fov_deg=180.0
    for tilt in (45.0,60.0,75.0):
        row=[]
        for z in (1.0,1.5,2.0,3.0):
            tot=0; n=0
            for t in range(0,48,8):
                f=s.read(t)
                for pan in (-180.0,-90.0,0.0,90.0):
                    v=h.project_view(f,pan,tilt,z)
                    tot+=len(h._detect_objects(v,pose=(pan,tilt,z),step=t)); n+=1
            row.append(tot/n)
        print(f"{sq:>6} {tilt:>5.0f} " + " ".join(f"{v:<6.2f}" for v in row))
