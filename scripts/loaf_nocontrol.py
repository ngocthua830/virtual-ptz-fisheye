"""Two no-control baselines on LOAF, where the reference is HUMAN so neither is degenerate:
   (a) full-frame  : the detector on the raw 2048^2 fisheye, no crops at all;
   (b) static tiles: K fixed rectilinear crops, evenly spaced azimuth at fixed tilt.
Both scored against the same human annotations and 15 deg gate as Table 2."""
import sys, math, json
sys.path.insert(0,'/app')
from evaluation.loaf import LoafSequence, load_annotations
from evaluation.metrics import (discovery_rate, time_to_detect, observed_time_frac,
                                max_unobserved_gap, fisheye_pixel_to_bearing)
from environments.dual_fisheye_env import DualFisheyePTZEnvironment as E

SEQS=['0062','0055','0071','0051','0052','0054']
STEPS=300; GATE=15.0
env=E({'data_path':'data/test','max_steps':STEPS,'state_dim':29,
       'detector_weights':'yolo26n_obb_topview_person_290526.pt','loop_video':False})
h=env._host
out={}
for sq in SEQS:
    ann=load_annotations('/loaf','val',sequences=[sq])
    s=LoafSequence('/loaf','val',sq,ann=ann); s.read(0)
    h.fish_cx,h.fish_cy,h.fish_R=s.cx,s.cy,s.R; h.fish_fov_deg=180.0
    gt=s.reference(max_frames=STEPS)
    n=min(STEPS,len(gt))
    res={}
    # (a) full frame
    ff=[]
    for t in range(n):
        f=s.read(t); dets=[]
        for d in h._detect_objects(f):
            x1,y1,x2,y2=d['bbox']
            dets.append(fisheye_pixel_to_bearing((x1+x2)/2,(y1+y2)/2,s.cx,s.cy,s.R,180.0))
        ff.append(dets)
    res['fullframe']=dict(discovery=discovery_rate(gt[:n],ff,GATE), ttd=time_to_detect(gt[:n],ff,GATE),
                          obsfrac=observed_time_frac(gt[:n],ff,GATE), maxgap=max_unobserved_gap(gt[:n],ff,GATE))
    # (b) static tiles, K=2 and K=4, at the LOSO-selected tilt 65
    for K in (2,4):
        pans=[(-180.0+360.0*i/K) for i in range(K)]
        ag=[]
        for t in range(n):
            f=s.read(t); dets=[]
            for pan in pans:
                v=h.project_view(f,pan,65.0,1.0)
                for d in h._detect_objects(v,pose=(pan,65.0,1.0),step=t):
                    dets.append(h.world_bearing(d['bbox'],pan,65.0,1.0,h.ptz_out_w,h.ptz_out_h,h.base_fov))
            ag.append(dets)
        res[f'tiles{K}']=dict(discovery=discovery_rate(gt[:n],ag,GATE), ttd=time_to_detect(gt[:n],ag,GATE),
                              obsfrac=observed_time_frac(gt[:n],ag,GATE), maxgap=max_unobserved_gap(gt[:n],ag,GATE))
    out[sq]=res
    print(f"{sq}: " + "  ".join(f"{k} disc {v['discovery']:.3f} obs {v['obsfrac']:.3f}" for k,v in res.items()), flush=True)
    json.dump(out, open('/app/results/loaf/nocontrol.json','w'), indent=1)
