"""Does the detector actually work on each val sequence we drew conclusions from?
Compares full-frame detector yield against the human annotation count."""
import sys, statistics as st
sys.path.insert(0,'/app')
from evaluation.loaf import LoafSequence, load_annotations
from environments.dual_fisheye_env import DualFisheyePTZEnvironment as E
env=E({'data_path':'data/test','detector_weights':'yolo26n_obb_topview_person_290526.pt','device':'cuda'})
h=env._host
SEQS=['0055','0062','0071','0051','0052','0054']
ann=load_annotations('/loaf','val',sequences=SEQS)
print(f"{'seq':>6} {'annotated/frame':>16} {'detected/frame':>15} {'recall-ish':>11}")
for sq in SEQS:
    s=LoafSequence('/loaf','val',sq,ann=ann); s.read(0)
    h.fish_cx,h.fish_cy,h.fish_R=s.cx,s.cy,s.R; h.fish_fov_deg=180.0
    a_cnt,d_cnt=[],[]
    for t in range(0,60,6):
        f=s.read(t)
        a_cnt.append(len([x for x in s._by_frame.get(s.frame_key(t),[]) if not (x.get('ignore') or x.get('iscrowd'))]))
        d_cnt.append(len(h._detect_objects(f)))
    ra,rd=st.mean(a_cnt),st.mean(d_cnt)
    print(f"{sq:>6} {ra:>16.1f} {rd:>15.1f} {rd/max(ra,1e-9):>11.2f}")
