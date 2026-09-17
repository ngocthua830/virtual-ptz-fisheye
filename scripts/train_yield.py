"""Pick DENSE INDOOR LOAF training sequences by detector yield.

Density alone is the wrong criterion: the densest train sequences are outdoor
plazas where our indoor-fine-tuned detector finds almost nobody, so training
there would optimise against an empty reward signal.
"""
import sys, json, collections, statistics as st
sys.path.insert(0,'/app')
from evaluation.loaf import LoafSequence
from environments.dual_fisheye_env import DualFisheyePTZEnvironment as E

CAND=['0017','0021','0045','0027','0039','0048','0046','0043','0022','0001',
      '0002','0003','0005','0008','0012','0015','0019','0024','0031','0041']
d=json.load(open('/loaf/annotations/resolution_2k/instances_train.json'))
img={i['id']:i['file_name'] for i in d['images']}
per=collections.Counter(a['image_id'] for a in d['annotations'])
dens=collections.defaultdict(list)
for iid,n in per.items():
    s=img[iid].split('_')[0]
    if s in CAND: dens[s].append(n)
del d
env=E({'data_path':'data/test','detector_weights':'yolo26n_obb_topview_person_290526.pt','device':'cuda'})
h=env._host
print(f"{'seq':>6} {'ppl/frame':>10} {'det/frame':>10} {'yield':>7}")
rows=[]
for sq in sorted(dens, key=lambda s:-st.mean(dens[s])):
    try:
        s=LoafSequence('/loaf','train',sq,ann={}); s.read(0)
    except Exception:
        continue
    h.fish_cx,h.fish_cy,h.fish_R=s.cx,s.cy,s.R; h.fish_fov_deg=180.0
    got=[len(h._detect_objects(s.read(t))) for t in range(0,40,8)]
    a=st.mean(dens[sq]); dd=st.mean(got)
    rows.append((dd/max(a,1e-9), a, dd, sq))
    print(f"{sq:>6} {a:>10.1f} {dd:>10.1f} {dd/max(a,1e-9):>7.2f}", flush=True)
rows.sort(reverse=True)
good=[r for r in rows if r[0]>=0.25 and r[1]>=10]
print("\nDENSE + DETECTABLE (yield>=0.25, >=10 people/frame):")
for y,a,dd,sq in good[:8]:
    print(f"  {sq}: {a:.1f} people/frame, yield {y:.2f}")
print("\n--loaf_seqs", ",".join(sq for _,_,_,sq in good[:6]))
