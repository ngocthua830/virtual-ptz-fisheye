"""FSAC -- Four-Sector Alternating Coverage: a budget-matched NO-CONTROL baseline.

Four fixed virtual views partition azimuth at pan = 0/90/180/270 deg, common tilt.
Each step activates only the ANTIPODAL PAIR, alternating:

    t even -> {  0 deg, 180 deg }
    t odd  -> { 90 deg, 270 deg }

so the crop budget is K=2 per step -- identical to GeoSweep, MP-DQN and PPO -- while
the whole azimuth circle is revisited every TWO steps. The active pair is always
180 deg apart, so it inherits GeoSweep's disjoint-by-construction property.

The point: a virtual PTZ has no mechanical slew cost, so a fixed schedule may jump
between sectors freely. This asks whether ANY control -- geometric or learned -- is
needed once that is exploited. Like `tile_baseline.py` this is a no-control
reference: no policy, no detector feedback, no training.
"""
import sys, json, os
sys.path.insert(0, '/app')
import numpy as np
from environments.dual_fisheye_env import DualFisheyePTZEnvironment as E
from evaluation.metrics import (Oracle, _SeqReader, discovery_rate, time_to_detect,
                                observed_time_frac, max_unobserved_gap,
                                time_to_detect_mover)

GATE, STEPS = 15.0, 300
TILT = float(os.environ.get('FSAC_TILT', 45.0))
PHASE = [(0.0, 180.0), (90.0, 270.0)]          # alternating antipodal pairs

env = E({'data_path': 'data/test', 'max_steps': STEPS,
         'detector_weights': 'yolo26n_obb_topview_person_290526.pt', 'device': 'cuda'})
h = env._host

gts, agents = [], []
for clip in h.video_paths:
    h.video_path = clip
    rdr = _SeqReader(clip); orc = Oracle(h)
    for t in range(STEPS):
        f = rdr.read(t * h.frame_skip)
        if f is None:
            break
        if h.fish_R is None or h.fish_cx is None:
            h._detect_fisheye_circle(f)
        gts.append(orc.update(f, t))
        dets = []
        for pan in PHASE[t % 2]:               # <- exactly 2 crops per step
            view = h.project_view(f, pan, TILT, 1.0)
            for d in h._detect_objects(view):
                dets.append(h.world_bearing(d['bbox'], pan, TILT, 1.0,
                                            h.ptz_out_w, h.ptz_out_h, h.base_fov))
        agents.append(dets)
    rdr.close()

r = dict(name='fsac', crops_per_step=2, tilt=TILT,
         discovery=discovery_rate(gts, agents, GATE),
         ttd=time_to_detect(gts, agents, GATE),
         obsfrac=observed_time_frac(gts, agents, GATE),
         maxgap=max_unobserved_gap(gts, agents, GATE),
         ttd_mover=time_to_detect_mover(gts, agents, GATE))
print(f"FSAC (K=2, tilt {TILT:.0f}): discovery {r['discovery']:.3f}  ttd {r['ttd']:.2f}  "
      f"obs {r['obsfrac']:.3f}  gap {r['maxgap']:.2f}  ttd_mover {r['ttd_mover']:.2f}")
print("GeoSweep (K=2, controlled) reference: 0.92 / 1.09 / 0.64 / 3.59 / 0.72")
OUT = os.environ.get('FSAC_OUT', '/app/results/full128_bearing/fsac.json')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(r, open(OUT, 'w'), indent=1)
print('wrote', OUT)
