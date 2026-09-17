"""Non-PTZ baselines at a matched crop budget (review #6, #7).

If K FIXED rectilinear tiles match a controlled pair, then control -- learned OR
geometric -- is unnecessary and the PTZ formulation itself is the wrong frame.
No policy is involved: each step we render K fixed views, detect in each, and
score the union against the same pseudo-reference and 15 deg gate as Table 1.

Note on full-frame: the pseudo-reference IS the detector on the raw fisheye, so a
full-frame 'baseline' scores 1.0 by construction. It is reported only to make that
degeneracy explicit -- it cannot be compared honestly without human annotation.
"""
import sys, math, json
sys.path.insert(0,'/app')
import numpy as np
from environments.dual_fisheye_env import DualFisheyePTZEnvironment as E
from evaluation.metrics import (Oracle, _SeqReader, discovery_rate, time_to_detect,
                                observed_time_frac, max_unobserved_gap)

env = E({'data_path':'data/test','max_steps':300,
         'detector_weights':'yolo26n_obb_topview_person_290526.pt','device':'cuda'})
h = env._host
GATE, TILT, STEPS = 15.0, 45.0, 300

def run_tiles(K, tilt=TILT):
    pans = [(-180.0 + 360.0*i/K) for i in range(K)]
    gts, agents = [], []
    for clip in h.video_paths:
        h.video_path = clip
        rdr = _SeqReader(clip); orc = Oracle(h)
        for t in range(STEPS):
            f = rdr.read(t*h.frame_skip)
            if f is None: break
            if h.fish_R is None or h.fish_cx is None: h._detect_fisheye_circle(f)
            gts.append(orc.update(f, t))
            dets = []
            for pan in pans:
                view = h.project_view(f, pan, tilt, 1.0)
                for d in h._detect_objects(view):
                    dets.append(h.world_bearing(d['bbox'], pan, tilt, 1.0,
                                                h.ptz_out_w, h.ptz_out_h, h.base_fov))
            agents.append(dets)
        rdr.close()
    return dict(K=K,
                discovery=discovery_rate(gts, agents, GATE),
                ttd=time_to_detect(gts, agents, GATE),
                obsfrac=observed_time_frac(gts, agents, GATE),
                maxgap=max_unobserved_gap(gts, agents, GATE))

import os
K = int(os.environ['TILE_K'])
r = run_tiles(K)
print(f"static tiles K={K}: crops/step {K}  discovery {r['discovery']:.3f}  "
      f"ttd {r['ttd']:.2f}  obsfrac {r['obsfrac']:.3f}  maxgap {r['maxgap']:.2f}")
json.dump(r, open(f'/app/results/full128_bearing/tiles_K{K}.json','w'), indent=1)
print("\nOvlp-Sweep (controlled, K=2) for reference: 0.92 / 1.09 / 0.64 / 3.59")
