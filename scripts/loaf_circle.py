import sys, math, glob
sys.path.insert(0,'/app')
import numpy as np, cv2, json
from environments.dual_fisheye_env import DualFisheyePTZEnvironment as E
env=E({'data_path':'data/test','detector_weights':'yolo26n_obb_topview_person_290526.pt','device':'cpu'})
h=env._host
L='/loaf'
for seq in ('0051','0054','0055'):
    f=sorted(glob.glob(f'{L}/images/val/{seq}_*.jpg'))[0]
    im=cv2.imread(f)
    h.fish_R=None; h.fish_cx=None; h.fish_cy=None
    h._detect_fisheye_circle(im)
    print(f"{seq}: image {im.shape[1]}x{im.shape[0]} -> circle cx={h.fish_cx:.1f} cy={h.fish_cy:.1f} R={h.fish_R:.1f}")
