"""Measured cost of the pipeline (review #48-#50).

The paper's framing says detector throughput, not viewing direction, is the scarce
resource. That claim needs numbers: per-crop render and detect time, policy
inference time, and the cost of a full-frame pass for comparison -- because K
crops and one full frame are NOT the same compute just because K is small.
"""
import sys, time
sys.path.insert(0,'/app')
import numpy as np, torch
from environments.dual_fisheye_env import DualFisheyePTZEnvironment as E

env = E({'data_path':'data/test','max_steps':300,
         'detector_weights':'yolo26n_obb_topview_person_290526.pt','device':'cuda'})
h=env._host; env.reset()
frame = h._get_frame()
print('GPU:', torch.cuda.get_device_name(0))
print('source frame:', frame.shape)

def timeit(fn, n=30, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/n*1000

t_render = timeit(lambda: h.project_view(frame, 30.0, 45.0, 1.0))
view = h.project_view(frame, 30.0, 45.0, 1.0)
t_detect = timeit(lambda: h._detect_objects(view, pose=(30.0,45.0,1.0), step=1))
t_full   = timeit(lambda: h.detector.predict(frame, conf=h.yolo_conf, verbose=False), n=15)
import cv2
small = cv2.resize(frame, (1024,1024))
t_full1024 = timeit(lambda: h.detector.predict(small, conf=h.yolo_conf, verbose=False), n=15)

print(f"\n  render one 960x540 crop        : {t_render:7.1f} ms")
print(f"  detect one 960x540 crop        : {t_detect:7.1f} ms")
print(f"  => K=2 controlled crops        : {2*(t_render+t_detect):7.1f} ms/step")
print(f"  detect full fisheye 2320x2320  : {t_full:7.1f} ms")
print(f"  detect full fisheye @1024      : {t_full1024:7.1f} ms")
px_crop = 2*960*540; px_full = 2320*2320; px_1024=1024*1024
print(f"\n  pixels: 2 crops {px_crop/1e6:.2f} MP | full {px_full/1e6:.2f} MP"
      f" | full@1024 {px_1024/1e6:.2f} MP")
print(f"  2-crop budget is {px_full/px_crop:.1f}x FEWER pixels than the raw frame,"
      f" {px_crop/px_1024:.2f}x vs a 1024 resize")
