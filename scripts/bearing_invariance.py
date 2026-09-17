import sys, math, numpy as np
sys.path.insert(0,'/app')
from environments.fisheye_env import FisheyePTZEnvironment as F, _angular_sep_deg
W,H,FOV = 960,540,90.0
def project(bearing, pan, tilt, zoom=1.0):
    az,pol = math.radians(bearing[0]), math.radians(bearing[1])
    v = np.array([math.sin(pol)*math.cos(az), math.sin(pol)*math.sin(az), -math.cos(pol)])  # polar from nadir
    t,p = math.radians(tilt), math.radians(pan)
    Rz = np.array([[math.cos(p),-math.sin(p),0],[math.sin(p),math.cos(p),0],[0,0,1]])
    Rx = np.array([[1,0,0],[0,math.cos(t),-math.sin(t)],[0,math.sin(t),math.cos(t)]])
    c = Rx.T @ (Rz.T @ v)
    f = (W/2)/math.tan(math.radians(FOV/zoom)/2)
    if -c[2] <= 1e-9: return None
    x = c[0]*f/(-c[2]) + W/2; y = c[1]*f/(-c[2]) + H/2
    if not (0<=x<W and 0<=y<H): return None
    return [x-2,y-2,x+2,y+2]
truth = (37.0, 52.0)
print(f"stationary world point at bearing {truth}; camera pans 25 deg/step\n")
print(f"  {'pan':>5} {'recovered bearing':>22} {'err':>9}  {'img IoU vs prev':>15}")
prev=None; maxerr=0.0
for pan in range(0, 51, 25):
    box = project(truth, pan, 45.0)
    if box is None:
        print(f"  {pan:>5} {'(out of view)':>22}"); prev=None; continue
    b = F.world_bearing(box, pan, 45.0, 1.0, W, H, FOV)
    err = _angular_sep_deg(b, truth); maxerr=max(maxerr,err)
    if prev:
        xx1=max(box[0],prev[0]); yy1=max(box[1],prev[1])
        xx2=min(box[2],prev[2]); yy2=min(box[3],prev[3])
        inter=max(0,xx2-xx1)*max(0,yy2-yy1)
        iou=inter/(32.0-inter) if inter>0 else 0.0
    else: iou=float('nan')
    print(f"  {pan:>5} {f'({b[0]:.2f}, {b[1]:.2f})':>22} {err:>8.4f}d {iou:>15.3f}")
    prev=box
print(f"\nmax bearing error across the pan sweep: {maxerr:.6f} deg")
print("-> bearing is invariant to the camera's own motion; image-space IoU is not.")
