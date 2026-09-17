"""Exact rectilinear-frustum vs circular-cone overlap, using the SAME camera
basis as FisheyePTZEnvironment.project_view (tilt about X, then pan about Z,
optical axis -Z, f = (W/2)/tan(HFOV/2))."""
import numpy as np, math

W, H, BASE_FOV = 960, 540, 90.0

def basis(pan_deg, tilt_deg):
    """Return (right, up, fwd) world vectors for the virtual camera."""
    t, p = math.radians(tilt_deg), math.radians(pan_deg)
    def rot(v):
        x, y, z = v
        ry = y*math.cos(t) - z*math.sin(t)
        rz = y*math.sin(t) + z*math.cos(t)
        rx = x
        return np.array([rx*math.cos(p) - ry*math.sin(p),
                         rx*math.sin(p) + ry*math.cos(p), rz])
    return rot((1,0,0)), rot((0,1,0)), rot((0,0,-1))

def in_frustum(V, pan, tilt, zoom):
    """EXACT: ray is in view iff it projects inside the W x H image."""
    fov = BASE_FOV / zoom
    f = (W/2.0)/math.tan(math.radians(fov)/2.0)
    r, u, fw = basis(pan, tilt)
    z = V @ fw
    ok = z > 1e-9
    zz = np.where(ok, z, 1.0)
    x = (V @ r)/zz * f
    y = (V @ u)/zz * f
    return ok & (np.abs(x) <= W/2.0) & (np.abs(y) <= H/2.0)

def in_cone(V, pan, tilt, zoom):
    """What the paper's metrics actually do: circular cap, half-angle fov/2."""
    _, _, fw = basis(pan, tilt)
    return (V @ fw) >= math.cos(math.radians(0.5*BASE_FOV/zoom))

def grid(n_theta=900, n_phi=1800):
    th = (np.arange(n_theta)+0.5)*(math.pi/n_theta)
    ph = (np.arange(n_phi)+0.5)*(2*math.pi/n_phi)
    T, P = np.meshgrid(th, ph, indexing='ij')
    st = np.sin(T)
    V = np.stack([st*np.cos(P), st*np.sin(P), np.cos(T)], -1).reshape(-1,3)
    Wt = (st*(math.pi/n_theta)*(2*math.pi/n_phi)).reshape(-1)
    return V, Wt

V, Wt = grid()
hfov = BASE_FOV
vfov = math.degrees(2*math.atan((H/2.0)/((W/2.0)/math.tan(math.radians(hfov)/2))))
corner = math.degrees(math.atan(math.hypot(W/2, H/2)/((W/2.0)/math.tan(math.radians(hfov)/2))))
print(f"HFOV {hfov:.2f}  VFOV {vfov:.2f}  corner half-angle {corner:.2f}  (cone half-angle used: {hfov/2:.2f})")
o_f = Wt[in_frustum(V,0,0,1)].sum(); o_c = Wt[in_cone(V,0,0,1)].sum()
print(f"solid angle: frustum {o_f:.4f} sr | cone {o_c:.4f} sr | cone/frustum = {o_c/o_f:.3f}")

print("\n tilt | axis sep |  cone IoU | frustum IoU | cone says | frustum says")
for tilt in [30,35,40,44,45,46,48,50,55,58,60,65,70]:
    A = dict(pan=0.0, tilt=tilt, zoom=1.0); B = dict(pan=180.0, tilt=tilt, zoom=1.0)
    for name, fn in (("cone", in_cone), ("frustum", in_frustum)):
        i1 = fn(V, A['pan'], A['tilt'], 1.0); i2 = fn(V, B['pan'], B['tilt'], 1.0)
        inter = Wt[i1 & i2].sum(); union = Wt[i1 | i2].sum()
        globals()[name+"_iou"] = inter/union if union > 0 else 0.0
    print(f" {tilt:>4} | {2*tilt:>7}° | {cone_iou:9.5f} | {frustum_iou:11.5f} | "
          f"{'OVERLAP' if cone_iou>1e-6 else 'disjoint':>8} | {'OVERLAP' if frustum_iou>1e-6 else 'disjoint'}")
