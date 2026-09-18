"""Build the problem figure: one overhead fisheye -> two virtual PTZ views.
Replicates fisheye_env.project_view exactly (equidistant, r = f_fish * theta)."""
import cv2, numpy as np, sys

OUT_W, OUT_H, BASE_FOV, FISH_FOV = 960, 540, 90.0, 180.0

def project_view(frame, pan, tilt, zoom=1.0, out_w=OUT_W, out_h=OUT_H):
    h, w = frame.shape[:2]
    cx, cy, R = w/2.0, h/2.0, min(w, h)/2.0
    fov = np.radians(BASE_FOV/max(zoom,1e-6)); f=(out_w/2.0)/np.tan(fov/2.0)
    xs=np.arange(out_w,dtype=np.float32)-out_w/2.0
    ys=np.arange(out_h,dtype=np.float32)-out_h/2.0
    xx,yy=np.meshgrid(xs,ys)
    dz=-np.full_like(xx,f,dtype=np.float32)
    t=np.radians(tilt); ct,st=np.cos(t),np.sin(t)
    rx,ry,rz = xx, yy*ct-dz*st, yy*st+dz*ct
    p=np.radians(pan); cp,sp=np.cos(p),np.sin(p)
    wx,wy,wz = rx*cp-ry*sp, rx*sp+ry*cp, rz
    r3=np.sqrt(wx*wx+wy*wy+wz*wz)+1e-9
    theta=np.arccos(np.clip(-wz/r3,-1,1)); phi=np.arctan2(wy,wx)
    f_fish=R/(np.radians(FISH_FOV)/2.0); r_pix=f_fish*theta
    mx=(cx+r_pix*np.cos(phi)).astype(np.float32)
    my=(cy+r_pix*np.sin(phi)).astype(np.float32)
    inside=r_pix<=R
    mx=np.where(inside,mx,-1).astype(np.float32); my=np.where(inside,my,-1).astype(np.float32)
    return cv2.remap(frame,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0)), (mx,my)

def footprint(frame, pan, tilt, n=240):
    """Border of the rendered view, mapped back into fisheye pixels."""
    _, (mx,my) = project_view(frame, pan, tilt)
    h,w = mx.shape
    idx = ([(0,c) for c in range(0,w,max(1,w//n))] + [(r,w-1) for r in range(0,h,max(1,h//n))] +
           [(h-1,c) for c in range(w-1,-1,-max(1,w//n))] + [(r,0) for r in range(h-1,-1,-max(1,h//n))])
    pts=[(mx[r,c],my[r,c]) for r,c in idx if mx[r,c]>=0]
    return np.array(pts,dtype=np.int32)

if __name__=='__main__':
    frame_no=int(sys.argv[1]); clip=sys.argv[2]; outp=sys.argv[3]
    cap=cv2.VideoCapture(clip); cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no); ok,img=cap.read(); cap.release()
    assert ok, 'frame read failed'
    PAN_A, PAN_B, TILT = -90.0, 90.0, 45.0
    viewA,_=project_view(img,PAN_A,TILT); viewB,_=project_view(img,PAN_B,TILT)
    disk=img.copy()
    BLUE=(210,120,0); RED=(40,40,210)
    cv2.polylines(disk,[footprint(img,PAN_A,TILT)],True,RED,9)
    cv2.polylines(disk,[footprint(img,PAN_B,TILT)],True,BLUE,9)
    cv2.imwrite(outp+'_disk.png',disk)
    cv2.imwrite(outp+'_a.png',viewA); cv2.imwrite(outp+'_b.png',viewB)
    print('wrote', outp+'_{disk,a,b}.png')


def vcenter(im, H):
    """Pad top AND bottom so a panel is vertically centred in a taller canvas.
    Padding only the bottom top-aligns it, which left the fisheye sitting high."""
    import numpy as np
    if im.shape[0] >= H:
        return im
    total = H - im.shape[0]; top = total // 2
    pad = lambda n: np.full((n, im.shape[1], 3), 255, np.uint8)
    return np.vstack([pad(top), im, pad(total - top)])
