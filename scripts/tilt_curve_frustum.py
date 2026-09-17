"""Corrected tilt curve for the overlap-optimized sweep.

The sweep's actions depend only on env.tilt, and reset() is deterministic
(pan [0,180], tilt [30,30], zoom [1,1]), so its trajectory is a pure function of
target_tilt -- no video, no detector. We replay the env kinematics exactly and
score overlap with BOTH the circular-cone approximation used in the submitted
version and the exact rendered-frustum test.
"""
import sys
sys.path.insert(0, '.')
from evaluation.metrics import (inter_camera_overlap, solid_angle_overlap,
                                inter_camera_overlap_frustum, solid_angle_overlap_frustum)
import numpy as np, math

PAN_SPEED, TILT_SPEED = 25.0, 15.0
TILT_RANGE = (0.0, 80.0)
STEPS = 300
STAY, PAN_L, PAN_R, ZI, ZO, TU, TD = range(7)

def wrap(p):
    if p > 180.0: return p - 360.0
    if p < -180.0: return p + 360.0
    return p

def replay(target_tilt, steps=STEPS, pan0=(0.0, 180.0)):
    pan = list(pan0); tilt = [30.0, 30.0]; zoom = [1.0, 1.0]
    poses = []
    for _ in range(steps):
        poses.append([(pan[i], tilt[i], zoom[i]) for i in (0, 1)])
        for i in (0, 1):                      # OverlapOptimizedSweepPolicy.act
            if abs(tilt[i] - target_tilt) > 3.0:
                prm = min(1.0, abs(tilt[i] - target_tilt) / TILT_SPEED)
                if tilt[i] < target_tilt:
                    tilt[i] = min(TILT_RANGE[1], tilt[i] + TILT_SPEED * prm)
                else:
                    tilt[i] = max(TILT_RANGE[0], tilt[i] - TILT_SPEED * prm)
            else:
                pan[i] = wrap(pan[i] + PAN_SPEED * 0.4)
    return poses

def bearings(poses):
    """(az, polar) axis bearing; for this rotation convention az=pan, polar=tilt."""
    return [[(p, t) for (p, t, _z) in step] for step in poses]

print("Ovlp-Sweep, replayed kinematics, 300 steps -- cone vs exact frustum\n")
print(f"  {'target tilt':>11} {'axis sep':>9} | {'cone bin':>9} {'cone IoU':>9} | "
      f"{'frust bin':>10} {'frust IoU':>10}")
for tt in (30, 35, 40, 44, 46, 48, 50, 55, 60, 65, 70):
    poses = replay(float(tt))
    zooms = [[z for (_p, _t, z) in s] for s in poses]
    cb = inter_camera_overlap(bearings(poses), zooms, 90.0)
    ci = solid_angle_overlap(bearings(poses), zooms, 90.0)
    fb = inter_camera_overlap_frustum(poses, 90.0, 960, 540)
    fi = solid_angle_overlap_frustum(poses, 90.0, 960, 540)
    print(f"  {tt:>11} {2*tt:>8}deg | {cb:>9.3f} {ci:>9.4f} | {fb:>10.3f} {fi:>10.4f}")

print("\nstart-state check (reviewer: cameras always start exactly 180 deg apart)")
for off in (180.0, 150.0, 120.0, 90.0, 45.0, 0.0):
    poses = replay(48.0, pan0=(0.0, off))
    fb = inter_camera_overlap_frustum(poses, 90.0, 960, 540)
    fi = solid_angle_overlap_frustum(poses, 90.0, 960, 540)
    print(f"  initial azimuth offset {off:>5.0f}deg -> frustum binary {fb:.3f}  IoU {fi:.4f}")
