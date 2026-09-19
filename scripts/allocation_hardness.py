"""Dataset screening gate: does a K-view budget actually BIND on this sequence?

A sequence is informative for virtual-PTZ control only if the limited view budget
excludes a non-negligible fraction of simultaneously relevant regions. Detection
difficulty is NOT allocation difficulty: a crowded scene whose people all sit in
one sector is trivial to allocate, and a sparse scene with people spread around
the hemisphere may not be.

Computed from GROUND TRUTH ONLY -- no detector, no policy, no GPU.

  C_static*  best FIXED pair of views, chosen with hindsight over the whole
             sequence.            -> if ~1.0 the budget never binds: reject.
  C_oracle   best pair chosen PER FRAME with hindsight (an upper bound on what
             any causal controller could achieve at this budget).
  H_alloc    C_oracle - C_static*  -> headroom available to adaptive control.
             ~0 means there is nothing for a controller to learn.
  H_spatial  normalised entropy of azimuth occupancy (0 = one hotspot, 1 = uniform)
  V_temporal mean L1 drift of the azimuth histogram between frames
             -> >0 means the hotspot MOVES, which is when adaptation can pay.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import _in_frustum, bearing_to_vec


def wb_vec(az_w, polar_w):
    """world_bearing (polar 0 = nadir) -> the frustum test's vector convention."""
    return bearing_to_vec(az_w + 90.0, 180.0 - polar_w)


def view_grid(n_az=12, tilts=(45.0,)):
    return [(-180.0 + 360.0 * i / n_az, t, 1.0)
            for t in tilts for i in range(n_az)]


def screen(frames, views, base_fov=90.0, out_w=960, out_h=540, n_bins=12):
    """frames: list of per-frame lists of (az, polar) world bearings."""
    M = len(views)
    # cover[v][t] = how many people view v sees at t ; and the member sets
    seen = []          # per frame: list of bool arrays, one per view
    totals = []
    for bearings in frames:
        if not bearings:
            seen.append(None); totals.append(0); continue
        V = np.array([wb_vec(a, p) for a, p in bearings])
        masks = [_in_frustum(V, pan, tilt, zoom, base_fov, out_w, out_h)
                 for (pan, tilt, zoom) in views]
        seen.append(masks); totals.append(len(bearings))
    N = sum(totals)
    if N == 0:
        return None

    # --- best STATIC pair (one choice for the whole sequence) ---
    pair_tot = np.zeros((M, M))
    for masks in seen:
        if masks is None:
            continue
        for i in range(M):
            for j in range(i, M):
                pair_tot[i, j] += int(np.count_nonzero(masks[i] | masks[j]))
    c_static = pair_tot.max() / N
    bi, bj = np.unravel_index(pair_tot.argmax(), pair_tot.shape)

    # --- best pair PER FRAME (hindsight oracle) ---
    orc = 0
    for masks in seen:
        if masks is None:
            continue
        best = 0
        for i in range(M):
            for j in range(i, M):
                c = int(np.count_nonzero(masks[i] | masks[j]))
                if c > best:
                    best = c
        orc += best
    c_oracle = orc / N

    # --- azimuth occupancy entropy and its temporal drift ---
    hists = []
    for bearings in frames:
        h = np.zeros(n_bins)
        for a, _p in bearings:
            h[int(((a + 180.0) % 360.0) / (360.0 / n_bins))] += 1
        if h.sum():
            hists.append(h / h.sum())
    if hists:
        agg = np.mean(hists, axis=0)
        nz = agg[agg > 0]
        h_spatial = float(-(nz * np.log(nz)).sum() / math.log(n_bins))
        v_temporal = float(np.mean([np.abs(b - a).sum()
                                    for a, b in zip(hists, hists[1:])])) if len(hists) > 1 else 0.0
    else:
        h_spatial = v_temporal = float('nan')

    return {'frames': len(frames), 'people_per_frame': N / max(len(frames), 1),
            'C_static': c_static, 'C_oracle': c_oracle,
            'H_alloc': c_oracle - c_static,
            'best_static_pans': (views[bi][0], views[bj][0]),
            'H_spatial': h_spatial, 'V_temporal': v_temporal}


def verdict(r):
    if r is None:
        return 'NO DATA'
    if r['C_static'] > 0.90:
        return 'REJECT: budget never binds (a fixed pair already sees %.0f%%)' % (100 * r['C_static'])
    if r['H_alloc'] < 0.05:
        return 'WEAK: almost no headroom for adaptation (H_alloc=%.3f)' % r['H_alloc']
    return 'KEEP: budget binds, headroom %.3f' % r['H_alloc']


# ------------------------------------------------------------------ loaders
def frida_frames(ann_json, max_frames=None):
    recs = json.load(open(ann_json))
    by = {}
    for r in recs:
        by.setdefault(r['image_id'], []).append(r)
    out = []
    for fid in sorted(by)[:max_frames]:
        fr = []
        for r in by[fid]:
            cx, cy = float(r['bbox'][0]), float(r['bbox'][1])
            R = min(r.get('width', 2048), r.get('height', 2048)) / 2.0
            dx, dy = cx - R, cy - R
            rr = math.hypot(dx, dy)
            fr.append((math.degrees(math.atan2(dy, dx)),
                       math.degrees((rr / max(R, 1e-9)) * (math.pi / 2))))
        out.append(fr)
    return out


def loaf_frames(root, split, seq, max_frames=None):
    """LOAF: human boxes, 2048x2048 disc inscribed. Bearing from box centre,
    matching evaluation.loaf's convention so the two screenings are comparable."""
    from evaluation.loaf import load_annotations
    ann = load_annotations(root, split, sequences=[seq])
    by = ann.get(seq, {})
    out = []
    for fid in sorted(by)[:max_frames]:
        fr = []
        for a_ in by[fid]:
            x, y, w, h = a_['bbox'][:4]
            cx, cy = x + w / 2.0, y + h / 2.0
            R = 1024.0
            dx, dy = cx - R, cy - R
            rr = math.hypot(dx, dy)
            fr.append((math.degrees(math.atan2(dy, dx)),
                       math.degrees((rr / max(R, 1e-9)) * (math.pi / 2))))
        out.append(fr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frida_ann', default=None, help='dir containing Segment_*/Camera_*/data2.json')
    ap.add_argument('--loaf_root', default=None)
    ap.add_argument('--loaf_split', default='val')
    ap.add_argument('--loaf_seqs', default='0062,0055,0071,0051,0052,0054')
    ap.add_argument('--max_frames', type=int, default=300)
    ap.add_argument('--n_az', type=int, default=12)
    a = ap.parse_args()

    views = view_grid(a.n_az)
    print('view grid: %d views at tilt 45, HFOV 90 (K=2 chosen from these)\n' % len(views))
    hdr = '%-16s %6s %8s %9s %9s %8s %9s %9s' % (
        'sequence', 'frames', 'ppl/frm', 'C_static', 'C_oracle', 'H_alloc', 'H_spatial', 'V_temporal')
    print(hdr); print('-' * len(hdr))
    if a.frida_ann:
        for j in sorted(glob.glob(os.path.join(a.frida_ann, 'Segment_*/Camera_*/data2.json'))):
            name = '/'.join(j.split('/')[-3:-1]).replace('Segment_', 'S').replace('/Camera_', 'C')
            r = screen(frida_frames(j, a.max_frames), views)
            if r is None:
                print('%-16s NO DATA' % name); continue
            print('%-16s %6d %8.1f %9.3f %9.3f %8.3f %9.3f %9.3f' % (
                name, r['frames'], r['people_per_frame'], r['C_static'], r['C_oracle'],
                r['H_alloc'], r['H_spatial'], r['V_temporal']))
            print('%-16s   -> %s' % ('', verdict(r)))
    if a.loaf_root:
        for seq in a.loaf_seqs.split(','):
            r = screen(loaf_frames(a.loaf_root, a.loaf_split, seq, a.max_frames), views)
            if r is None:
                print('%-16s NO DATA' % ('LOAF ' + seq)); continue
            print('%-16s %6d %8.1f %9.3f %9.3f %8.3f %9.3f %9.3f' % (
                'LOAF ' + seq, r['frames'], r['people_per_frame'], r['C_static'], r['C_oracle'],
                r['H_alloc'], r['H_spatial'], r['V_temporal']))
            print('%-16s   -> %s' % ('', verdict(r)))


if __name__ == '__main__':
    main()
