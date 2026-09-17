"""LOAF as a second evaluation scene, with HUMAN annotations as the reference.

Motivation. Our own clips come from one room and are scored against a
detector-based pseudo-reference (the same detector run on the raw fisheye), which
cannot credit a person that only the rectified crop finds. LOAF removes both
limitations at once: it is a public, human-annotated, overhead fisheye dataset,
and its val split is a density sweep (3.4 to 26.6 people per frame) recorded by
ceiling cameras at 250-365 cm -- the same geometry as ours.

Key properties (verified, see SUBMISSION.md):
  * 2048x2048, circle inscribed exactly at (1024, 1024) r=1024.
  * Split is by SEQUENCE, so a detector trained on LOAF *train* has not seen val.
  * Annotation fields: bbox, rotated_box, and `world_location` = [ground radius
    (cm), person height (cm)] -- NOT a floor (x, y) pair.
  * No persistent person_ID, so tracks are built here.

Two conventions matter and are deliberate:
  1. Reference BEARINGS come from the annotation's box centre, because that is
     exactly how a policy's detections are converted (`world_bearing` of a box
     centre). Using the annotated ground point instead would compare a foot
     position against a body centre; for an overhead camera those differ by
     ~7 deg, which is a large fraction of the 15 deg matching gate.
  2. IDENTITY association uses a ground position derived from the IMAGE
     (`r = (camera_height - 170 cm) * tan(polar)`, azimuth from the image), in
     centimetres, rather than the dataset's `world_location` radius. Both were
     measured: the image-derived radius gives smaller frame-to-frame
     displacement on 6 of 8 val sequences (e.g. 0052 p90 294 -> 195 cm), because
     `world_location` is noisy in some sequences. That is a
     physical distance, so one gate is meaningful across the whole image,
     unlike an angular gate that tightens toward the rim. The default gate of
     250 cm is a physical bound, not a fit: the frame stride is 15 source frames
     (0.5 s), so 250 cm is a 5 m/s sprint. Measured nearest-neighbour
     displacement has median 67-98 cm across sequences, and a 150 cm gate sits
     below the 90th percentile -- it rejects real matches and fragments tracks.
"""

import json
import math
import os
import glob
from collections import defaultdict


def pixel_to_bearing(px, py, cx, cy, R, fov_deg=180.0):
    """Fisheye pixel -> (azimuth, polar-from-nadir) in degrees, equidistant model.
    Matches ``evaluation.metrics.fisheye_pixel_to_bearing``."""
    dx, dy = px - cx, py - cy
    r = math.hypot(dx, dy)
    theta = (r / max(R, 1e-9)) * (math.radians(fov_deg) / 2.0)
    return (math.degrees(math.atan2(dy, dx)), math.degrees(theta))


class LoafSequence:
    """Frames + human reference for one LOAF sequence."""

    def __init__(self, root, split, seq, fov_deg=180.0,
                 assoc_gate_cm=250.0, move_thresh_cm=25.0, max_age=6, ann=None):
        self.root, self.split, self.seq = root, split, seq
        self.fov_deg = fov_deg
        self.assoc_gate_cm = assoc_gate_cm
        self.move_thresh_cm = move_thresh_cm
        # Frames a track survives unannotated. Matching only against the
        # immediately preceding frame makes a single missed annotation spawn a
        # new identity; at 15-frame stride that fragments most tracks.
        self.max_age = max_age
        self.files = sorted(glob.glob(os.path.join(root, 'images', split, f'{seq}_*.jpg')))
        if not self.files:
            raise FileNotFoundError(f'no frames for sequence {seq} in {split}')
        self.cx = self.cy = self.R = None          # set on first frame
        self._ann = ann if ann is not None else load_annotations(root, split)
        self._by_frame = self._ann.get(seq, {})

    # ---------------------------------------------------------------- frames
    def __len__(self):
        return len(self.files)

    def frame_key(self, i):
        return os.path.basename(self.files[i])

    def read(self, i):
        import cv2
        img = cv2.imread(self.files[i])
        if img is not None and self.R is None:
            h, w = img.shape[:2]
            # LOAF val: the 180 deg circle is inscribed in the square frame.
            self.cx, self.cy, self.R = w / 2.0, h / 2.0, min(w, h) / 2.0
        return img

    NOMINAL_PERSON_CM = 170.0

    def _ground(self, bearing, ann):
        """Ground position (cm) from image geometry: a person's floor point sits
        at radius (camera_height - person_height) * tan(polar)."""
        phi = math.radians(bearing[0])
        pol = math.radians(min(bearing[1], 88.0))       # cap: tan explodes at the rim
        camh = float(ann.get('camera_height') or 300.0)
        r = max(camh - self.NOMINAL_PERSON_CM, 1.0) * math.tan(pol)
        return (r * math.cos(phi), r * math.sin(phi))

    def quality(self, max_frames=80):
        """90th-percentile nearest-neighbour ground displacement (cm) between
        consecutive annotated frames. The stride is 0.5 s, so anything much above
        ~250 cm (5 m/s) means the annotations cannot be tracked reliably and the
        sequence should not carry identity-based metrics."""
        if self.R is None:
            self.read(0)
        prev, d = None, []
        for t in range(min(max_frames, len(self.files))):
            cur = [self._ground(pixel_to_bearing(a['bbox'][0] + a['bbox'][2] / 2.0,
                                                 a['bbox'][1] + a['bbox'][3] / 2.0,
                                                 self.cx, self.cy, self.R, self.fov_deg), a)
                   for a in self._by_frame.get(self.frame_key(t), [])
                   if not (a.get('ignore') or a.get('iscrowd'))]
            if prev:
                for c in cur:
                    d.append(min(math.dist(c, p) for p in prev))
            prev = cur
        if not d:
            return float('nan')
        d.sort()
        return d[int(0.9 * (len(d) - 1))]

    # ------------------------------------------------------------- reference
    def reference(self, max_frames=None):
        """Human-annotated reference in the shape the metrics expect:
        ``gt[t] = [{'id', 'bearing': (az, polar), 'moving': bool}, ...]``."""
        if self.R is None:
            self.read(0)
        n = len(self.files) if max_frames is None else min(max_frames, len(self.files))
        pool, next_id = {}, 1        # id -> {'ground', 'last_t'}
        out = []
        for t in range(n):
            anns = self._by_frame.get(self.frame_key(t), [])
            cur = []
            for a in anns:
                if a.get('ignore') or a.get('iscrowd'):
                    continue
                x, y, w, h = a['bbox']
                bear = pixel_to_bearing(x + w / 2.0, y + h / 2.0,
                                        self.cx, self.cy, self.R, self.fov_deg)
                cur.append({'bearing': bear, 'ground': self._ground(bear, a)})
            # retire tracks unseen for longer than max_age, then match against the
            # whole live pool (not just the previous frame) best-pair-first
            pool = {k: v for k, v in pool.items() if t - v['last_t'] <= self.max_age}
            pairs = []
            for i, c in enumerate(cur):
                for tid, p in pool.items():
                    d = math.dist(c['ground'], p['ground'])
                    # allow a proportionally larger jump after a gap
                    gate = self.assoc_gate_cm * max(1, t - p['last_t'])
                    if d <= gate:
                        pairs.append((d, i, tid))
            pairs.sort()
            used_c, used_t = set(), set()
            for d, i, tid in pairs:
                if i in used_c or tid in used_t:
                    continue
                gap = max(1, t - pool[tid]['last_t'])
                cur[i]['id'] = tid
                cur[i]['moving'] = (d / gap) > self.move_thresh_cm
                used_c.add(i); used_t.add(tid)
            for i, c in enumerate(cur):
                if 'id' not in c:
                    c['id'] = next_id; next_id += 1
                    c['moving'] = False
                pool[c['id']] = {'ground': c['ground'], 'last_t': t}
            out.append([{'id': c['id'], 'bearing': c['bearing'], 'moving': c['moving']}
                        for c in cur])
        self.n_tracks = next_id - 1
        return out


# Only the fields the reference actually needs. The raw annotations carry
# `segmentation` polygons, which dominate memory (the val JSON is ~60 MB on disk
# and far larger once parsed); keeping everything for all sequences has been
# enough to get the process OOM-killed on a busy host.
_KEEP = ('bbox', 'ignore', 'iscrowd', 'camera_height', 'world_location')


def load_annotations(root, split, sequences=None):
    """sequence -> {frame filename -> [annotation, ...]}.

    ``sequences`` restricts the result to those sequence ids, which matters:
    loading all of them keeps ~8x more than a single-sequence run needs. Callers
    evaluating several sequences in one process should still load once and share.
    """
    keep = set(sequences) if sequences else None
    p = os.path.join(root, 'annotations', 'resolution_2k', f'instances_{split}.json')
    d = json.load(open(p))
    img = {i['id']: i['file_name'] for i in d['images']}
    out = defaultdict(lambda: defaultdict(list))
    for a in d['annotations']:
        fn = img[a['image_id']]
        seq = fn.split('_')[0]
        if keep is not None and seq not in keep:
            continue
        out[seq][fn].append({k: a[k] for k in _KEEP if k in a})
    del d
    return {k: dict(v) for k, v in out.items()}


def sequences(root, split):
    return sorted({os.path.basename(f).split('_')[0]
                   for f in glob.glob(os.path.join(root, 'images', split, '*.jpg'))})


# ======================================================================
# Self-test: no GPU, no detector. Run as
#   python evaluation/loaf.py --root /path/to/loaf [--split val]
# ======================================================================

def _selftest(root, split='val'):
    import statistics as st
    from collections import Counter
    ok = True
    ann = load_annotations(root, split)
    seqs = sequences(root, split)
    print(f'{len(seqs)} sequences in {split}: {seqs}')
    for seq in seqs:
        s = LoafSequence(root, split, seq, ann=ann)
        q = s.quality()
        gt = s.reference(max_frames=120)
        per = [len(f) for f in gt]
        life = Counter()
        for f in gt:
            for p in f:
                life[p['id']] += 1
        L = list(life.values())
        pol = [p['bearing'][1] for f in gt for p in f]
        print(f'  {seq}: {len(gt)} frames | {st.mean(per):.1f} people/frame (max {max(per)}) | '
              f'{len(L)} tracks, lifetime median {st.median(L):.0f}, '
              f'{100*sum(1 for x in L if x == 1)/len(L):.0f}% singletons | '
              f'polar {min(pol):.1f}-{max(pol):.1f} deg | p90 disp {q:.0f} cm'
              + ('  <-- UNTRACKABLE' if q > 400 else ''))
        if max(pol) > 90.5:
            print('    [FAIL] polar angle beyond the horizon'); ok = False
        if st.median(L) < 4 and q <= 400:
            print('    [FAIL] track lifetimes too short -- association is fragmenting'); ok = False
        sing = sum(1 for x in L if x == 1) / len(L)
        if sing > 0.35:
            # not a failure by itself: a corridor with high turnover legitimately
            # produces people who appear in one frame and leave.
            print(f'    [warn] {100*sing:.0f}% singleton tracks -- check turnover vs association')
    print('SELFTEST', 'PASSED' if ok else 'FAILED')
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--split', default='val')
    a = ap.parse_args()
    raise SystemExit(0 if _selftest(a.root, a.split) else 1)


def attach_multi(env, root, split, seq_ids, frame_skip=1, seed=0):
    """Draw episodes from LOAF sequences instead of video files.

    Training needs only pixels -- rewards come from the environment's own
    detector, not from annotations -- so sequences are built without them. Each
    reset picks one sequence at random, mirroring how the video environment
    picks a clip per episode. ``frame_skip`` should be 1: LOAF frames are already
    15 source frames apart, so one env step must consume one LOAF frame to match
    the 0.5 s control interval used at evaluation time.
    """
    import numpy as np

    host = getattr(env, '_host', env)
    seqs = [LoafSequence(root, split, s, ann={}) for s in seq_ids]
    for s in seqs:
        s.read(0)
    rng = np.random.default_rng(seed)
    state = {'seq': seqs[0]}

    host.frame_skip = frame_skip
    host.video_paths = [f'loaf:{s.seq}' for s in seqs]

    def _get_frame():
        s = state['seq']
        idx = int(host.current_frame // max(frame_skip, 1))
        if idx >= len(s):
            host._clip_ended = True
            idx = len(s) - 1
        return s.read(idx)

    host._get_frame = _get_frame

    base_reset = env.reset

    def reset(*a, **kw):
        s = seqs[int(rng.integers(0, len(seqs)))]
        state['seq'] = s
        out = base_reset(*a, **kw)
        # base reset re-opens a video file and resets the capture bookkeeping;
        # re-assert the LOAF source and geometry afterwards.
        host.fish_cx, host.fish_cy, host.fish_R = s.cx, s.cy, s.R
        host.fish_fov_deg = 180.0
        host.frame_count = len(s) * max(frame_skip, 1)
        host.video_path = f'loaf:{s.seq}'
        host.frame_skip = frame_skip
        host._get_frame = _get_frame
        return out

    env.reset = reset
    return seqs
