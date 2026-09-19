"""FRIDA loader: overhead-fisheye frames with GROUND-TRUTH persistent person IDs.

Why this dataset. Our LOAF evaluation reconstructs identity tracks by greedy
nearest-neighbour association, because LOAF annotates each frame independently.
Every identity-based metric there (discovery, time-to-detect, max unobserved
gap) therefore inherits the association gate. FRIDA removes that dependency:
``person_id`` is annotated and consistent across frames, so the reference tracks
are GIVEN rather than inferred. It is the control for our own tracking step.

Two further properties make it a drop-in:
  * frames are 2048x2048 with the fisheye disc inscribed at (1024,1024) r=1024 --
    verified on the imagery, and identical to the LOAF assumption, so no new
    camera model is needed;
  * the detector was pre-trained on CEPDOF and LOAF only, never on FRIDA, so this
    is also an uncontaminated cross-dataset test.

Layout:  FRIDA/Frames/Segment_<S>/Camera_<C>/<image_id>.jpg
         FRIDA/Annotations/Segment_<S>/Camera_<C>/data2.json

`data2.json` is a flat list of records:
    {"bbox": [cx, cy, w, h, angle_deg], "image_id": "0001155",
     "person_id": 1, "width": 2048, "height": 2048, ...}
bbox[0:2] is the box CENTRE (verified by cropping: the centre convention frames
people, the top-left convention does not).

Reference bearings come from the box centre, matching how a policy's detections
are converted (`world_bearing` of a box centre) -- the same convention as
`evaluation.loaf`, so the two datasets stay comparable.
"""
import json
import math
import os
import glob

import cv2

FISH_FOV_DEG = 180.0


def pixel_to_bearing(px, py, cx, cy, R, fov_deg=FISH_FOV_DEG):
    """Fisheye pixel -> (azimuth, polar-from-nadir) degrees, equidistant model.
    Identical to evaluation.loaf.pixel_to_bearing and metrics.fisheye_pixel_to_bearing."""
    dx, dy = px - cx, py - cy
    r = math.hypot(dx, dy)
    theta = (r / max(R, 1e-9)) * (math.radians(fov_deg) / 2.0)
    return (math.degrees(math.atan2(dy, dx)), math.degrees(theta))


class FridaSequence:
    """One (segment, camera) of FRIDA, with ground-truth identity tracks."""

    def __init__(self, root, segment, camera, move_thresh_px=6.0):
        self.root, self.segment, self.camera = root, segment, camera
        self.seq = f'S{segment}C{camera}'
        self.move_thresh_px = float(move_thresh_px)
        fdir = os.path.join(root, 'Frames', f'Segment_{segment}', f'Camera_{camera}')
        self.files = sorted(glob.glob(os.path.join(fdir, '*.jpg')))
        if not self.files:
            raise FileNotFoundError(f'no frames under {fdir}')
        adir = os.path.join(root, 'Annotations', f'Segment_{segment}',
                            f'Camera_{camera}', 'data2.json')
        self._recs = json.load(open(adir))
        # frames present BOTH on disk and in the annotations, in temporal order
        have = {os.path.splitext(os.path.basename(f))[0]: f for f in self.files}
        ids = sorted({r['image_id'] for r in self._recs} & set(have))
        self.frame_ids = ids
        self.files = [have[i] for i in ids]
        w = self._recs[0].get('width', 2048)
        h = self._recs[0].get('height', 2048)
        if (w, h) != (2048, 2048):
            raise ValueError(f'FRIDA expected 2048x2048, got {w}x{h}')
        # disc inscribed -- verified on the imagery, same as LOAF
        self.cx, self.cy, self.R = w / 2.0, h / 2.0, min(w, h) / 2.0
        self.n_tracks = len({r['person_id'] for r in self._recs})

    def __len__(self):
        return len(self.files)

    def read(self, idx):
        return cv2.imread(self.files[min(idx, len(self.files) - 1)])

    def reference(self, max_frames=None):
        """Per-frame [{'id', 'bearing', 'moving'}] using the ANNOTATED person_id.

        No association step: this is the whole point of using FRIDA.
        `moving` compares the box centre with that identity's previous frame.
        """
        by_frame = {}
        for r in self._recs:
            by_frame.setdefault(r['image_id'], []).append(r)
        ids = self.frame_ids[:max_frames] if max_frames else self.frame_ids
        prev = {}
        out = []
        for fid in ids:
            frame = []
            for r in by_frame.get(fid, []):
                cx_, cy_ = float(r['bbox'][0]), float(r['bbox'][1])
                pid = r['person_id']
                p = prev.get(pid)
                moving = bool(p is not None and math.dist((cx_, cy_), p) > self.move_thresh_px)
                prev[pid] = (cx_, cy_)
                frame.append({'id': pid,
                              'bearing': pixel_to_bearing(cx_, cy_, self.cx, self.cy, self.R),
                              'moving': moving})
            out.append(frame)
        return out


def attach(host, seq, frame_skip):
    """Point the environment at a FRIDA sequence (mirrors evaluation.loaf.attach)."""
    host.fish_cx, host.fish_cy, host.fish_R = seq.cx, seq.cy, seq.R
    host.fish_fov_deg = FISH_FOV_DEG
    host.frame_count = len(seq) * frame_skip
    host.video_path = f'frida:{seq.seq}'

    def _get_frame():
        idx = int(host.current_frame // frame_skip)
        if idx >= len(seq):
            host._clip_ended = True
            idx = len(seq) - 1
        return seq.read(idx)

    host._get_frame = _get_frame
