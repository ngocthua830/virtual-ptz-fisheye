"""Known-answer checks for the FRIDA loader, before any GPU time is spent.

Validates the three things that have burned this project before: the bearing
convention, the box-centre convention, and whether identities are really GIVEN
rather than silently re-derived per frame.
"""
import sys
import math

sys.path.insert(0, '/app')
from evaluation.frida import FridaSequence, pixel_to_bearing

ROOT = '/app/data/frida/FRIDA'
ok = True

# --- 1. bearing model: nadir at the disc centre, horizon at the rim ---
c = pixel_to_bearing(1024, 1024, 1024, 1024, 1024)
e = pixel_to_bearing(2048, 1024, 1024, 1024, 1024)
m = pixel_to_bearing(1536, 1024, 1024, 1024, 1024)
print('centre  -> polar %.2f  (expect 0)' % c[1]);  ok &= abs(c[1]) < 1e-6
print('rim     -> polar %.2f  (expect 90)' % e[1]); ok &= abs(e[1] - 90) < 1e-6
print('mid     -> polar %.2f  (expect 45)' % m[1]); ok &= abs(m[1] - 45) < 1e-6
print('rim az  -> %.2f  (expect 0, +x)' % e[0]);    ok &= abs(e[0]) < 1e-6

seq = FridaSequence(ROOT, 1, 1)
print('\nsequence %s: %d frames on disk, %d annotated identities'
      % (seq.seq, len(seq), seq.n_tracks))
print('disc: centre=(%.0f,%.0f) R=%.0f' % (seq.cx, seq.cy, seq.R))

ref = seq.reference(max_frames=300)
ppl = [len(f) for f in ref]
print('reference: %d frames, %.1f people/frame (min %d max %d)'
      % (len(ref), sum(ppl) / max(len(ppl), 1), min(ppl), max(ppl)))

# --- 2. identities are GIVEN and persistent, not re-derived per frame ---
ids = [p['id'] for f in ref for p in f]
uniq = set(ids)
print('distinct ids in window: %d ; total boxes %d' % (len(uniq), len(ids)))
ok &= len(uniq) < len(ids) / 10          # ids must recur, not be one-per-box
first = {p['id'] for p in ref[0]}
persist = sum(1 for f in ref[1:20] if first & {p['id'] for p in f})
print('ids from frame 0 reappear in %d of the next 19 frames' % persist)
ok &= persist >= 15

# --- 3. all bearings physical ---
bad = [p for f in ref for p in f if not (0 <= p['bearing'][1] <= 90.001)]
print('bearings outside polar 0..90: %d' % len(bad))
ok &= not bad

# --- 4. movers exist but are not everything ---
mv = sum(p['moving'] for f in ref for p in f)
print('moving flags: %d / %d (%.0f%%)' % (mv, len(ids), 100 * mv / max(len(ids), 1)))
ok &= 0 < mv < len(ids)

print('\nFRIDA SELFTEST', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
