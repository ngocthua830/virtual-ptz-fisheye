"""K=1 vs K=2: does the value of learned control rise as the view budget shrinks?

Prints the two arms side by side and the quantity the phase-diagram claim rests on,
  Delta(K) = discovery(geometric sweep at K) - discovery(learned at K),
plus per-seed consistency. A SMALLER Delta at K=1 than at K=2 is the positive signal.
"""
import glob, json, statistics, sys

def load(pat, pol):
    out = []
    for f in sorted(glob.glob(pat)):
        try:
            s = json.load(open(f))['summary'][pol]
        except (KeyError, json.JSONDecodeError):
            continue
        out.append((f.split('/')[-1], s))
    return out

def m(s, k):
    return s[k]['mean']

# ---------------- K=2 (already in the paper) ----------------
k2_rl = load('results/full128_bearing/eval/full_s4*.json', 'rl')
k2_base = load('results/full128_bearing/eval/baselines.json', 'ovsweep')
k2_rl_d = [m(s, 'discovery_rate') for _, s in k2_rl]
k2_sw_d = m(k2_base[0][1], 'discovery_rate') if k2_base else 0.92

# ---------------- K=1 (new) ----------------
k1_solo = load('results/solo_k1/eval/solo_s4*.json', 'solo')
k1_sw = load('results/solo_k1/eval/baselines_k1.json', 'ovsweep')
k1_rnd = load('results/solo_k1/eval/baselines_k1.json', 'random')
if not k1_solo or not k1_sw:
    print('K=1 results not ready yet'); sys.exit(0)
k1_solo_d = [m(s, 'discovery_rate') for _, s in k1_solo]
k1_sw_d = m(k1_sw[0][1], 'discovery_rate')
k1_rnd_d = m(k1_rnd[0][1], 'discovery_rate') if k1_rnd else float('nan')

def block(lbl, sw, learned, rnd=None):
    mu, sd = statistics.mean(learned), statistics.pstdev(learned)
    wins = sum(1 for v in learned if v > sw)
    print(f'  {lbl}: sweep {sw:.3f} | learned {mu:.3f}+-{sd:.3f} (n={len(learned)})'
          f' | delta {sw-mu:+.3f} | learned beats sweep on {wins}/{len(learned)} seeds'
          + (f' | random {rnd:.3f}' if rnd is not None else ''))
    return sw - mu

print('DISCOVERY, our room, full clip (300-step budget)')
d2 = block('K=2', k2_sw_d, k2_rl_d)
d1 = block('K=1', k1_sw_d, k1_solo_d, k1_rnd_d)
print()
print(f'Delta(K=2) = {d2:+.3f}   Delta(K=1) = {d1:+.3f}   change = {d1-d2:+.3f}')
if d1 < d2:
    print('=> the geometric lead SHRINKS at the tighter budget: learning gains as K falls.')
elif d1 > d2:
    print('=> the geometric lead GROWS at the tighter budget: learning loses further as K falls.')
else:
    print('=> unchanged.')
print()
print('per-seed K=1 solo:', [round(v, 3) for v in sorted(k1_solo_d, reverse=True)])
print('per-seed K=2 rl  :', [round(v, 3) for v in sorted(k2_rl_d, reverse=True)])
for lbl, rows in (('K=1 solo', k1_solo), ('K=1 sweep', k1_sw)):
    for f, s in rows:
        print(f'  {lbl:10s} {f:22s} disc={m(s,"discovery_rate"):.4f} '
              f'ttd={m(s,"time_to_detect_steps"):.2f}')
