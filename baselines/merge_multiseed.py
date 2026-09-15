"""Merge per-seed shard JSONs from evaluate_multiseed.py (one seed per GPU) into
one aggregate: mean +/- std across seeds + paired RL-vs-baseline t-tests.

    python baselines/merge_multiseed.py baselines/results/multiseed_seed*.json \
        --out baselines/results/multiseed_290526.json
"""
import argparse, glob, json, os
import numpy as np
try:
    from scipy import stats as sst
except Exception:
    sst = None

ap = argparse.ArgumentParser()
ap.add_argument('shards', nargs='+')
ap.add_argument('--out', default=None)
args = ap.parse_args()

paths = []
for pat in args.shards:
    paths.extend(sorted(glob.glob(pat)))
paths = sorted(set(paths))

# policy -> {seed: seed_mean_total}, and tracker/explorer seed means
per_seed = {}       # policy -> list of (seed, total_mean)
per_seed_t = {}
per_seed_e = {}
all_eps = {}
clips = None
cfg = None
for p in paths:
    with open(p) as f:
        d = json.load(f)
    clips = d.get('clips', clips)
    cfg = d.get('config', cfg)
    summ = d['summary']
    for pol, s in summ.items():
        if pol.startswith('_'):
            continue
        seeds = s['per_seed_totals']  # one entry here (single-seed shard)
        # recover the seed id from the shard config if present
        cfg_seeds = (d.get('config') or {}).get('seed_list')
        sid = cfg_seeds if cfg_seeds is not None else p
        per_seed.setdefault(pol, []).append((sid, float(np.mean(seeds))))
        per_seed_t.setdefault(pol, []).append(float(s['tracker_mean']))
        per_seed_e.setdefault(pol, []).append(float(s['explorer_mean']))
        all_eps.setdefault(pol, []).extend(s.get('all_episode_totals', []))

policies = list(per_seed.keys())
summary = {}
for pol in policies:
    vals = np.array([v for _, v in per_seed[pol]])
    summary[pol] = {
        'n_seeds': int(vals.size),
        'total_mean': float(vals.mean()),
        'total_std': float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        'total_sem': float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0,
        'tracker_mean': float(np.mean(per_seed_t[pol])),
        'explorer_mean': float(np.mean(per_seed_e[pol])),
        'per_seed_totals': [float(v) for v in vals],
        'n_episodes_total': len(all_eps.get(pol, [])),
    }

print('=' * 82)
print(f'MERGED MULTI-SEED | clips={clips} | shards={len(paths)} | '
      f'seeds/policy={summary[policies[0]]["n_seeds"]}')
print('=' * 82)
print(f'{"policy":10s} {"tracker":>10s} {"explorer":>10s} {"TOTAL mean":>12s} {"std":>9s} {"sem":>9s}')
print('-' * 82)
for pol in sorted(policies, key=lambda k: summary[k]['total_mean']):
    s = summary[pol]
    star = '  <-- RL' if pol == 'rl' else ''
    print(f'{pol:10s} {s["tracker_mean"]:>+10.1f} {s["explorer_mean"]:>+10.1f} '
          f'{s["total_mean"]:>+12.1f} {s["total_std"]:>9.1f} {s["total_sem"]:>9.1f}{star}')
print('=' * 82)

comparisons = {}
if 'rl' in summary:
    # align seeds for paired test
    rl_by_seed = dict(per_seed['rl'])
    rl_mean = summary['rl']['total_mean']
    print(f'\nPaired RL vs baseline (n={summary["rl"]["n_seeds"]} seeds):')
    for pol in policies:
        if pol == 'rl':
            continue
        base_by_seed = dict(per_seed[pol])
        common = [k for k in rl_by_seed if k in base_by_seed]
        rl_v = np.array([rl_by_seed[k] for k in common])
        base_v = np.array([base_by_seed[k] for k in common])
        diff = rl_v - base_v
        base_mean = summary[pol]['total_mean']
        pct = (rl_mean - base_mean) / abs(base_mean) * 100 if base_mean else float('nan')
        c = {'pct': float(pct), 'mean_diff': float(diff.mean()),
             'diff_std': float(diff.std(ddof=1)) if diff.size > 1 else 0.0,
             'n': int(diff.size)}
        line = (f'  RL vs {pol:8s}: {pct:+.0f}%  (RL {rl_mean:+.0f} vs {base_mean:+.0f})  '
                f'mean_diff={diff.mean():+.1f} +/- {c["diff_std"]:.1f}')
        if sst is not None and diff.size > 1:
            t, pv = sst.ttest_rel(rl_v, base_v)
            c['t_stat'] = float(t); c['p_value'] = float(pv)
            line += f'  paired t={t:+.2f} p={pv:.4f}' + (' *' if pv < 0.05 else '')
        comparisons[pol] = c
        print(line)

summary['_comparisons'] = comparisons
if args.out:
    with open(args.out, 'w') as f:
        json.dump({'merged_from': paths, 'clips': clips, 'summary': summary}, f, indent=2)
    print(f'\nSaved {args.out}')
