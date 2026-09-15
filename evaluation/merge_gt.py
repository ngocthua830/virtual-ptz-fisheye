"""Merge per-seed ground-truth-metric shards (one seed per GPU) into a cross-seed
table: mean +/- std over seeds, plus paired RL-vs-baseline tests where useful.

Each shard was produced by metrics.py with --seed_list <s>, so its per-policy
per-metric ``mean`` is that seed's mean-over-episodes — i.e. one seed-sample. We
aggregate those across seeds (unit of analysis = seed).

    python evaluation/merge_gt.py evaluation/results/gt_seed*.json \
        --out evaluation/results/gt_290526.json
"""
import argparse, glob, json
import numpy as np
try:
    from scipy import stats as sst
except Exception:
    sst = None

METRICS = ['discovery_rate', 'time_to_detect_steps', 'coverage_pct',
           'inter_camera_overlap', 'dwell_ratio_moving', 'time_to_detect_mover_steps']
HIGHER_BETTER = {'discovery_rate': True, 'time_to_detect_steps': False,
                 'coverage_pct': True, 'inter_camera_overlap': False,
                 'dwell_ratio_moving': True, 'time_to_detect_mover_steps': False}

ap = argparse.ArgumentParser()
ap.add_argument('shards', nargs='+')
ap.add_argument('--out', default=None)
args = ap.parse_args()

paths = []
for pat in args.shards:
    paths.extend(sorted(glob.glob(pat)))
paths = sorted(set(paths))

# policy -> metric -> {seed_key: value}
per_seed = {}
for p in paths:
    with open(p) as f:
        d = json.load(f)
    seed_key = (d.get('config') or {}).get('seed_list', p)
    for pol, mets in d['summary'].items():
        for mk in METRICS:
            v = mets.get(mk, {}).get('mean')
            if v is None:
                continue
            per_seed.setdefault(pol, {}).setdefault(mk, {})[seed_key] = float(v)

policies = list(per_seed.keys())
summary = {}
for pol in policies:
    summary[pol] = {}
    for mk in METRICS:
        vals = np.array([v for v in per_seed[pol].get(mk, {}).values() if v == v], dtype=float)
        if vals.size == 0:
            summary[pol][mk] = {'mean': float('nan'), 'std': float('nan'), 'n': 0}
        else:
            summary[pol][mk] = {
                'mean': float(vals.mean()),
                'std': float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                'sem': float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0,
                'n': int(vals.size),
            }

# ---- print table ----
hdr = ['discovery', 'ttd_stp', 'coverage', 'overlap', 'dwell_mov', 'ttd_mover']
print('=' * 100)
print('GROUND-TRUTH METRICS — merged across seeds (mean +/- std)')
print('=' * 100)
print(f'{"policy":9s} ' + ' '.join(f'{h:>14s}' for h in hdr))
print('-' * 100)
order = sorted(policies, key=lambda k: -summary[k]['discovery_rate']['mean'])
for pol in order:
    cells = []
    for mk in METRICS:
        s = summary[pol][mk]
        cells.append(f'{s["mean"]:>6.2f}±{s["std"]:<5.2f}' if s['mean'] == s['mean'] else f'{"nan":>12s}')
    star = ' <-RL' if pol == 'rl' else ''
    print(f'{pol:9s} ' + ' '.join(f'{c:>14s}' for c in cells) + star)
print('=' * 100)
print('cols: discovery_rate, time_to_detect(steps), coverage_pct, inter_camera_overlap, '
      'dwell_ratio_moving, time_to_detect_mover(steps)')
print('higher-is-better: discovery, coverage, dwell_mov | lower-is-better: ttd, overlap, ttd_mover')

# ---- paired RL vs baselines on the headline metric (discovery_rate) ----
comparisons = {}
if 'rl' in per_seed:
    print('\nPaired RL vs baseline (discovery_rate, unit=seed):')
    rl_map = per_seed['rl']['discovery_rate']
    for pol in policies:
        if pol == 'rl':
            continue
        base_map = per_seed[pol].get('discovery_rate', {})
        common = [k for k in rl_map if k in base_map]
        rl_v = np.array([rl_map[k] for k in common])
        bv = np.array([base_map[k] for k in common])
        if rl_v.size == 0:
            continue
        diff = rl_v - bv
        c = {'metric': 'discovery_rate', 'mean_diff': float(diff.mean()),
             'diff_std': float(diff.std(ddof=1)) if diff.size > 1 else 0.0, 'n': int(diff.size)}
        line = (f'  RL vs {pol:8s}: disc {summary["rl"]["discovery_rate"]["mean"]:.3f} vs '
                f'{summary[pol]["discovery_rate"]["mean"]:.3f}  '
                f'mean_diff={diff.mean():+.3f} +/- {c["diff_std"]:.3f}')
        if sst is not None and diff.size > 1 and diff.std() > 0:
            t, pv = sst.ttest_rel(rl_v, bv)
            c['t_stat'] = float(t); c['p_value'] = float(pv)
            line += f'  paired t={t:+.2f} p={pv:.4f}' + (' *' if pv < 0.05 else '')
        comparisons[pol] = c
        print(line)

summary['_comparisons'] = comparisons
if args.out:
    with open(args.out, 'w') as f:
        json.dump({'merged_from': paths, 'summary': summary}, f, indent=2)
    print(f'\nSaved {args.out}')
