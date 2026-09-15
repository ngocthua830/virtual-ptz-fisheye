"""Aggregate the ablation / multi-seed evaluation into an honest results table.

Unit of analysis = INDEPENDENT TRAINING SEED (the reviewer's statistical fix):
each ``<config>_s<seed>.json`` holds one trained model's RL score (mean over the
fixed eval episodes). We aggregate across seeds:

  * full model:      5 seeds  -> mean +/- std (n=5)
  * each ablation:   3 seeds  -> mean +/- std (n=3)
  * baselines:       from baselines.json (no training)

Because n is small (the reviewer's point), we lead with EFFECT SIZES and
consistency (how many seeds beat the baseline), and report t-tests only as a
secondary signal with the small-n caveat.

    python evaluation/merge_ablation.py results/ablation/eval --out results/ablation/ablation_summary.json
"""
import argparse, glob, json, os, re
import numpy as np
try:
    from scipy import stats as sst
except Exception:
    sst = None

METRICS = ['discovery_rate', 'time_to_detect_steps', 'coverage_pct',
           'inter_camera_overlap', 'dwell_ratio_moving', 'time_to_detect_mover_steps']
SHORT = {'discovery_rate': 'discovery', 'time_to_detect_steps': 'ttd',
         'coverage_pct': 'coverage', 'inter_camera_overlap': 'overlap',
         'dwell_ratio_moving': 'dwell_mov', 'time_to_detect_mover_steps': 'ttd_mover'}
HIGHER_BETTER = {'discovery_rate': True, 'time_to_detect_steps': False,
                 'coverage_pct': True, 'inter_camera_overlap': False,
                 'dwell_ratio_moving': True, 'time_to_detect_mover_steps': False}
CONFIGS = ['full', 'nocov', 'nonov', 'noovlp', 'unif']
CONFIG_LABEL = {
    'full':   'Full model',
    'nocov':  '- shared coverage map',
    'nonov':  '- novelty reward',
    'noovlp': '- anti-overlap penalty',
    'unif':   'uniform (no peripheral) wt',
}

ap = argparse.ArgumentParser()
ap.add_argument('eval_dir')
ap.add_argument('--out', default=None)
args = ap.parse_args()


def rl_scores(pattern):
    """Return {metric: [per-seed values]} for all run JSONs matching pattern."""
    out = {m: [] for m in METRICS}
    for p in sorted(glob.glob(os.path.join(args.eval_dir, pattern))):
        with open(p) as f:
            d = json.load(f)
        s = (d.get('summary') or {}).get('rl')
        if not s:
            continue
        for m in METRICS:
            v = s.get(m, {}).get('mean')
            if v is not None and v == v:
                out[m].append(float(v))
    return out


# ---- baselines ----
baselines = {}
bpath = os.path.join(args.eval_dir, 'baselines.json')
if os.path.exists(bpath):
    with open(bpath) as f:
        bd = json.load(f)
    for pol, mets in (bd.get('summary') or {}).items():
        baselines[pol] = {m: mets.get(m, {}).get('mean') for m in METRICS}

# ---- per-config RL scores ----
configs = {}
for cfg in CONFIGS:
    sc = rl_scores(f'{cfg}_s*.json')
    if any(sc[m] for m in METRICS):
        configs[cfg] = sc


def agg(vals):
    a = np.array([v for v in vals if v == v], dtype=float)
    if a.size == 0:
        return {'mean': float('nan'), 'std': float('nan'), 'n': 0, 'vals': []}
    return {'mean': float(a.mean()),
            'std': float(a.std(ddof=1)) if a.size > 1 else 0.0,
            'n': int(a.size), 'vals': a.tolist()}


def fmt(x):
    return 'nan' if x != x else f'{x:.2f}'


# ============ TABLE 1: full model (n=5) vs baselines ============
print('=' * 92)
print('TABLE 1 — Full RL model (mean +/- std over 5 INDEPENDENT training seeds) vs baselines')
print('=' * 92)
hdr = ['discovery', 'ttd', 'coverage', 'overlap', 'dwell_mov', 'ttd_mover']
print(f'{"policy":26s} ' + ' '.join(f'{h:>11s}' for h in hdr))
print('-' * 92)
full = configs.get('full', {m: [] for m in METRICS})
fa = {m: agg(full[m]) for m in METRICS}
print(f'{"RL full (n=" + str(fa["discovery_rate"]["n"]) + ")":26s} ' +
      ' '.join(f'{fmt(fa[m]["mean"]) + "±" + fmt(fa[m]["std"]):>11s}' for m in METRICS))
for pol in ['heuristic', 'sweep', 'greedy', 'random']:
    if pol in baselines:
        b = baselines[pol]
        print(f'{pol:26s} ' + ' '.join(f'{fmt(b[m]):>11s}' for m in METRICS))
print('-' * 92)
print('cols: ' + ', '.join(f'{SHORT[m]}({"+" if HIGHER_BETTER[m] else "-"})' for m in METRICS)
      + '   (+ higher-better, - lower-better)')

# ---- RL(full) vs each baseline on discovery + ttd: effect size + consistency ----
comp = {}
print('\nRL(full) vs baselines — discovery_rate & time_to_detect (unit = training seed, n=5):')
for pol in ['heuristic', 'sweep', 'greedy', 'random']:
    if pol not in baselines:
        continue
    comp[pol] = {}
    for m in ['discovery_rate', 'time_to_detect_steps']:
        rl = np.array(full[m], dtype=float)
        b = baselines[pol][m]
        if rl.size == 0 or b is None:
            continue
        diff = rl - b
        better = (diff > 0) if HIGHER_BETTER[m] else (diff < 0)
        d = float(diff.mean() / rl.std(ddof=1)) if rl.size > 1 and rl.std(ddof=1) > 0 else float('nan')
        rec = {'rl_mean': float(rl.mean()), 'baseline': float(b),
               'mean_diff': float(diff.mean()), 'cohens_d': d,
               'n_seeds_better': int(better.sum()), 'n': int(rl.size)}
        if sst is not None and rl.size > 1:
            t, pv = sst.ttest_1samp(rl, b)
            rec['t'] = float(t); rec['p'] = float(pv)
        comp[pol][m] = rec
        line = (f'  RL vs {pol:9s} {SHORT[m]:9s}: RL {rl.mean():.2f} vs {b:.2f}  '
                f'diff={diff.mean():+.2f}  d={d:+.2f}  '
                f'seeds_better={better.sum()}/{rl.size}')
        if 'p' in rec:
            line += f'  t={rec["t"]:+.2f} p={rec["p"]:.3f}'
        print(line)

# ============ TABLE 2: ablations vs full ============
print('\n' + '=' * 92)
print('TABLE 2 — Ablations: effect of removing each component (RL, mean +/- std over 3 seeds)')
print('=' * 92)
print(f'{"config":28s} ' + ' '.join(f'{h:>11s}' for h in hdr) + f' {"n":>3s}')
print('-' * 92)
abl_summary = {}
for cfg in CONFIGS:
    if cfg not in configs:
        continue
    a = {m: agg(configs[cfg][m]) for m in METRICS}
    abl_summary[cfg] = a
    n = a['discovery_rate']['n']
    print(f'{CONFIG_LABEL[cfg]:28s} ' +
          ' '.join(f'{fmt(a[m]["mean"]) + "±" + fmt(a[m]["std"]):>11s}' for m in METRICS) +
          f' {n:>3d}')
print('-' * 92)
print('Delta vs full (discovery_rate, ttd): how much each component matters')
for cfg in CONFIGS:
    if cfg == 'full' or cfg not in abl_summary:
        continue
    dd = abl_summary[cfg]['discovery_rate']['mean'] - fa['discovery_rate']['mean']
    dt = abl_summary[cfg]['time_to_detect_steps']['mean'] - fa['time_to_detect_steps']['mean']
    print(f'  {CONFIG_LABEL[cfg]:28s} discovery {dd:+.3f}   ttd {dt:+.2f} steps')

# ---- save ----
out = {
    'unit_of_analysis': 'independent_training_seed',
    'full': {m: fa[m] for m in METRICS},
    'baselines': baselines,
    'ablations': {c: {m: abl_summary[c][m] for m in METRICS} for c in abl_summary},
    'rl_vs_baseline': comp,
}
if args.out:
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {args.out}')
