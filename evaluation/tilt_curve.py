"""Aggregate the ovsweep tilt-sensitivity sweep (scripts/eval_ovsweep_tilt.sh).

Shows that the overlap-optimized sweep's dominance is a GEOMETRIC property, not a
test-set-tuned hyperparameter: inter-camera overlap collapses to ~0 exactly where the
optical-axis separation 2*tilt exceeds the 90-deg sum of the two 45-deg FOV half-angles
(tilt > 45 deg), and discovery/latency are flat over a wide plateau above that knee.
Reports both the held-out TEST clips and the held-in TRAIN clips, so target_tilt can be
shown to be selectable without touching test data.

Usage: python evaluation/tilt_curve.py
"""
import json, glob, os, re

M = ['discovery_rate', 'time_to_detect_steps', 'coverage_pct',
     'inter_camera_overlap', 'solid_angle_overlap', 'time_to_detect_mover_steps']
OUT = 'results/full128/ovsweep_tilt'


def load(split):
    rows = {}
    for f in sorted(glob.glob(f'{OUT}/{split}_t*.json')):
        tilt = int(re.search(r'_t(\d+)\.json$', f).group(1))
        s = json.load(open(f))['summary']['ovsweep']
        # metrics.py writes {mean,std,n} dicts
        rows[tilt] = {m: (s[m]['mean'] if isinstance(s.get(m), dict) else s.get(m)) for m in M}
    return dict(sorted(rows.items()))


def main():
    all_rows = {}
    for split in ('test', 'train'):
        rows = load(split)
        if not rows:
            print(f'[{split}] no data'); continue
        all_rows[split] = rows
        print(f'\n=== ovsweep tilt sweep: {split} clips '
              f'(separation = 2*tilt; cones disjoint once > 90 deg) ===')
        print(f"{'tilt':>5} {'sep':>5} {'disc':>6} {'TTD':>7} {'cov':>6} "
              f"{'ovlp':>6} {'ovlpO':>7} {'TTDmov':>7}")
        for t, r in rows.items():
            def g(m):
                v = r.get(m)
                return '   n/a' if v is None else f'{v:6.2f}'
            print(f'{t:>5} {2*t:>5} {g("discovery_rate")} {g("time_to_detect_steps")} '
                  f'{g("coverage_pct")} {g("inter_camera_overlap")} '
                  f'{g("solid_angle_overlap")} {g("time_to_detect_mover_steps")}')
        # knee check
        disj = [t for t, r in rows.items()
                if (r.get('inter_camera_overlap') or 1.0) <= 0.05]
        if disj:
            print(f'  -> overlap<=0.05 for tilt >= {min(disj)} deg '
                  f'(predicted knee: tilt > 45 deg)')
        best = max(rows.items(), key=lambda kv: kv[1]['discovery_rate'])
        print(f'  -> best discovery {best[1]["discovery_rate"]:.2f} at tilt {best[0]} deg')
    json.dump(all_rows, open(f'{OUT}/tilt_summary.json', 'w'), indent=1)
    print(f'\nwrote {OUT}/tilt_summary.json')


if __name__ == '__main__':
    main()
