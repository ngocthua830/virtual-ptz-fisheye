"""Survival analysis of time-to-detection (review #14).

The plain time-to-detect averages only over *discovered* people, so a policy can look
fast by missing hard targets. Here we treat detection latency as a survival time with
right-censoring: a person never detected within their observation window is censored, not
dropped. We report, per policy:

  * Kaplan-Meier survival S(t) = P(not yet detected at latency t)  [curve]
  * Restricted mean time-to-detect RMTTD(tau) = integral_0^tau S(t) dt
    (expected latency capped at tau; undetected people contribute up to tau) -- the
    censoring-aware analogue of mean TTD, lower = faster.
  * censored fraction = share of appeared people never detected.

Input: results/full128/horizon5/*.json (each run dict carries a 'survival' list of
[time, event] pairs; event=1 detected at that latency, event=0 censored at that time).
RL pools all 5 seeds x 3 clips; baselines pool their runs.
"""
import json, glob, sys
import numpy as np


def km(pairs):
    """Kaplan-Meier estimate. pairs: list of [t, event]. Returns (times, S) step fn."""
    if not pairs:
        return np.array([0.0]), np.array([1.0])
    t = np.array([p[0] for p in pairs], float)
    e = np.array([p[1] for p in pairs], int)
    order = np.argsort(t)
    t, e = t[order], e[order]
    n = len(t)
    times, S = [0.0], [1.0]
    surv = 1.0
    for ut in np.unique(t):
        at_risk = int(np.sum(t >= ut - 1e-9))
        d = int(np.sum((np.abs(t - ut) < 1e-9) & (e == 1)))
        if at_risk > 0 and d > 0:
            surv *= (1.0 - d / at_risk)
        times.append(float(ut)); S.append(surv)
    return np.array(times), np.array(S)


def rmttd(times, S, tau):
    """Restricted mean time-to-detect = area under S(t) from 0 to tau."""
    area = 0.0
    for i in range(len(times) - 1):
        t0, t1 = times[i], min(times[i + 1], tau)
        if t1 <= t0:
            continue
        area += S[i] * (t1 - t0)          # S is left-continuous step fn
        if times[i + 1] >= tau:
            break
    if times[-1] < tau:                    # extend last level to tau
        area += S[-1] * (tau - times[-1])
    return area


def load_policy_pairs():
    pols = {}
    # learned / seed-pooled controllers: pool all 5 seeds x 3 clips
    for label, pat, key in [('RL', "results/full128/horizon5/full_s*.json", 'rl'),
                            ('hybrid', "results/full128/horizon5_hybrid/full_s*.json", 'hybrid')]:
        pols[label] = []
        for f in sorted(glob.glob(pat)):
            for run in json.load(open(f))['policies'].get(key, []):
                pols[label].extend(run.get('survival', []))
    b = json.load(open("results/full128/horizon5/baselines.json"))['policies']
    for name in ['sweep', 'coordsweep', 'random', 'greedy', 'heuristic']:
        pols[name] = []
        for run in b.get(name, []):
            pols[name].extend(run.get('survival', []))
    # overlap-optimized sweep: deterministic, one run per clip
    ovf = "results/full128/horizon5_ovsweep/ovsweep.json"
    pols['ovsweep'] = []
    for run in json.load(open(ovf))['policies'].get('ovsweep', []):
        pols['ovsweep'].extend(run.get('survival', []))
    return pols


def main():
    pols = load_policy_pairs()
    taus = [30, 60, 90]
    print(f"{'policy':<11} {'n':>5} {'cens%':>7} " + " ".join(f"RMTTD@{t:<3}" for t in taus))
    rows = {}
    for name, pairs in pols.items():
        if not pairs:
            print(f"{name:<11} (no data)"); continue
        n = len(pairs)
        cens = 100.0 * sum(1 for p in pairs if p[1] == 0) / n
        times, S = km(pairs)
        vals = [rmttd(times, S, t) for t in taus]
        rows[name] = {'n': n, 'cens_pct': cens, 'times': times.tolist(), 'S': S.tolist(),
                      'rmttd': dict(zip(map(str, taus), vals))}
        print(f"{name:<11} {n:>5} {cens:>6.1f}% " + " ".join(f"{v:>8.2f}" for v in vals))
    json.dump(rows, open("results/full128/horizon5/survival_summary.json", "w"), indent=1)
    print("\nwrote results/full128/horizon5/survival_summary.json")


if __name__ == '__main__':
    main()
