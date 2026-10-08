"""E1: strict reliability evaluation of the per-history waiting rules on B3.

Implements configs/b3_waiting_strict_protocol.json, frozen before its only run. Receivers are the distinct
posterior vectors with independent reporter noise; each split assigns disjoint reference, calibration and test
roles. Rule parameters are chosen on calibration receivers (certified by fixed-sequence binomial tests, or by the
earlier plug-in targets) and scored once on test receivers with one-sided Clopper-Pearson bounds and paired
bootstrap intervals. Simulation of one published model structure, not evidence about measured cells.
Usage: python b3_waiting_strict.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # one BLAS thread per process: no oversubscription in pools
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json

import numpy as np

import b3_model as b3
import b3_readiness as br
import b3_waiting as bw
import b3_waiting_frontier as fr
import strict_eval as se

PROTOCOL = br.ROOT / 'configs/b3_waiting_strict_protocol.json'
FREEZE = br.OUT / 'b3_waiting_strict_protocol_freeze.json'
RESULTS = br.OUT / 'waiting_strict_results.json'
SPLITS, SPLIT_SEED, NOISE_SEED = 20, 20261009, 20261109
NOISES = (.005, .01)
ALPHA = .025
ARMS = ('certified', 'plugin_090', 'plugin_094')
PLUGIN_TARGETS = {'plugin_090': .90, 'plugin_094': .94}
RULES = ('fixed', 'current', 'smoothed', 'history')
ADAPTIVE = ('current', 'smoothed', 'history')
BOOT, BOOT_SEED = 2000, 20261012
CHOICE_KEYS = ('certified', 'calibration', 'target_reached', 'fallback')


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    bw.load_protocol()
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def roles(r):
    """Reference, calibration and test receivers of split r (indices into the readiness tables)."""
    return se.three_way_split(se.distinct_samples(b3.load_posterior()), SPLIT_SEED + r)


def noise_seed(r):
    return NOISE_SEED + 100000 * r


def options_and_waits(probability, success, plants, deadline):
    """Every parameter of every rule as (label, ok, completion) with arrays (H, n), in the order of
    b3_waiting_frontier.candidates, and the decision waits keyed by (rule, tie key)."""
    options, waits = {}, {}

    def add(name, label, k):
        ok, completion = fr.scored(success, plants, k, deadline)
        options.setdefault(name, []).append((label, ok, completion))
        waits[(name, se.tie_key(label))] = br.GRID[k]

    for name in ('current', 'history'):
        k = fr.crossing_index(probability[name], fr.THRESHOLDS)
        for i, c in enumerate(fr.THRESHOLDS):
            add(name, dict(threshold=float(c)), k[i])
    for tau in bw.SMOOTH_TAUS:
        k = fr.crossing_index(probability[f'smoothed_{tau:g}'], fr.THRESHOLDS)
        for i, c in enumerate(fr.THRESHOLDS):
            add('smoothed', dict(tau_min=tau, threshold=float(c)), k[i])
    shape = (success.shape[0], len(plants))
    for j, w in enumerate(br.GRID):
        add('fixed', dict(wait_min=float(w)), np.full(shape, j))
    return options, waits


def chains(options, name):
    """Candidate chains from the most conservative to the most aggressive parameter."""
    if name == 'smoothed':
        return [[o for o in reversed(options[name]) if o[0]['tau_min'] == tau] for tau in bw.SMOOTH_TAUS]
    return [list(reversed(options[name]))]


def choose_arm(options, arm):
    if arm == 'certified':
        return {name: {hist: se.certified_choice(chains(options, name), h, br.Q, ALPHA)
                       for h, hist in enumerate(br.HISTORIES)} for name in RULES}
    return {name: {hist: fr.choose(options[name], h, PLUGIN_TARGETS[arm]) for h, hist in enumerate(br.HISTORIES)}
            for name in RULES}


def action(options, waits, name, choice):
    if choice.get('fallback') == se.FALLBACK:
        name, label = 'fixed', dict(wait_min=120.)
    else:
        label = {k: v for k, v in choice.items() if k not in CHOICE_KEYS}
    ok, completion = next((ok, c) for lab, ok, c in options[name] if lab == label)
    return ok, completion, waits[(name, se.tie_key(label))]


def per_history(ok, completion, wait):
    out = {}
    for h, hist in enumerate(br.HISTORIES):
        k, n = int(ok[h].sum()), int(ok.shape[1])
        out[hist] = dict(n=n, successes=k, success_rate=k / n, lower_95=se.cp_lower(k, n, .05),
                         lower_975=se.cp_lower(k, n, .025), mean_completion_min=float(completion[h].mean()),
                         mean_completion_success_min=float(completion[h][ok[h]].mean()) if k else None,
                         mean_wait_min=float(wait[h].mean()), failures=n - k)
    out['all'] = dict(success_rate=float(ok.mean()), mean_completion_min=float(completion.mean()))
    return out


def paired(outcomes):
    out = {}
    fixed_ok, fixed_completion = outcomes['fixed'][0].astype(float), outcomes['fixed'][1]
    for name in ADAPTIVE:
        ok, completion = outcomes[name][0].astype(float), outcomes[name][1]
        parts = {'all': ((ok - fixed_ok).mean(0), (completion - fixed_completion).mean(0))}
        parts.update({hist: (ok[h] - fixed_ok[h], completion[h] - fixed_completion[h])
                      for h, hist in enumerate(br.HISTORIES)})
        out[name] = {part: dict(success=float(ds.mean()), completion=float(dc.mean()),
                                success_ci=se.paired_bootstrap(ds, BOOT, BOOT_SEED),
                                completion_ci=se.paired_bootstrap(dc, BOOT, BOOT_SEED))
                     for part, (ds, dc) in parts.items()}
    return out


def run_split(r, noise, tables=None, success=None):
    tables = dict(np.load(br.TABLES)) if tables is None else tables
    success = br.success_table(tables) if success is None else success
    R, C, T = roles(r)
    observed = bw.noisy_observations(tables, noise, seed=noise_seed(r))
    options_c, _ = options_and_waits(bw.estimator_probabilities(tables, success, noise, R, C, observed),
                                     success, C, br.DEADLINE_MIN)
    options_t, waits_t = options_and_waits(bw.estimator_probabilities(tables, success, noise, R, T, observed),
                                           success, T, br.DEADLINE_MIN)
    k_oracle = fr.crossing_index(success[:, T, :].astype(float), [1.])[0]
    oracle_ok, oracle_completion = fr.scored(success, T, k_oracle, br.DEADLINE_MIN)
    arms = {}
    for arm in ARMS:
        choices = choose_arm(options_c, arm)
        outcomes = {}
        for name in RULES:
            parts = [action(options_t, waits_t, name, choices[name][hist]) for hist in br.HISTORIES]
            outcomes[name] = tuple(np.stack([parts[h][i][h] for h in range(len(parts))]) for i in range(3))
        arms[arm] = dict(choices=choices, rules={name: per_history(*outcomes[name]) for name in RULES},
                         paired=paired(outcomes))
    return dict(r=r, noise=noise, roles=dict(reference=R.tolist(), calibration=C.tolist(), test=T.tolist()),
                arms=arms, oracle=per_history(oracle_ok, oracle_completion, br.GRID[k_oracle]))


def _job(job):
    return run_split(*job)


def stats(values):
    v = np.asarray(values, dtype=float)
    return dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()))


def summarize(repeats):
    out = {}
    for noise in (f'{n:g}' for n in NOISES):
        out[noise] = {}
        for arm in ARMS:
            rows = [repeats[str(r)][noise]['arms'][arm] for r in range(SPLITS)]
            out[noise][arm] = dict(
                meets_per_history={name: int(sum(all(row['rules'][name][h]['success_rate'] >= br.Q
                                                     for h in br.HISTORIES) for row in rows)) for name in RULES},
                joint_bound={name: int(sum(all(row['rules'][name][h]['lower_975'] >= br.Q for h in br.HISTORIES)
                                           for row in rows)) for name in RULES},
                certified_both={name: (int(sum(all(row['choices'][name][h]['certified'] for h in br.HISTORIES)
                                               for row in rows)) if arm == 'certified' else None) for name in RULES},
                success={name: {h: stats([row['rules'][name][h]['success_rate'] for row in rows])
                                for h in br.HISTORIES} for name in RULES},
                completion={name: stats([row['rules'][name]['all']['mean_completion_min'] for row in rows])
                            for name in RULES},
                paired_completion={name: stats([row['paired'][name]['all']['completion'] for row in rows])
                                   for name in ADAPTIVE},
                faster={name: int(sum(row['paired'][name]['all']['completion'] < 0 for row in rows))
                        for name in ADAPTIVE})
        out[noise]['oracle'] = dict(
            success={h: stats([repeats[str(r)][noise]['oracle'][h]['success_rate'] for r in range(SPLITS)])
                     for h in br.HISTORIES},
            completion=stats([repeats[str(r)][noise]['oracle']['all']['mean_completion_min'] for r in range(SPLITS)]))
    return out


def statements(repeats, summary):
    primary = repeats['0']['0.005']['arms']
    block = summary['0.005']
    s1 = {}
    for name in RULES:
        s1[name] = {hist: {k: primary['certified']['rules'][name][hist][k]
                           for k in ('successes', 'n', 'success_rate', 'lower_95')} for hist in br.HISTORIES}
        s1[name]['splits_meeting'] = block['certified']['meets_per_history'][name]
    s2 = {name: dict(split0=primary['certified']['paired'][name]['all']['completion'],
                     ci=primary['certified']['paired'][name]['all']['completion_ci'],
                     median=block['certified']['paired_completion'][name]['median'],
                     faster=block['certified']['faster'][name]) for name in ADAPTIVE}
    s3 = {name: block['plugin_090']['meets_per_history'][name] for name in RULES}
    s4 = {name: float(np.median([repeats[str(r)]['0.005']['arms']['certified']['rules'][name]['all']['mean_completion_min']
                                 - repeats[str(r)]['0.005']['arms']['plugin_094']['rules'][name]['all']['mean_completion_min']
                                 for r in range(SPLITS)])) for name in RULES}
    return {'E1-S1': s1, 'E1-S2': s2, 'E1-S3': s3, 'E1-S4': s4}


def main():
    load_protocol()
    jobs = [(r, noise) for r in range(SPLITS) for noise in NOISES]
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count() or 1)) as pool:
        results = list(pool.map(_job, jobs))
    repeats = {}
    for (r, noise), result in zip(jobs, results):
        repeats.setdefault(str(r), {})[f'{noise:g}'] = result
    summary = summarize(repeats)
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  repeats=repeats, summary=summary, statements=statements(repeats, summary),
                  limitations=['Simulation of one published model structure; not evidence about measured cells.',
                               'Receivers are seen posterior vectors with new noise and new roles, not unseen receivers.',
                               'Posterior vectors express population-mean uncertainty, not cell heterogeneity.',
                               'Calibration uses complete outcome curves (ideal information).'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    for noise in ('0.005', '0.01'):
        for arm in ARMS:
            b = summary[noise][arm]
            print(noise, arm, 'meets', b['meets_per_history'], 'joint', b['joint_bound'], 'cert', b['certified_both'],
                  'faster', b['faster'], 'dT', {k: round(v['median'], 2) for k, v in b['paired_completion'].items()},
                  'T', {k: round(v['median'], 2) for k, v in b['completion'].items()})
    print(json.dumps(result['statements'], indent=1))


if __name__ == '__main__':
    main()
