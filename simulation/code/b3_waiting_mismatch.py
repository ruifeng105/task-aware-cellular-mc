"""Robustness of the per-history calibrated waiting rules to recovery errors of the B3 model.

Implements configs/b3_waiting_mismatch_protocol.json, frozen before its only
run. The rules of b3_waiting_frontier are scored on plants whose recovery
after the 30-min command differs from the model: a lag hidden from the
reporter (outcomes delayed, reporter unchanged), a lag the reporter reveals
(reporter and outcomes delayed, so plant trajectories leave the B3 family) and
a slow posterior tail (plants restricted to slow posterior samples). Rules
keep their model-calibrated parameters or are recalibrated against the
plant's outcomes on calibration samples, with thresholds only or with an
added delay after the threshold crossing. Simulation of one published model
structure with stylized recovery errors, not evidence about measured cells.
"""
import hashlib
import json
import os

import numpy as np

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_frontier as wf
import b3_waiting_robustness as rb
from parallel_jobs import run_jobs

PROTOCOL = br.ROOT / 'configs/b3_waiting_mismatch_protocol.json'
FREEZE = br.OUT / 'b3_waiting_mismatch_protocol_freeze.json'
RESULTS = br.OUT / 'waiting_mismatch_results.json'
SPLITS = br.OUT / 'waiting_mismatch_splits.json'
NATURAL_PROBE = br.OUT / 'natural_probe_results.json'
MC_LAGS = tuple(range(0, 41, 2))
RECAL_LAGS = tuple(range(2, 17, 2))
VISIBLE_LAGS = (4, 8, 12, 16)
ONSET_CUTS = (90., 98.)
DELTAS = tuple(range(0, 41, 2))
TARGETS = wf.TARGETS
NOISES = wf.NOISES
SHORT, LONG = br.HISTORIES.index('short'), br.HISTORIES.index('long')
ESS_TIMES = (60., 90.)
SLACK_QUANTILES = (.05, .10, .25, .50)
PRIMARY_NOISE, PRIMARY_TARGET = '0.005', '0.94'
ROBUST_SPLITS = 16
NULL_DRAWS, NULL_SEED = 100_000, 20261010
PROBES = {'mixed': ('fgf_mixed_25ng/pulse_3', LONG, 60.), 'three_twenty': ('fgf_3_20_25ng/pulse_2', SHORT, 20.)}
DELAY_ORDER = ('delta_min', 'tau_min', 'threshold', 'wait_min')


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def check_inputs(protocol):
    for rel, expected in protocol['inputs_sha256'].items():
        actual = hashlib.sha256((br.ROOT.parent / rel).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f'Input {rel} differs from the hash pinned in the protocol.')


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    wf.load_protocol()
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    check_inputs(protocol)
    return protocol


# ----- plants -----------------------------------------------------------------------------------

def lagged(success, lag):
    """Plant outcomes whose readiness after the 30-min command is delayed by `lag` minutes (failure before)."""
    shift = int(round(lag / 2))
    out = success.copy()
    out[LONG] = False
    if shift < success.shape[-1]:
        out[LONG, :, shift:] = success[LONG, :, :success.shape[-1] - shift]
    return out


def visible_observations(tables, observed, lag):
    """Observations of plants whose noise-free reporter after the 30-min command is delayed by `lag` minutes."""
    fret = tables['history_fret'].astype(float)
    shift = int(round(lag / 2))
    shifted = np.empty_like(fret[LONG])
    shifted[:, :shift] = fret[LONG, :, :1]
    shifted[:, shift:] = fret[LONG, :, :fret.shape[-1] - shift]
    out = observed.copy()
    out[LONG] = shifted + (observed[LONG] - fret[LONG])
    return out


def onsets(success):
    """Readiness onset (min) per history and sample: first decision time with success, +inf if never."""
    return np.where(success.any(-1), br.GRID[success.argmax(-1)], np.inf)


def slow_mask(success, cut):
    return onsets(success)[LONG] >= cut


def subset(probability, idx):
    return {name: value[:, idx] for name, value in probability.items()}


# ----- rule options, choices and cells ----------------------------------------------------------

def delay_candidates(probability, success, plants, deadline):
    """Every (threshold, delta) [and scale] of the adaptive rules and every wait of the fixed rule."""
    out = {}
    last = br.GRID.size - 1
    for name in ('current', 'history'):
        cross = wf.crossing_index(probability[name], wf.THRESHOLDS)
        out[name] = []
        for delta in DELTAS:
            ok, completion = wf.scored(success, plants, np.minimum(cross + delta // 2, last), deadline)
            out[name] += [(dict(threshold=float(c), delta_min=float(delta)), ok[i], completion[i])
                          for i, c in enumerate(wf.THRESHOLDS)]
    out['smoothed'] = []
    for tau in bw.SMOOTH_TAUS:
        cross = wf.crossing_index(probability[f'smoothed_{tau:g}'], wf.THRESHOLDS)
        for delta in DELTAS:
            ok, completion = wf.scored(success, plants, np.minimum(cross + delta // 2, last), deadline)
            out['smoothed'] += [(dict(tau_min=tau, threshold=float(c), delta_min=float(delta)), ok[i], completion[i])
                                for i, c in enumerate(wf.THRESHOLDS)]
    out['fixed'] = wf.candidates(probability, success, plants, deadline)['fixed']
    return out


def choose_delay(options, h, target):
    """wf.choose with ties broken by smaller delta, then scale, then threshold (then wait)."""
    key = lambda label: tuple(label[k] for k in DELAY_ORDER if k in label)
    rows = [(label, float(ok[h].mean()), float(completion[h].mean())) for label, ok, completion in options]
    feasible = [row for row in rows if row[1] >= target]
    if feasible:
        label, s, c = min(feasible, key=lambda row: (row[2], key(row[0])))
        return dict(label, target_reached=True, calibration=dict(success_rate=s, mean_completion_min=c))
    label, s, c = min(rows, key=lambda row: (-row[1], row[2], key(row[0])))
    return dict(label, target_reached=False, calibration=dict(success_rate=s, mean_completion_min=c))


def make_cell(options_cal, options_ev, oracle, target, chooser, oracle_cal=None, keep_choices=True):
    """Choices on calibration options, outcomes on evaluation options, with feasibility flags."""
    choices = {name: {hist: chooser(options_cal[name], h, target) for h, hist in enumerate(br.HISTORIES)}
               for name in wf.RULES}
    outcomes = wf.evaluated(options_ev, choices)
    rules = {name: rb.summarise(*pair) for name, pair in outcomes.items()}
    rules['oracle'] = rb.summarise(*oracle)
    differences = wf.paired(outcomes)
    cell = dict(rules=rules, gap_closure=wf.gap_closure(rules),
                meets_per_history={name: bool(all(rules[name][h]['success_rate'] >= br.Q for h in br.HISTORIES))
                                   for name in wf.RULES},
                paired={pair: {key: float(v.mean()) for key, v in d.items()} for pair, d in differences.items()},
                feasible_evaluation=bool(all(rules['oracle'][h]['success_rate'] >= br.Q for h in br.HISTORIES)))
    if oracle_cal is not None:
        cell['feasible_calibration'] = bool(oracle_cal[0][LONG].mean() >= target)
    if keep_choices:
        cell['choices'] = choices
        cell['target_reached'] = {name: bool(all(choices[name][h]['target_reached'] for h in br.HISTORIES))
                                  for name in wf.RULES}
    return json.loads(json.dumps(cell)), choices


def decision_indices(probability, choices, n_plants):
    """Decision index per rule, history and plant for chosen parameters (with an optional delay)."""
    last = br.GRID.size - 1
    out = {}
    for name in wf.RULES:
        k = np.empty((len(br.HISTORIES), n_plants), dtype=int)
        for h, hist in enumerate(br.HISTORIES):
            c = choices[name][hist]
            if name == 'fixed':
                k[h] = int(np.flatnonzero(br.GRID == c['wait_min'])[0])
                continue
            key = f"smoothed_{c['tau_min']:g}" if name == 'smoothed' else name
            cross = wf.crossing_index(probability[key][h:h + 1], [c['threshold']])[0, 0]
            k[h] = np.minimum(cross + int(round(c.get('delta_min', 0.) / 2)), last)
        out[name] = k
    return out


def slack_summary(indices, success, plants):
    """Slack = decision time - model readiness onset per rule and history (never ready: -inf)."""
    onset = onsets(success[:, plants, :])
    out = {}
    for name, k in indices.items():
        slack = br.GRID[k] - onset
        out[name] = {hist: dict(quantiles={f'{q:.2f}': float(np.quantile(slack[h], q)) for q in SLACK_QUANTILES},
                                share_at_least={f'{lag}': float((slack[h] >= lag).mean()) for lag in MC_LAGS})
                     for h, hist in enumerate(br.HISTORIES)}
    return out


def ess_summary(tables, observed, noise, reference, plants):
    """Median and quartiles of 1/sum(w^2) of the B3-belief weights after the 30-min command."""
    fret = tables['history_fret'][LONG].astype(float)[reference]
    sigma = bw.likelihood_sd(noise)
    out = {}
    for t in ESS_TIMES:
        k = int(np.flatnonzero(br.GRID == t)[0])
        values = []
        for p in plants:
            log_w = -((fret[:, :k + 1] - observed[LONG, p, :k + 1]) ** 2).sum(1) / (2 * sigma ** 2)
            w = np.exp(log_w - log_w.max())
            w /= w.sum()
            values.append(1. / (w ** 2).sum())
        out[f'{t:g}'] = dict(zip(('q25', 'median', 'q75'), map(float, np.percentile(values, [25, 50, 75]))))
    return out


# ----- one split -------------------------------------------------------------------------------

def recalibrated_cells(on_cal, on_ev, plant, cal, ev, deadline, oracle_ev):
    """Threshold-only and delay-recalibrated cells for one plant outcome table."""
    oracle_cal = wf.oracle_outcomes(plant, cal, deadline)
    plain_cal, plain_ev = wf.candidates(on_cal, plant, cal, deadline), wf.candidates(on_ev, plant, ev, deadline)
    delay_cal, delay_ev = delay_candidates(on_cal, plant, cal, deadline), delay_candidates(on_ev, plant, ev, deadline)
    plain = {f'{t:.2f}': make_cell(plain_cal, plain_ev, oracle_ev, t, wf.choose, oracle_cal)[0] for t in TARGETS}
    delay = {f'{t:.2f}': make_cell(delay_cal, delay_ev, oracle_ev, t, choose_delay, oracle_cal)[0] for t in TARGETS}
    return plain, delay


def run_repeat(r, noise):
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    split_seed, noise_seed = rb.seeds(r)
    cal, ev = bw.split_samples(success.shape[1], seed=split_seed)
    observed = bw.noisy_observations(tables, noise, seed=noise_seed)
    deadline = br.DEADLINE_MIN
    on_cal = bw.estimator_probabilities(tables, success, noise, cal, cal, observed)
    on_ev = bw.estimator_probabilities(tables, success, noise, cal, ev, observed)
    base_cal, base_ev = wf.candidates(on_cal, success, cal, deadline), wf.candidates(on_ev, success, ev, deadline)
    oracle = wf.oracle_outcomes(success, ev, deadline)
    out = dict(r=r, noise=noise, split_seed=split_seed, noise_seed=noise_seed, baseline={}, slack={},
               baseline_delay={}, hidden_lag=dict(model_calibrated={}, outcome_recalibrated={}, outcome_recalibrated_delay={}),
               visible_lag=dict(model_calibrated={}, outcome_recalibrated={}, outcome_recalibrated_delay={}),
               slow_posterior_tail={}, ess=dict(baseline=ess_summary(tables, observed, noise, cal, ev), visible_lag={}))
    base_choices = {}
    for t in TARGETS:
        key = f'{t:.2f}'
        out['baseline'][key], base_choices[key] = make_cell(base_cal, base_ev, oracle, t, wf.choose)
        out['slack'][key] = slack_summary(decision_indices(on_ev, base_choices[key], ev.size), success, ev)
    delay_cal, delay_ev = delay_candidates(on_cal, success, cal, deadline), delay_candidates(on_ev, success, ev, deadline)
    out['baseline_delay'] = {f'{t:.2f}': make_cell(delay_cal, delay_ev, oracle, t, choose_delay)[0] for t in TARGETS}
    del delay_cal, delay_ev
    for lag in MC_LAGS:
        plant = lagged(success, lag)
        plant_ev = wf.candidates(on_ev, plant, ev, deadline)
        oracle_lag = wf.oracle_outcomes(plant, ev, deadline)
        out['hidden_lag']['model_calibrated'][f'{lag}'] = {
            f'{t:.2f}': make_cell(base_cal, plant_ev, oracle_lag, t, wf.choose, keep_choices=False)[0] for t in TARGETS}
        if lag in RECAL_LAGS:
            plain, delay = recalibrated_cells(on_cal, on_ev, plant, cal, ev, deadline, oracle_lag)
            out['hidden_lag']['outcome_recalibrated'][f'{lag}'] = plain
            out['hidden_lag']['outcome_recalibrated_delay'][f'{lag}'] = delay
    for lag in VISIBLE_LAGS:
        plant = lagged(success, lag)
        seen = visible_observations(tables, observed, lag)
        vis_cal = bw.estimator_probabilities(tables, success, noise, cal, cal, seen)
        vis_ev = bw.estimator_probabilities(tables, success, noise, cal, ev, seen)
        oracle_lag = wf.oracle_outcomes(plant, ev, deadline)
        plant_ev = wf.candidates(vis_ev, plant, ev, deadline)
        out['visible_lag']['model_calibrated'][f'{lag}'] = {
            f'{t:.2f}': make_cell(base_cal, plant_ev, oracle_lag, t, wf.choose, keep_choices=False)[0] for t in TARGETS}
        plain, delay = recalibrated_cells(vis_cal, vis_ev, plant, cal, ev, deadline, oracle_lag)
        out['visible_lag']['outcome_recalibrated'][f'{lag}'] = plain
        out['visible_lag']['outcome_recalibrated_delay'][f'{lag}'] = delay
        out['ess']['visible_lag'][f'{lag}'] = ess_summary(tables, seen, noise, cal, ev)
    for cut in ONSET_CUTS:
        mask = slow_mask(success, cut)
        ev_idx, cal_idx = np.flatnonzero(mask[ev]), np.flatnonzero(mask[cal])
        slow_ev = wf.candidates(subset(on_ev, ev_idx), success, ev[ev_idx], deadline)
        oracle_slow = wf.oracle_outcomes(success, ev[ev_idx], deadline)
        model = {f'{t:.2f}': make_cell(base_cal, slow_ev, oracle_slow, t, wf.choose)[0] for t in TARGETS}
        plain, delay = recalibrated_cells(subset(on_cal, cal_idx), subset(on_ev, ev_idx), success, cal[cal_idx], ev[ev_idx],
                                          deadline, oracle_slow)
        out['slow_posterior_tail'][f'{cut:g}'] = dict(
            n_evaluation=int(ev_idx.size), n_calibration=int(cal_idx.size),
            never_ready_evaluation=int((~success[LONG, ev[ev_idx]].any(-1)).sum()),
            model_calibrated=model, outcome_recalibrated=plain, outcome_recalibrated_delay=delay,
            ess=ess_summary(tables, observed, noise, cal, ev[ev_idx]))
    return json.loads(json.dumps(out))


def _job(job):
    return run_repeat(*job)


# ----- summaries, statements and descriptive results -------------------------------------------

def summarize(rows):
    """Split statistics of a list of cells (frontier-style, plus feasibility and calibration counts)."""
    out = dict(
        meets_per_history={name: int(sum(row['meets_per_history'][name] for row in rows)) for name in wf.RULES},
        feasible_evaluation=int(sum(row['feasible_evaluation'] for row in rows)),
        success={name: {part: rb.stats([row['rules'][name][part]['success_rate'] for row in rows]) for part in wf.PARTS}
                 for name in wf.RULES + ('oracle',)},
        completion={name: {part: rb.stats([row['rules'][name][part]['mean_completion_min'] for row in rows])
                           for part in wf.PARTS} for name in wf.RULES + ('oracle',)},
        paired={pair: {key: rb.stats([row['paired'][pair][key] for row in rows]) for key in ('success', 'completion')}
                for pair in rows[0]['paired']},
        faster={pair: int(sum(row['paired'][pair]['completion'] < 0 for row in rows)) for pair in rows[0]['paired']},
        gap_closure={name: {part: rb.stats([row['gap_closure'][name][part] for row in rows
                                            if row['gap_closure'][name][part] is not None]) if any(
                                                row['gap_closure'][name][part] is not None for row in rows) else None
                            for part in wf.PARTS} for name in wf.ADAPTIVE})
    if 'feasible_calibration' in rows[0]:
        out['feasible_calibration'] = int(sum(row['feasible_calibration'] for row in rows))
    if 'target_reached' in rows[0]:
        out['target_reached_on_calibration'] = {name: int(sum(row['target_reached'][name] for row in rows)) for name in wf.RULES}
    return out


def collect(repeats, noise, path):
    """Cells at `path` (a tuple of keys) of every split for one noise level."""
    rows = []
    for r in range(rb.REPEATS):
        node = repeats[str(r)][noise]
        for key in path:
            node = node[key]
        rows.append(node)
    return rows


def summaries(repeats):
    out = {}
    for noise in (f'{n:g}' for n in NOISES):
        first = repeats['0'][noise]
        block = dict(baseline={t: summarize(collect(repeats, noise, ('baseline', t))) for t in first['baseline']},
                     baseline_delay={t: summarize(collect(repeats, noise, ('baseline_delay', t))) for t in first['baseline_delay']},
                     hidden_lag={}, visible_lag={}, slow_posterior_tail={})
        for scenario in ('hidden_lag', 'visible_lag'):
            for arm, lags in first[scenario].items():
                block[scenario][arm] = {lag: {t: summarize(collect(repeats, noise, (scenario, arm, lag, t))) for t in cells}
                                        for lag, cells in lags.items()}
        for cut, entry in first['slow_posterior_tail'].items():
            rows = collect(repeats, noise, ('slow_posterior_tail', cut))
            block['slow_posterior_tail'][cut] = dict(
                n_evaluation=rb.stats([row['n_evaluation'] for row in rows]),
                n_calibration=rb.stats([row['n_calibration'] for row in rows]),
                never_ready_evaluation=rb.stats([row['never_ready_evaluation'] for row in rows]),
                **{arm: {t: summarize([row[arm][t] for row in rows]) for t in entry[arm]}
                   for arm in ('model_calibrated', 'outcome_recalibrated', 'outcome_recalibrated_delay')})
        out[noise] = block
    return out


def statements(repeats, summary):
    s = summary[PRIMARY_NOISE]
    t = PRIMARY_TARGET
    tolerated = {}
    for name in wf.RULES:
        best = None
        for lag in MC_LAGS:
            if s['hidden_lag']['model_calibrated'][f'{lag}'][t]['meets_per_history'][name] >= ROBUST_SPLITS:
                best = lag
            else:
                break
        tolerated[name] = best
    m1 = dict(tolerated_lag_min=tolerated,
              success_after_long_median={f'{lag}': {name: s['hidden_lag']['model_calibrated'][f'{lag}'][t]['success'][name]['long']['median']
                                                    for name in wf.RULES + ('oracle',)} for lag in MC_LAGS},
              baseline_gain_over_fixed_min={name: s['baseline'][t]['paired'][f'{name}-fixed/all']['completion']['median']
                                            for name in ('history', 'smoothed')})
    m2 = {}
    for noise in (f'{n:g}' for n in NOISES):
        m2[noise] = {}
        for lag in map(str, VISIBLE_LAGS):
            d = []
            for r in range(rb.REPEATS):
                base = repeats[str(r)][noise]['baseline'][t]['rules']
                vis = repeats[str(r)][noise]['visible_lag']['model_calibrated'][lag][t]['rules']
                d.append((vis['history']['long']['success_rate'] - base['history']['long']['success_rate'])
                         - (vis['smoothed']['long']['success_rate'] - base['smoothed']['long']['success_rate']))
            positive, negative = int(sum(x > 0 for x in d)), int(sum(x < 0 for x in d))
            verdict = ('B3 belief more robust' if positive >= ROBUST_SPLITS else
                       'smoothing more robust' if negative >= ROBUST_SPLITS else 'no consistent difference')
            m2[noise][lag] = dict(D=rb.stats(d), positive=positive, negative=negative, verdict=verdict,
                                  ess_90=rb.stats([repeats[str(r)][noise]['ess']['visible_lag'][lag]['90']['median']
                                                   for r in range(rb.REPEATS)]),
                                  history_minus_smoothed_completion=summary[noise]['visible_lag']['model_calibrated'][lag][t]
                                  ['paired']['history-smoothed/all']['completion'])
    m3 = {}
    for scenario, lags in (('hidden_lag', RECAL_LAGS), ('visible_lag', VISIBLE_LAGS)):
        m3[scenario] = {}
        for arm in ('outcome_recalibrated', 'outcome_recalibrated_delay'):
            m3[scenario][arm] = {}
            for lag in map(str, lags):
                cells = [repeats[str(r)][PRIMARY_NOISE][scenario][arm][lag][t] for r in range(rb.REPEATS)]
                entry = dict(meets_per_history={name: int(sum(c['meets_per_history'][name] for c in cells)) for name in wf.RULES},
                             feasible_calibration=int(sum(c['feasible_calibration'] for c in cells)))
                for name in ('history', 'smoothed'):
                    both = [c for c in cells if c['target_reached'][name] and c['target_reached']['fixed']]
                    diffs = [c['paired'][f'{name}-fixed/all']['completion'] for c in both]
                    entry[f'{name}-fixed'] = dict(splits=len(both), faster=int(sum(x < 0 for x in diffs)),
                                                  median=float(np.median(diffs)) if diffs else None)
                m3[scenario][arm][lag] = entry
    m4 = {}
    for cut in (f'{c:g}' for c in ONSET_CUTS):
        m4[cut] = {}
        for arm in ('model_calibrated', 'outcome_recalibrated', 'outcome_recalibrated_delay'):
            cells = [repeats[str(r)][PRIMARY_NOISE]['slow_posterior_tail'][cut][arm][t] for r in range(rb.REPEATS)]
            entry = dict(meets_per_history={name: int(sum(c['meets_per_history'][name] for c in cells)) for name in wf.RULES},
                         oracle_success={hist: rb.stats([c['rules']['oracle'][hist]['success_rate'] for c in cells])
                                         for hist in br.HISTORIES})
            for name in ('history', 'smoothed', 'current'):
                for hist in br.HISTORIES:
                    diffs = [c['rules'][name][hist]['success_rate'] - c['rules']['fixed'][hist]['success_rate'] for c in cells]
                    entry[f'{name}-fixed/{hist}'] = dict(success_difference=rb.stats(diffs), positive=int(sum(x > 0 for x in diffs)))
            m4[cut][arm] = entry
    return dict(M1=m1, M2=m2, M3=m3, M4=m4)


def oracle_feasibility(success):
    plants = np.arange(success.shape[1])
    return {f'{lag}': rb.summarise(*wf.oracle_outcomes(lagged(success, lag), plants, br.DEADLINE_MIN)) for lag in MC_LAGS}


def median_ratio(tables, h):
    ratio = tables['rise'][h].astype(float) / tables['naive_rise'].astype(float)[:, None]
    return np.median(ratio, axis=0), ratio


def shift_equivalent(curve, measured, wait):
    """wait minus the time at which the rising branch of `curve` crosses `measured` (linear interpolation)."""
    k0 = int(np.flatnonzero(br.GRID == wait)[0])
    if curve[k0] > measured:
        k = int(np.flatnonzero(curve[:k0] <= measured)[-1])
    else:
        k = k0 + int(np.flatnonzero(curve[k0:] >= measured)[0]) - 1
    crossing = br.GRID[k] + (br.GRID[k + 1] - br.GRID[k]) * (measured - curve[k]) / (curve[k + 1] - curve[k])
    return float(crossing), float(wait - crossing)


def no_response_null():
    """Natural-probe rise statistic of the mixed probe under no response (flat reporter plus white noise)."""
    import expanded_fgf2 as e
    import natural_probe as npr
    probe = json.loads(NATURAL_PROBE.read_text(encoding='utf-8'))
    blocks, _ = e.load_blocks(e.load_protocol())
    times = np.asarray(blocks['fgf_mixed_25ng']['time'], dtype=float)
    sd = float(probe['noise_floor']['fgf_mixed_25ng'])
    naive = float(probe['naive_reference']['mixed_measured_median'])
    y = 1. + sd * np.random.default_rng(NULL_SEED).normal(0., 1., (NULL_DRAWS, times.size))
    ratio = npr.rise(times, y, npr.MIXED_ONSET, 1.) / naive
    q25, q50, q75 = map(float, np.percentile(ratio, [25, 50, 75]))
    measured = float(probe['probes'][PROBES['mixed'][0]]['rise_over_naive']['measured_median'])
    return dict(noise_sd=sd, naive_measured_median=naive, frames=int(times.size), draws=NULL_DRAWS, seed=NULL_SEED,
                q25=q25, median=q50, q75=q75, measured_median=measured, measured_inside_iqr=bool(q25 <= measured <= q75))


def descriptive(tables, success):
    probe = json.loads(NATURAL_PROBE.read_text(encoding='utf-8'))['probes']
    shifts = {}
    for name, (key, h, wait) in PROBES.items():
        curve, _ = median_ratio(tables, h)
        measured = float(probe[key]['rise_over_naive']['measured_median'])
        crossing, shift = shift_equivalent(curve, measured, wait)
        shifts[name] = dict(probe=key, history=br.HISTORIES[h], wait_min=wait, measured_median=measured,
                            b3_median_at_wait=float(curve[int(np.flatnonzero(br.GRID == wait)[0])]),
                            crossing_min=crossing, shift_equivalent_min=shift)
    _, ratio = median_ratio(tables, LONG)
    k60 = int(np.flatnonzero(br.GRID == 60.)[0])
    return dict(oracle_feasibility=oracle_feasibility(success), shift_equivalents=shifts,
                long_ratio_at_60=dict(minimum=float(ratio[:, k60].min()), q01=float(np.quantile(ratio[:, k60], .01))),
                no_response_null=no_response_null())


def main():
    load_protocol()
    tables = dict(np.load(br.TABLES))
    success = br.success_table(tables)
    jobs = [(r, noise) for r in range(rb.REPEATS) for noise in NOISES]
    workers = int(os.environ.get('B3_WORKERS', min(32, os.cpu_count() or 1)))   # runtime only, not part of the protocol
    results = run_jobs(_job, jobs, workers)
    repeats = {}
    for (r, noise), result in zip(jobs, results):
        repeats.setdefault(str(r), {})[f'{noise:g}'] = result
    summary = summaries(repeats)
    SPLITS.write_text(json.dumps(dict(protocol_sha256=protocol_sha256(), repeats=repeats), separators=(',', ':')) + '\n',
                      newline='\n')
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  splits_file=SPLITS.name, splits_sha256=hashlib.sha256(SPLITS.read_bytes()).hexdigest(),
                  slow_subsets={f'{c:g}': int(slow_mask(success, c).sum()) for c in ONSET_CUTS},
                  summary=summary, statements=statements(repeats, summary),
                  descriptive=descriptive(tables, success),
                  limitations=['Simulation of one published model structure with stylized recovery errors; not evidence about measured cells.',
                               'No rule is model-free: all are calibrated on B3 outcomes; only the outcome-recalibrated fixed wait uses no B3 output.',
                               'Recalibrated arms use complete outcome curves of all calibration receivers and a common lag (an upper bound).',
                               'Repeats redraw the split and observation noise only; one finite posterior set is reused.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    st = result['statements']
    print('M1 tolerated lag (0.005, 0.94):', st['M1']['tolerated_lag_min'], '| baseline gain', st['M1']['baseline_gain_over_fixed_min'])
    print('M2:', json.dumps({n: {l: (v['verdict'], v['positive'], v['negative'], round(v['ess_90']['median'], 1)) for l, v in b.items()}
                             for n, b in st['M2'].items()}))
    print('M3:', json.dumps(st['M3'])[:3000])
    print('M4:', json.dumps(st['M4'])[:2000])
    print('descriptive:', json.dumps({k: v for k, v in result['descriptive'].items() if k != 'oracle_feasibility'}))


if __name__ == '__main__':
    main()
