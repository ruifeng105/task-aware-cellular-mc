"""No-probe control and causal rescoring of the B3 waiting rules at kappa 0.4, 0.5 and 0.6.

Implements configs/b3_noprobe_kappa_protocol.json (review item P0-3), frozen
before its only run. Reuses the frozen no-probe arrays of b3_noprobe without
new simulation: for each kappa it counts states that meet the criterion
without a probe, compares the original labels with causal (probe-induced)
labels over all 122,000 states and at the stop states of the frozen rules,
and rescores those rules. Simulation of one published model structure.
"""
import hashlib
import json

import numpy as np

import b3_noprobe as npb
import b3_readiness as br
import b3_waiting as bw
import b3_waiting_baselines as bl
import b3_waiting_sensitivity as sv

PROTOCOL = br.ROOT / 'configs/b3_noprobe_kappa_protocol.json'
FREEZE = br.OUT / 'b3_noprobe_kappa_protocol_freeze.json'
RESULTS = br.OUT / 'noprobe_kappa_results.json'
RULES = ('current', 'smoothed', 'history', 'fixed', 'conditioned', 'oracle')


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    npb.load_protocol()
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def labels(tables, arrays, kappa):
    """Original, without-probe and causal success labels, arrays (histories, samples, waits)."""
    threshold = kappa * tables['naive_rise'].astype(float)[None, :, None]
    return (br.success_table(sv.with_kappa(tables, kappa)), arrays['rise_without_probe'] >= threshold,
            arrays['probe_increment'] >= threshold)


def state_summary(original, without, causal):
    per_history = lambda a: {name: float(a[h].mean()) for h, name in enumerate(br.HISTORIES)}
    return dict(states=int(original.size), success_without_probe=dict(fraction=float(without.mean()), count=int(without.sum()),
                                                                     per_history=per_history(without),
                                                                     fraction_of_original_successes=float(without[original].mean())),
                original_success_fraction=float(original.mean()), causal_success_fraction=float(causal.mean()),
                agreement=dict(both=int((original & causal).sum()), original_only=int((original & ~causal).sum()),
                               causal_only=int((~original & causal).sum()), neither=int((~original & ~causal).sum()),
                               fraction_agree=float((original == causal).mean())))


def stop_indices(tables, kappa, noise):
    """Stop-time indices (histories, evaluation samples) of the frozen kappa rules, as in the sensitivity study."""
    choices = json.loads(sv.RESULTS.read_text(encoding='utf-8'))['cells'][f'{kappa:g}'][f'{noise:g}']['choices']
    conditioned = json.loads(bl.RESULTS.read_text(encoding='utf-8'))['kappa'][f'{kappa:g}']['waits_min']
    scaled = sv.with_kappa(tables, kappa)
    success = br.success_table(scaled)
    cal, ev = bw.split_samples(success.shape[1])
    probability = bw.estimator_probabilities(scaled, success, noise, cal, ev, bw.noisy_observations(tables, noise))
    crossing = lambda p, threshold: np.apply_along_axis(br.first_crossing, -1, p, threshold)
    index = lambda wait: int(np.flatnonzero(br.GRID == wait)[0])
    k = {name: crossing(probability[name], choices[name]['threshold']) for name in ('current', 'history')}
    k['smoothed'] = crossing(probability[f"smoothed_{choices['smoothed']['tau_min']:g}"], choices['smoothed']['threshold'])
    k['fixed'] = np.full((success.shape[0], len(ev)), index(choices['fixed']['wait_min']))
    k['conditioned'] = np.stack([np.full(len(ev), index(w)) for w in conditioned])
    k['oracle'] = crossing(success[:, ev, :].astype(float), 1.)
    return k, ev


def rescore(tables, arrays, kappa, noise):
    original, _, causal = labels(tables, arrays, kappa)
    k, ev = stop_indices(tables, kappa, noise)
    out = {'original': {}, 'causal': {}, 'disagreement_at_stop': {}}
    for name in RULES:
        at = {}
        for label, table in (('original', original), ('causal', causal)):
            ok = np.take_along_axis(table[:, ev, :], k[name][..., None], -1)[..., 0]
            completion = np.where(ok, br.GRID[k[name]] + br.WINDOW_MIN, br.DEADLINE_MIN)
            out[label][name] = dict(success_rate=float(ok.mean()), mean_completion_min=float(completion.mean()))
            at[label] = ok
        out['disagreement_at_stop'][name] = float((at['original'] != at['causal']).mean())
    order = lambda label: sorted((n for n in RULES if n != 'oracle'), key=lambda n: out[label][n]['mean_completion_min'])
    out['order_by_completion'] = dict(original=order('original'), causal=order('causal'))
    out['order_unchanged'] = out['order_by_completion']['original'] == out['order_by_completion']['causal']
    return out


def main():
    load_protocol()
    tables = dict(np.load(br.TABLES))
    arrays = dict(np.load(npb.ARRAYS))
    naive = tables['naive_rise'].astype(float)[None, :, None]
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'],
                  max_rise_without_probe_over_naive=float((arrays['rise_without_probe'] / naive).max()),
                  kappa={f'{kappa:g}': dict(states=state_summary(*labels(tables, arrays, kappa)),
                                            rescored={f'{noise:g}': rescore(tables, arrays, kappa, noise) for noise in sv.NOISES})
                         for kappa in sv.KAPPAS},
                  limitations=['Simulation of one published model structure; not evidence about measured cells.',
                               'The original task definition remains the primary analysis.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print('max no-probe rise / naive', round(result['max_rise_without_probe_over_naive'], 4))
    for kappa, block in result['kappa'].items():
        s = block['states']
        print(kappa, 'without probe', s['success_without_probe']['count'], round(100 * s['success_without_probe']['fraction'], 3),
              'agree', round(100 * s['agreement']['fraction_agree'], 2), s['agreement'])
        for noise, r in block['rescored'].items():
            print('  ', noise, {n: (round(100 * r['original'][n]['success_rate'], 1), round(100 * r['causal'][n]['success_rate'], 1),
                                    round(r['original'][n]['mean_completion_min'], 1), round(r['causal'][n]['mean_completion_min'], 1),
                                    round(100 * r['disagreement_at_stop'][n], 2)) for n in RULES},
                  'order unchanged', r['order_unchanged'])


if __name__ == '__main__':
    main()
