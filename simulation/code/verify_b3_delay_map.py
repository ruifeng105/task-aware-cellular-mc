"""Audit the delay / sampling-interval / noise condition map.

Checks that at d = 0 with 2-min sampling the B3 rules equal the per-history
history rule of the frontier study and the fixed rule equals its per-history
fixed waits, that estimates are causal (perturbing later observations leaves
earlier estimates unchanged), the binned kernel against direct evaluation,
that choices ignore evaluation labels, and recomputes one job from scratch.
"""
import json
import sys

import numpy as np

import b3_delay_map as dm
import b3_readiness as br
import b3_waiting as bw
import b3_waiting_frontier as wf
import b3_waiting_robustness as rb

VERIFY = br.OUT / 'delay_map_verification_results.json'


def frontier_check(saved):
    frontier = json.loads(wf.RESULTS.read_text(encoding='utf-8'))['repeats']
    worst = 0.
    for noise in ('0.005', '0.01'):
        for r in range(rb.REPEATS):
            mine = saved['splits'][f'{noise}/2/0'][str(r)]['rules']
            theirs = frontier[str(r)][noise]['cells']['0.90']['rules']
            for name, other in (('b3_predictive', 'history'), ('b3_nowcast', 'history'), ('fixed', 'fixed')):
                for part, row in mine[name].items():
                    for key, value in row.items():
                        worst = max(worst, abs(value - theirs[other][part][key]))
    assert worst < 1e-12, worst
    return dict(cells=['0.005/2/0', '0.01/2/0'], splits=rb.REPEATS, max_abs_difference=worst)


def causality_check():
    rng = np.random.default_rng(7)
    observed = 1 + .05 * rng.random((4, 21))
    references = 1 + .05 * rng.random((9, 21))
    later = observed.copy()
    later[:, 11:] += rng.normal(0, .02, (4, 10))
    exclude = np.array([-1, 2, -1, 5])
    w0, w1 = (dm.b3_weights(x, references, .005, exclude) for x in (observed, later))
    assert np.allclose(w0[..., :11], w1[..., :11]) and not np.allclose(w0[..., 11:], w1[..., 11:])
    assert np.all(w0[1, 2] == 0) and np.all(w0[3, 5] == 0)
    l0, g0 = dm.ema_increment(observed, .6)
    l1, g1 = dm.ema_increment(later, .6)
    assert np.allclose(l0[:, :11], l1[:, :11]) and np.allclose(g0[:, :11], g1[:, :11])
    probability = np.linspace(0, 1, 6)[None, None, :] * np.ones((2, 3, 1))
    instants = dm.instant_index(10)[:6]
    k = dm.threshold_arrivals(probability, instants, 20, [.5, 1.01])
    assert (k[0] == instants[3] + 10).all() and (k[1] == br.GRID.size - 1).all()
    return dict(weights_and_smoothing_causal=True, self_excluded=True, arrival_includes_delay=True)


def kernel_check(tables):
    success = br.success_table(tables)
    cal, ev = bw.split_samples(success.shape[1])
    fret = tables['history_fret'].astype(float)
    worst = 0.
    for noise, interval, scale, d in ((.0025, 2, 16., 20), (.01, 10, 2., 0), (.005, 6, 8., 40)):
        observed = bw.noisy_observations(tables, noise, seed=123)
        instants = dm.instant_index(interval)
        usable = dm.valid(instants, d)
        cols = np.arange(usable.size)
        a = np.exp(-interval / scale)
        whitener = dm.kernel_whitener(noise, a)
        rl, ri = dm.ema_increment(fret[:, cal][:, :, instants], a)
        ol, oi = dm.ema_increment(observed[:, ev[:20]][:, :, instants], a)
        states = np.stack([rl[..., cols], ri[..., cols]], -1).reshape(-1, 2)
        labels = success[:, cal][:, :, usable + d // 2].astype(float).reshape(-1)
        query = np.stack([ol, oi], -1)[:, :, cols]
        binned = dm.smoothed_slope_probabilities(states, labels, query, None, None, whitener)
        qw, sw = (query @ whitener.T).reshape(-1, 2), states @ whitener.T
        kernel = np.exp(-((qw[:, None, :] - sw[None]) ** 2).sum(-1) / 2)
        s0, s1 = kernel.sum(1), kernel @ labels
        exact = np.where(s0 > 1e-9, s1 / np.maximum(s0, 1e-300), 0.).reshape(binned.shape)
        worst = max(worst, float(np.abs(binned - exact).max()))
    assert worst < .01, worst
    return dict(max_abs_probability_difference=worst, tolerance=.01)


def leakage_check(tables, noise=.005, interval=6):
    success = br.success_table(tables)
    cal, ev = bw.split_samples(success.shape[1])
    observed = bw.noisy_observations(tables, noise)
    flipped = success.copy()
    flipped[:, ev, :] = ~flipped[:, ev, :]
    choices = []
    for table in (success, flipped):
        options = dm.rule_options(tables, table, observed, noise, interval, cal, cal)
        choices.append({d: {n: [wf.choose(options[d][n], h, dm.TARGET) for h in range(2)] for n in dm.RULES}
                        for d in dm.DELAYS})
    assert choices[0] == choices[1]
    return dict(evaluation_labels_flipped=True, choices_unchanged=True)


def recompute_check(saved, r=5, noise=.01, interval=10):
    fresh = dm.run_job(r, noise, interval)
    worst = 0.
    for d, cell in fresh['cells'].items():
        stored = saved['splits'][dm.cell_key(noise, interval, int(d))][str(r)]
        assert cell['choices'] == stored['choices'], d
        for name, parts in cell['rules'].items():
            for part, row in parts.items():
                for key, value in row.items():
                    worst = max(worst, abs(value - stored['rules'][name][part][key]))
    assert worst < 1e-12, worst
    return dict(repeat=r, noise=noise, interval=interval, max_abs_difference=worst)


def main():
    dm.load_protocol()
    tables = dict(np.load(br.TABLES))
    saved = json.loads(dm.RESULTS.read_text(encoding='utf-8'))
    checks = dict(frontier=frontier_check(saved), causality=causality_check(), kernel=kernel_check(tables),
                  leakage=leakage_check(tables), recompute=recompute_check(saved))
    result = dict(status='passed', protocol_sha256=dm.protocol_sha256(), checks=checks)
    VERIFY.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
