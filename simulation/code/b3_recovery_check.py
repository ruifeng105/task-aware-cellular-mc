"""Posterior predictive check of B3 recovery and diagnosis of weight collapse on measured cells.

Implements configs/b3_recovery_check_protocol.json, frozen after an exploratory
analysis (disclosed in the protocol) and run once. (1) Probe rise over naive
rise after a 30-min pulse (mixed protocol, probe 60 min later) and after a
3-min pulse (3/20 protocol, 20 min later) at 2.5, 25 and 250 ng/ml: measured
median with a cell bootstrap interval against B3's posterior predictive
distribution, and a no-response null. (2) Why the B3 weights collapse: per
measured cell, the residual to the best posterior sample (raw, after a per-cell
offset, after a per-cell affine map), the resulting effective sample size, the
residual autocorrelation and the posterior's own reporter spread. Descriptive;
nothing here re-fits B3.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import b3_forecast as bf
import expanded_fgf2 as e
import natural_probe as npb
import nested_extensions as nx

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PROTOCOL = ROOT / 'configs/b3_recovery_check_protocol.json'
OUT = ROOT / 'results/b3'
FREEZE = OUT / 'b3_recovery_check_protocol_freeze.json'
RESULTS = OUT / 'recovery_check_results.json'
DOSES = ('2-5ng', '25ng', '250ng')
FITTED = ('2-5ng', '250ng')
BOOT, BOOT_SEED, NULL_DRAWS, NULL_SEED = 2000, 20261012, 20000, 20261013
MIXED_ONSET, THREE_TWENTY_ONSET = npb.MIXED_ONSET, npb.PULSES_3_20[1]
WINDOWS = {'mixed': MIXED_ONSET, '3_20': THREE_TWENTY_ONSET, 'sus': 60.}


def protocol_sha256():
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def load_protocol():
    if protocol_sha256() != json.loads(FREEZE.read_text(encoding='utf-8'))['protocol_sha256']:
        raise SystemExit('Protocol differs from its freeze record; refusing to run.')
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    for rel, digest in protocol['inputs_sha256'].items():
        if hashlib.sha256((REPO / rel).read_bytes()).hexdigest() != digest:
            raise SystemExit(f'Pinned input changed: {rel}')
    return protocol


def keys():
    return ([f'fgf_mixed_{d}' for d in DOSES] + [f'fgf_3_20_{d}' for d in DOSES] + [f'fgf_sp_5_{d}' for d in DOSES]
            + [f'fgf_sus_{d}' for d in FITTED])


def load():
    blocks, _ = e.load_blocks(e.load_protocol())
    blocks = {k: blocks[k] for k in keys()}
    population = bf.simulate_population(blocks)
    data = {}
    for k in blocks:
        t = np.asarray(blocks[k]['time'], dtype=float)
        y = np.asarray(blocks[k]['y'], dtype=float)
        data[k] = dict(t=t, y=y[np.isfinite(y).all(1)], f=population[k][:, t.astype(int)], start=population[k][:, 0])
    return data, nx.noise_floor(blocks)


def quantiles(values):
    q = np.percentile(values, [5, 50, 95])
    return dict(q05=float(q[0]), median=float(q[1]), q95=float(q[2]))


def null_ratio(times, onset, sd, denominator, rng):
    noise = 1. + rng.normal(0., sd, (NULL_DRAWS, times.size))
    ratio = npb.rise(times, noise, onset, 1.) / denominator
    q = np.percentile(ratio, [25, 50, 75])
    return dict(q25=float(q[0]), median=float(q[1]), q75=float(q[2]))


def mixed_check(data, floor, dose, rng, null_rng):
    m, p = data[f'fgf_mixed_{dose}'], data[f'fgf_sp_5_{dose}']
    probe = npb.rise(m['t'], m['y'], MIXED_ONSET, 1.)
    naive = npb.rise(p['t'], p['y'], 0., 1.)
    measured = float(np.median(probe) / np.median(naive))
    boot = [np.median(rng.choice(probe, probe.size)) / np.median(rng.choice(naive, naive.size)) for _ in range(BOOT)]
    b3 = npb.rise(m['t'], m['f'], MIXED_ONSET, m['start']) / npb.rise(p['t'], p['f'], 0., p['start'])
    return dict(cells=int(probe.size), naive_cells=int(naive.size), measured=measured,
                interval95=[float(v) for v in np.percentile(boot, [2.5, 97.5])], b3=quantiles(b3),
                b3_share_at_or_below=float(np.mean(b3 <= measured)),
                null=null_ratio(m['t'], MIXED_ONSET, floor[f'fgf_mixed_{dose}'], float(np.median(naive)), null_rng))


def three_twenty_check(data, floor, dose, rng, null_rng):
    s = data[f'fgf_3_20_{dose}']
    first = npb.rise(s['t'], s['y'], 0., 1.)
    second = npb.rise(s['t'], s['y'], THREE_TWENTY_ONSET, 1.)
    ratio = second / first
    measured = float(np.median(ratio))
    boot = [np.median(rng.choice(ratio, ratio.size)) for _ in range(BOOT)]
    b3 = npb.rise(s['t'], s['f'], THREE_TWENTY_ONSET, s['start']) / npb.rise(s['t'], s['f'], 0., s['start'])
    return dict(cells=int(ratio.size), measured=measured, interval95=[float(v) for v in np.percentile(boot, [2.5, 97.5])],
                b3=quantiles(b3), b3_share_at_or_below=float(np.mean(b3 <= measured)),
                null=null_ratio(s['t'], THREE_TWENTY_ONSET, floor[f'fgf_3_20_{dose}'], float(np.median(first)), null_rng))


def residual_sums(y, f):
    """Residual sums of squares (cells x samples): raw, after a per-cell offset, after a per-cell affine response map."""
    dev = y[:, None, :] - f[None]
    raw = (dev ** 2).sum(-1)
    offset = ((dev - dev.mean(-1, keepdims=True)) ** 2).sum(-1)
    x, z = f - 1., y - 1.
    xc = x - x.mean(-1, keepdims=True)
    zc = z - z.mean(-1, keepdims=True)
    sxx = (xc ** 2).sum(-1)
    sxz = zc @ xc.T
    b = sxz / np.maximum(sxx[None], 1e-300)
    affine = (zc ** 2).sum(-1)[:, None] - b * sxz
    return dict(raw=raw, offset=offset, affine=np.maximum(affine, 0.))


def ess(rss, sd):
    log_w = -rss / (2 * sd ** 2)
    log_w -= log_w.max(1, keepdims=True)
    w = np.exp(log_w)
    w /= w.sum(1, keepdims=True)
    return 1. / (w ** 2).sum(1)


def collapse_check(data, floor, key, onset):
    d = data[key]
    i = npb.last_index(d['t'], onset)
    y, f = d['y'][:, :i + 1], d['f'][:, :i + 1]
    frames = y.shape[1]
    sd = max(floor[key], npb.SD_FLOOR)
    sums = residual_sums(y, f)
    best = sums['raw'].argmin(1)
    residual = y - f[best]
    lag1 = [np.corrcoef(r[:-1], r[1:])[0, 1] for r in residual]
    out = dict(cells=int(y.shape[0]), frames=int(frames), noise_floor=float(floor[key]), likelihood_sd=float(sd),
               posterior_spread=float(np.median(f.std(0))), best_residual_lag1_autocorrelation=float(np.median(lag1)))
    for name, rss in sums.items():
        out[name] = dict(best_residual_over_noise=float(np.median(np.sqrt(rss.min(1) / frames)) / floor[key]),
                         ess=float(np.median(ess(rss, sd))))
    return out


def reading(result):
    mixed, tt = result['mixed'], result['three_twenty']

    def outside(block):
        lo, hi = block['interval95']
        return bool(hi < block['b3']['q05'] or lo > block['b3']['q95'])

    r1 = {f'mixed/{d}': outside(mixed[d]) for d in DOSES} | {f'three_twenty/{d}': outside(tt[d]) for d in DOSES}
    r2 = bool(r1['mixed/25ng'] and r1['mixed/250ng'] and not r1['mixed/2-5ng']
              and not any(r1[f'three_twenty/{d}'] for d in FITTED))
    r3 = {}
    for label, block in (('mixed', mixed), ('three_twenty', tt)):
        for d in DOSES:
            r3[f'{label}/{d}'] = bool(block[d]['null']['q25'] <= block[d]['measured'] <= block[d]['null']['q75'])
    fitted = [result['collapse'][k] for k in result['collapse'] if k.split('_')[-1] in FITTED and '_mixed_' not in k]
    r4 = bool(all(c['raw']['ess'] <= 3 and c['affine']['best_residual_over_noise'] > 2 and c['posterior_spread'] < c['noise_floor']
                  for c in fitted))
    return dict(R1_inconsistent=r1, R2_localized_to_long_strong_commands=r2, R3_indistinguishable_from_no_response=r3,
                R4_collapse_from_unrepresented_cell_deviations=r4, fitted_conditions=[k for k in result['collapse']
                                                                                     if k.split('_')[-1] in FITTED and '_mixed_' not in k])


def compute():
    data, floor = load()
    rng, null_rng = np.random.default_rng(BOOT_SEED), np.random.default_rng(NULL_SEED)
    result = dict(mixed={d: mixed_check(data, floor, d, rng, null_rng) for d in DOSES},
                  three_twenty={d: three_twenty_check(data, floor, d, rng, null_rng) for d in DOSES},
                  collapse={})
    for kind, onset in WINDOWS.items():
        for d in (FITTED if kind == 'sus' else DOSES):
            key = f'fgf_{kind}_{d}'
            result['collapse'][key] = collapse_check(data, floor, key, onset)
    result['reading'] = reading(result)
    return json.loads(json.dumps(result))


def main():
    load_protocol()
    result = dict(status='complete', protocol_sha256=protocol_sha256(),
                  frozen_utc=json.loads(FREEZE.read_text(encoding='utf-8'))['frozen_utc'], **compute(),
                  limitations=['Descriptive; cells of one condition share a session; no B3 parameter is re-fitted.',
                               'The mixed naive reference comes from different cells (single 5-min pulse).',
                               'Mixed 2.5 and 250 ng/ml are inspected data; the protocol was frozen after exploration.'])
    RESULTS.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    for name in ('mixed', 'three_twenty'):
        for d in DOSES:
            b = result[name][d]
            print(f"{name:12s} {d:6s} measured {b['measured']:.3f} [{b['interval95'][0]:.3f}, {b['interval95'][1]:.3f}] "
                  f"B3 {b['b3']['q05']:.3f}/{b['b3']['median']:.3f}/{b['b3']['q95']:.3f} share<= {b['b3_share_at_or_below']:.3f} "
                  f"null {b['null']['median']:.3f} ({b['null']['q25']:.3f}-{b['null']['q75']:.3f})")
    for key, c in result['collapse'].items():
        print(f"{key:18s} spread {c['posterior_spread']:.4f} noise {c['noise_floor']:.4f} lag1 {c['best_residual_lag1_autocorrelation']:.2f} "
              + ' '.join(f"{n} {c[n]['best_residual_over_noise']:.1f}x ESS {c[n]['ess']:.1f}" for n in ('raw', 'offset', 'affine')))
    print(json.dumps(result['reading'], indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
