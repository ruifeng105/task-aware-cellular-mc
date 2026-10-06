"""Audit the low-noise degeneracy and mismatch study without changing any setting.

Checks that noise levels 0 and 0.005 reproduce version 3, the AR(1) noise
generator (marginal s.d. and lag-1 correlation), the offset observations,
and recomputes one mismatch evaluation from the frozen version-3 choices.
"""
import json
import sys

import numpy as np

import b3_readiness as br
import b3_waiting as bw
import b3_waiting_stress as st


def version3_check(saved):
    v3 = json.loads(bw.RESULTS.read_text(encoding='utf-8'))['noise']
    worst = 0.
    for noise in ('0', '0.005'):
        cell = saved['noise_grid'][noise]
        assert cell['choices'] == v3[noise]['choices'], noise
        for rule in ('current', 'smoothed', 'history', 'fixed_calibrated', 'oracle'):
            for key in ('success_rate', 'mean_completion_min'):
                worst = max(worst, abs(cell['evaluation'][rule][key] - v3[noise]['evaluation'][rule][key]))
    assert worst < 1e-12, worst
    return dict(max_abs_difference=worst)


def generator_check(tables):
    fret = tables['history_fret'].astype(float)
    noise = st.ar1_observations(tables) - fret
    lag1 = float(np.corrcoef(noise[..., 1:].ravel(), noise[..., :-1].ravel())[0, 1])
    sd = float(noise.std())
    assert abs(sd - .005) < 2e-4 and abs(lag1 - st.RHO) < .02, (sd, lag1)
    offset = st.offset_observations(tables) - bw.noisy_observations(tables, .005)
    assert np.allclose(offset, st.OFFSET, atol=1e-12)
    return dict(ar1_sd=sd, ar1_lag1=lag1, offset=st.OFFSET)


def recompute_check(tables, saved):
    v3 = json.loads(bw.RESULTS.read_text(encoding='utf-8'))['noise']['0.005']['choices']
    fresh = bw.evaluate(tables, .005, v3, observed=st.ar1_observations(tables))
    worst = max(abs(fresh[r][k] - saved['mismatch']['ar1'][r][k])
                for r in ('current', 'smoothed', 'history', 'fixed_calibrated', 'oracle') for k in ('success_rate', 'mean_completion_min'))
    assert worst < 1e-12, worst
    return dict(mismatch='ar1', max_abs_difference=worst)


def main():
    st.load_protocol()
    tables = dict(np.load(br.TABLES))
    saved = json.loads(st.RESULTS.read_text(encoding='utf-8'))
    report = dict(status='passed', protocol_sha256=st.protocol_sha256(), version3=version3_check(saved),
                  generators=generator_check(tables), recomputation=recompute_check(tables, saved),
                  scope='Simulation of the published B3 model; not evidence about measured cells.')
    (br.OUT / 'waiting_stress_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
