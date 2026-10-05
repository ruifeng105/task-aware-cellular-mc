"""Verify the Python execution of the authors' B3 model files.

Checks source identity, the parsed structure, conservation laws, the
unstimulated steady state, and the acceptance gate fixed before the first
run: the posterior-predictive mean of our simulation must match the
authors' exported sim_post means for the mixed protocol within 1e-3.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import b3_model as b3
import calibrate_fgf2 as c

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'public_data/fgf2_author_repository'
MANIFEST = ROOT / 'public_data/fgf2_b3_source_manifest.json'
OUT = ROOT / 'results/b3'
ACCEPTANCE_MAX_ABS = 1e-3
CONSERVED = (('Ras', 'Ras_star'), ('Raf', 'Raf_star'), ('Mek', 'Mek_star'), ('Erk', 'Erk_star'),
             ('Nfb', 'Nfb_star'), ('FgfR', 'FgfR_star', 'H_F_R'), ('H', 'H_F', 'H_F_R'))


def verify_sources():
    entries = [m for m in json.loads(MANIFEST.read_text(encoding='utf-8')) if 'path' in m]
    for item in entries:
        raw = (SOURCE / item['path']).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == item['blob_sha'], item['path']
    return len(entries)


def structure_check(model):
    assert len(model.species) == 15 and len(model.reactions) == 16, (len(model.species), len(model.reactions))
    assert len(model.parameter_names) == 34, len(model.parameter_names)
    return dict(species=len(model.species), reactions=len(model.reactions), free_parameters=len(model.parameter_names))


def conservation_check(model, posterior):
    t = np.arange(0., 181., 2.)
    worst = 0.
    for theta in posterior[:20]:
        states = model.simulate_states(theta, 'fgf_mixed', 250., t)
        for group in CONSERVED:
            total = sum(states[:, model.species.index(name)] for name in group)
            worst = max(worst, float(np.abs(total - total[0]).max()))
    assert worst < 1e-6, worst
    return dict(samples_checked=20, max_conservation_error=worst)


def unstimulated_check(model, posterior):
    t = np.arange(0., 181., 2.)
    worst = max(float(np.abs(model.simulate_fret(theta, 'fgf_sus', 0., t) - 1.).max()) for theta in posterior[:20])
    assert worst < 1e-9, worst
    return dict(samples_checked=20, max_deviation_from_baseline=worst)


def schedule_check(model, posterior):
    """Explicit schedules reproduce named protocols; a pre-origin pulse leaves t=0 stimulated."""
    t = np.arange(0., 181., 2.)
    named = model.simulate_fret(posterior[0], 'fgf_mixed', 25., t)
    explicit = model.fret_from_states(posterior[0], model.simulate_schedule_states(
        posterior[0], c.segments('fgf_mixed'), 25., t))
    assert np.array_equal(named, explicit)
    primed = model.fret_from_states(posterior[0], model.simulate_schedule_states(
        posterior[0], [(-30., 0.)], 25., np.array([0., 2.])))
    assert abs(primed[0] - 1.) > 1e-3, primed
    return dict(explicit_equals_named=True, pre_origin_history_applied=True)


def acceptance_check(model, posterior):
    prefix = SOURCE / 'Inference_results/Fgf_B3/results/sus_3_20'
    t = c.load_matrix(prefix / 'sim_post_times.txt').ravel()
    result = {}
    for token, concentration in c.CONCS.items():
        exported = c.load_matrix(prefix / f'sim_post_mixed_{token}_measurements.txt')
        ours = model.simulate_fret_batch(posterior, 'fgf_mixed', concentration, t)
        difference = float(np.abs(ours.mean(0) - exported.mean(0)).max())
        sample_rmse = np.sqrt(((ours - exported) ** 2).mean(1))
        result[token] = dict(max_abs_mean_difference=difference,
                             median_samplewise_rmse=float(np.median(sample_rmse)),
                             n_samples=len(posterior), passes=difference <= ACCEPTANCE_MAX_ABS)
    assert all(r['passes'] for r in result.values()), result
    return dict(gate_max_abs=ACCEPTANCE_MAX_ABS, conditions=result)


def main():
    model = b3.load_model()
    posterior = b3.load_posterior()
    report = dict(status='passed', b3_source_git_blobs_verified=verify_sources(),
                  structure=structure_check(model),
                  conservation=conservation_check(model, posterior),
                  unstimulated=unstimulated_check(model, posterior),
                  schedules=schedule_check(model, posterior),
                  acceptance=acceptance_check(model, posterior),
                  scope='Execution of the original-author B3 files; no parameter is refit here.')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
