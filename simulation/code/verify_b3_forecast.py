"""Audit the B3 matched-feedback forecaster and the M0-M3 comparison without retuning.

Checks the frozen protocol hash, that feedback weights use only past
observations, that equal weights reduce M2 to the anchored forecast, and
recomputes every saved score from the saved sigma and ridge coefficients.
"""
import json
import sys

import numpy as np

import b3_forecast as bf


def weight_checks():
    rng = np.random.default_rng(0)
    trajectories = 1 + .05 * rng.random((50, 40))
    observed = 1 + .05 * rng.random(40)
    weights = bf.feedback_weights(observed, trajectories, .02)
    assert np.allclose(weights.sum(0), 1.)
    changed = observed.copy()
    changed[20:] += 1.
    later = bf.feedback_weights(changed, trajectories, .02)
    assert np.allclose(weights[:, :20], later[:, :20]), 'weights used future observations'
    assert not np.allclose(weights[:, 20:], later[:, 20:])
    assert np.allclose(bf.feedback_weights(observed, trajectories, 'equal_weights'), 1 / 50)
    return dict(normalized=True, past_only=True, equal_weights_uniform=True)


def saved_score_check():
    saved = json.loads(bf.RESULTS.read_text(encoding='utf-8'))
    assert saved['protocol_sha256'] == bf.protocol_sha256()
    recomputed = bf.score_saved(saved)
    checked = 0
    for horizon, roles in saved['scores'].items():
        for role, models in roles.items():
            for name, metric in models.items():
                np.testing.assert_allclose(recomputed[horizon][role][name], metric['run_equal_mse'], rtol=1e-9, atol=1e-14)
                checked += 1
    return dict(role_model_scores_recomputed=checked)


def main():
    bf.load_protocol()
    report = dict(status='passed', protocol_sha256=bf.protocol_sha256(), weights=weight_checks(),
                  saved_scores=saved_score_check(),
                  scope='Post hoc reference comparison on inspected roles; sigma and alpha were chosen on validation.')
    (bf.OUT / 'forecast_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
