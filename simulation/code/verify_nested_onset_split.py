"""Audit the split of the next-command forecast errors by the following pulse.

Implements the checks of configs/nested_onset_split_protocol.json: onset
assignment on the mixed schedule, exact origins and row counts per group,
recombination to the frozen phase scores, shares summing to one, and a full
recomputation from the frozen models.
"""
import json
import sys

import nested_onset_split as ns


def onset_check():
    expected = {14.: 24., 22.: 24., 24.: None, 104.: 114., 112.: 114., 113.: 114., 114.: None, 30.: None}
    for origin, onset in expected.items():
        assert ns.following_onset(origin, ns.HORIZON, 'fgf_mixed') == onset, (origin, onset)
    return dict(cases=len(expected))


def structure_check(saved):
    for onset, (group, origins) in ns.GROUPS.items():
        for key, row in saved['groups'][group]['conditions'].items():
            assert row['origins'] == list(origins) and row['rows'] == len(origins) * row['cells'], (group, key)
    for key, share in saved['share_of_increase'].items():
        if None not in share.values():
            assert abs(sum(share.values()) - 1) < 1e-12, key
    return dict(origins_and_rows=True, shares_sum_to_one=True)


def recompute_check(saved):
    fresh = json.loads(json.dumps(ns.compute()))
    assert fresh == saved['groups']
    return dict(identical=True)


def main():
    ns.load_protocol()
    saved = json.loads(ns.RESULTS.read_text(encoding='utf-8'))
    report = dict(status='passed', protocol_sha256=ns.protocol_sha256(), onsets=onset_check(),
                  structure=structure_check(saved), combined_max_abs_difference=ns.combined_check(saved['groups']),
                  recomputation=recompute_check(saved), scope='Descriptive breakdown of frozen models on inspected roles.')
    (ns.nf.OUT / 'onset_split_verification_results.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
