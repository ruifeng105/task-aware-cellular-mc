"""Shared helpers of the strict waiting evaluations (E1-E4).

Distinct posterior vectors (the author file repeats some rows), three disjoint receiver roles, one-sided
Clopper-Pearson bounds, fixed-sequence calibration with a fallback, and paired bootstrap intervals.
"""
import numpy as np
from scipy import stats

FALLBACK = 'probe_at_120'


def distinct_samples(posterior):
    """Index of the first occurrence of every distinct parameter vector, ascending."""
    _, first = np.unique(posterior, axis=0, return_index=True)
    return np.sort(first)


def three_way_split(ids, seed):
    """Reference, calibration and test receivers: three disjoint thirds of `ids`, each sorted."""
    order = np.asarray(ids)[np.random.default_rng(seed).permutation(len(ids))]
    m = len(ids) // 3
    return np.sort(order[:m]), np.sort(order[m:2 * m]), np.sort(order[2 * m:3 * m])


def cp_lower(k, n, alpha):
    """One-sided (1 - alpha) Clopper-Pearson lower bound for k successes in n trials."""
    return 0. if k == 0 else float(stats.beta.ppf(alpha, k, n - k + 1))


def binom_pvalue(k, n, q):
    """P(Bin(n, q) >= k): p-value of H0 'success probability <= q' against 'greater than q'."""
    return float(stats.binom.sf(k - 1, n, q))


def fixed_sequence(successes, q, alpha):
    """Number of leading hypotheses rejected when (k, n) pairs are tested in the given order at level alpha."""
    m = 0
    for k, n in successes:
        if binom_pvalue(k, n, q) > alpha:
            break
        m += 1
    return m


def tie_key(label):
    return tuple(label[key] for key in ('tau_min', 'threshold', 'delta_min', 'wait_min') if key in label)


def certified_choice(chains, h, q, alpha):
    """Per-history choice certified by fixed-sequence testing.

    `chains`: list of chains, each a list of (label, ok, completion) ordered from the most conservative to the
    most aggressive parameter, with arrays (H, n) on calibration receivers. Each chain is tested for history h at
    alpha / len(chains); among all admitted parameters the smallest calibration mean completion wins (ties: smaller
    tie_key). With nothing admitted the rule probes at the 120-min horizon (certified False)."""
    level = alpha / len(chains)
    admitted = []
    for chain in chains:
        counts = [(int(ok[h].sum()), int(ok.shape[1])) for _, ok, _ in chain]
        admitted += chain[:fixed_sequence(counts, q, level)]
    if not admitted:
        return dict(certified=False, fallback=FALLBACK, calibration=dict(admitted=0))
    label, ok, completion = min(admitted, key=lambda o: (float(o[2][h].mean()), tie_key(o[0])))
    k, n = int(ok[h].sum()), int(ok.shape[1])
    return dict(label, certified=True,
                calibration=dict(admitted=len(admitted), successes=k, n=n, success_rate=k / n,
                                 mean_completion_min=float(completion[h].mean()), p_value=binom_pvalue(k, n, q)))


def paired_bootstrap(diff, reps, seed):
    """95% percentile interval of the mean of per-receiver differences."""
    diff = np.asarray(diff, dtype=float)
    index = np.random.default_rng(seed).integers(0, diff.size, (reps, diff.size))
    return [float(v) for v in np.percentile(diff[index].mean(1), [2.5, 97.5])]
