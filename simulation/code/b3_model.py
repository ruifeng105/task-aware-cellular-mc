"""Execute the original-author B3 model files in Python (no retyped equations).

Reactions, initial values and the FRET measurement are parsed from
FGF_Model_B3.txt, initial_states_B3.txt and measurement_FRET_B3.txt. The
posterior parameter order and fixed values come from the authors'
sim_post_model_summary.txt; samples come from their posterior.txt. Commands
use calibrate_fgf2.segments, the schedules of the forecasting analyses.
"""
from concurrent.futures import ProcessPoolExecutor
import os
import re
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import calibrate_fgf2 as c

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / 'public_data/fgf2_author_repository'
MODEL_DIR = AUTHOR / 'FGF2_models/Fgf_B3'
RESULTS_DIR = AUTHOR / 'Inference_results/Fgf_B3/results/sus_3_20'
RTOL, ATOL = 1e-9, 1e-12


def _block(lines, header):
    """Non-empty lines following a 'Header:' line, up to the next blank line."""
    start = next(i for i, line in enumerate(lines) if line.strip() == header) + 1
    block = []
    for line in lines[start:]:
        if not line.strip():
            if block:
                break
            continue
        block.append(line)
    return block


def _python(expression):
    return expression.replace('^', '**')


class B3Model:
    def __init__(self):
        model_lines = (MODEL_DIR / 'FGF_Model_B3.txt').read_text().splitlines()
        self.model_parameters = [p.strip() for p in _block(model_lines, 'Parameters:')[0].split(',')]
        self.species = _block(model_lines, 'Species:')[0].replace(' ', '').split(',')
        self.reactions = []
        for line in _block(model_lines, 'Reactions:'):
            scheme, propensity = line.split('Propensity:')
            reactants, products = scheme.split('Variables:')[0].split('-->')
            self.reactions.append(([s.strip() for s in reactants.split('+')],
                                   [s.strip() for s in products.split('+')], propensity.strip()))
        initial_lines = (MODEL_DIR / 'initial_states_B3.txt').read_text().splitlines()
        self.initial = {}
        for line in _block(initial_lines, 'Initial Values:'):
            name, value = (part.strip() for part in line.split(':'))
            if name in self.species:
                self.initial[name] = value
        measurement = (MODEL_DIR / 'measurement_FRET_B3.txt').read_text()
        expression = re.search(r'fret_measure\s*=\s*(.+)', measurement).group(1).strip()
        self.measurement = re.sub(r'\+\s*r_1\s*\*\s*FRET_sigma\s*$', '', expression).strip()
        summary = (RESULTS_DIR / 'sim_post_model_summary.txt').read_text().splitlines()
        start = next(i for i, line in enumerate(summary) if 'Parameters will be read from' in line) + 2
        self.parameter_names = []
        for line in summary[start:]:
            if not line.strip():
                break
            self.parameter_names.append(line.strip())
        self.fixed = {}
        for line in summary:
            match = re.match(r'\s*(\w+)\s+fixed to\s+(\S+)', line)
            if match and match.group(1) != 'fgf_input':
                self.fixed[match.group(1)] = float(match.group(2))
        self._compile()

    def _compile(self):
        """Generate one straight-line right-hand side from the parsed reactions."""
        code = ['def rhs(t, x, p, fgf_input):']
        code += [f'    {name} = x[{i}] if x[{i}] > 0. else 0.' for i, name in enumerate(self.species)]
        code += [f'    {name} = p[{i}]' for i, name in enumerate(self.parameter_names)]
        code += [f'    a{j} = {_python(prop)}' for j, (_, _, prop) in enumerate(self.reactions)]
        terms = {name: [] for name in self.species}
        for j, (reactants, products, _) in enumerate(self.reactions):
            for name in reactants:
                terms[name].append(f'- a{j}')
            for name in products:
                terms[name].append(f'+ a{j}')
        code.append('    return [' + ', '.join(' '.join(terms[name]) or '0.' for name in self.species) + ']')
        namespace = {}
        exec('\n'.join(code), namespace)
        self._rhs = namespace['rhs']
        measurement = compile(_python(self.measurement), 'measurement_FRET_B3', 'eval')
        self._measure = lambda env: eval(measurement, {}, env)

    def theta(self, row):
        return dict(zip(self.parameter_names, row), **self.fixed)

    def initial_state(self, row):
        values = self.theta(row)
        return np.array([float(values[v]) if v in values else float(v) for v in
                         (self.initial.get(name, '0') for name in self.species)])

    def simulate_states(self, row, protocol, concentration, times):
        """States at `times` (min) under the commanded schedule of `protocol`."""
        times = np.asarray(times, dtype=float)
        return self.simulate_schedule_states(row, c.segments(protocol, until=float(times.max()) + 1.),
                                             concentration, times)

    def simulate_schedule_states(self, row, segments, concentration, times):
        """States at `times` for explicit half-open pulse intervals; a pulse that
        starts before 0 is simulated from its onset on an unstimulated receiver."""
        times = np.asarray(times, dtype=float)
        start = min([0.] + [a for a, _ in segments])
        breaks = sorted({start, float(times.max())} | {x for seg in segments for x in seg if start < x < times.max()})
        p = np.asarray(row, dtype=float)
        x = self.initial_state(row)
        out = np.full((len(times), len(self.species)), np.nan)
        for left, right in zip(breaks[:-1], breaks[1:]):
            middle = .5 * (left + right)
            u = concentration if any(a <= middle < b for a, b in segments) else 0.
            inside = (times >= left) & (times <= right)
            t_eval = np.union1d(times[inside], [right])
            solution = solve_ivp(self._rhs, (left, right), x, method='LSODA', args=(p, u),
                                 rtol=RTOL, atol=ATOL, t_eval=t_eval)
            if not solution.success:
                raise RuntimeError(solution.message)
            if inside.any():
                out[inside] = solution.y.T[np.searchsorted(t_eval, times[inside])]
            x = solution.y[:, -1]
        if np.isnan(out).any():
            raise RuntimeError('Requested times outside the simulated window')
        return out

    def fret_from_states(self, row, states):
        env = dict(self.theta(row))
        env.update({name: states[:, i] for i, name in enumerate(self.species)})
        return np.asarray(self._measure(env), dtype=float)

    def simulate_fret(self, row, protocol, concentration, times):
        return self.fret_from_states(row, self.simulate_states(row, protocol, concentration, times))

    def simulate_fret_batch(self, rows, protocol, concentration, times, workers=None):
        """FRET for many parameter rows in parallel; deterministic per row."""
        jobs = [(tuple(row), protocol, concentration, tuple(times)) for row in rows]
        with ProcessPoolExecutor(max_workers=workers or min(32, os.cpu_count() or 1)) as pool:
            return np.array(list(pool.map(_simulate_job, jobs, chunksize=max(1, len(jobs) // 128))))


_WORKER_MODEL = None


def worker_model():
    """One parsed model per worker process (the compiled RHS is not picklable)."""
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        _WORKER_MODEL = B3Model()
    return _WORKER_MODEL


def _simulate_job(job):
    row, protocol, concentration, times = job
    return worker_model().simulate_fret(np.array(row), protocol, concentration, np.array(times))


def load_model():
    return B3Model()


def load_posterior():
    return np.loadtxt(RESULTS_DIR / 'posterior.txt', ndmin=2)
