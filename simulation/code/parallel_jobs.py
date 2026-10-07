"""Run independent deterministic jobs in a process pool and retry jobs that fail for lack of memory.

Each job's result depends only on its arguments, so a retried job returns
exactly what it would have returned the first time. Only memory errors and
broken pools are retried (with half the workers); any other exception is a
bug and is raised at once.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import time


def run_jobs(fn, jobs, workers, attempts=40, pause_s=90.):
    results, pending = {}, list(range(len(jobs)))
    for _ in range(attempts):
        if not pending:
            break
        try:
            with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(fn, jobs[i]): i for i in pending}
                for future in as_completed(futures):
                    try:
                        results[futures[future]] = future.result()
                    except (MemoryError, BrokenProcessPool):
                        pass
        except BrokenProcessPool:
            pass
        pending = [i for i in pending if i not in results]
        if pending:
            workers = max(1, workers // 2)
            print(f'retrying {len(pending)} jobs with {workers} workers after a memory failure', flush=True)
            time.sleep(pause_s)
    if pending:
        raise RuntimeError(f'{len(pending)} jobs still failing after {attempts} attempts')
    return [results[i] for i in range(len(jobs))]
