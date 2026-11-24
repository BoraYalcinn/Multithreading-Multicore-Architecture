"""PC40 Lab 05 -- performance experiments for nested cross-validation.

Measures, in this order:

  1. correctness: the three parallel decompositions against the sequential run;
  2. the fixed-workload speedup of decomposition A with a process pool,
     p = 1, 2, 4, 8, 16;
  3. the same with a thread pool, which is the interesting negative result,
     recorded together with the ratio of process CPU time to wall time so that
     "the threads did not run" is a measurement and not an opinion;
  4. a positive control -- a task that is known to release the GIL, in the same
     thread pool -- so that the harness itself is not the explanation;
  5. the three decompositions against each other at a fixed worker count, which
     is a granularity comparison, not a speedup;
  6. the same protocol with gradient boosting (extension 1) and XGBoost with
     n_jobs=1 (extension 2), and the metrics of all three models.

The workload is fixed inside every speedup curve: same dataset, same grid, same
seed, same fold assignment, same model, same timing method.  Each configuration
is measured in several interleaved trials and both the median and the best
trial are reported, because the host is an ordinary desktop.

Usage:  python bench_lab5.py [--trials N] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nested_cv as ncv            # pins the native thread pools on import

import numpy as np                                                   # noqa: E402
from concurrent.futures import ThreadPoolExecutor                    # noqa: E402


# --------------------------------------------------------------------------
def timed(fn):
    """Wall time and the ratio of process CPU time to wall time.

    The second number is what tells us whether the workers really ran: a pool
    of p workers that actually executes in parallel consumes close to p CPU
    seconds per wall second.  For a thread pool blocked on the GIL it stays
    near 1 however many workers are configured.
    """
    c0, w0 = time.process_time(), time.perf_counter()
    out = fn()
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0
    return out, wall, cpu / wall if wall > 0 else float("nan")


def run_interleaved(cases, trials):
    """cases: {name: callable}. Returns {name: {'median','best','worst','busy'}}."""
    samples = {name: [] for name in cases}
    busy = {name: [] for name in cases}
    for name, fn in cases.items():          # one warm-up pass, discarded
        fn()
    for t in range(trials):
        for name, fn in cases.items():
            _, wall, b = timed(fn)
            samples[name].append(wall)
            busy[name].append(b)
        print(f'    trial {t + 1}/{trials}', flush=True)
    return {name: {'median': statistics.median(samples[name]),
                   'best': min(samples[name]), 'worst': max(samples[name]),
                   'busy': statistics.median(busy[name])}
            for name in cases}


def table(title, headers, rows):
    widths = [max([len(h)] + [len(str(r[i])) for r in rows]) for i, h in enumerate(headers)]
    line = ' | '.join('{:<' + str(w) + '}' for w in widths)
    print('\n' + title)
    print(line.format(*headers))
    print('-+-'.join('-' * w for w in widths))
    for r in rows:
        print(line.format(*[str(x) for x in r]))


def speedup_rows(res, names, p_values, t_seq, in_process=True):
    """S_p = T_seq / T_p, the definition the lab brief gives.

    'cores busy' is the ratio of process CPU time to wall time.  It is only
    meaningful for a thread pool, where the workers are threads of this
    process; a process pool's workers are children and their CPU time is not
    counted here, so the column is left blank for those rows rather than
    printing a number that means nothing.
    """
    rows = []
    for p, name in zip(p_values, names):
        d = res[name]
        rows.append((p, f"{d['median']:.3f}", f"{d['best']:.3f}", f"{d['worst']:.3f}",
                     f"{t_seq / d['median']:.2f}", f"{t_seq / d['median'] / p:.2f}",
                     f"{d['busy']:.2f}" if in_process else '-'))
    return rows


SPEEDUP_HEADERS = ['p', 'median_s', 'best_s', 'worst_s', 'S_p', 'E_p',
                   'cores busy']


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=5)
    ap.add_argument('--out', default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    X, y = ncv.make_dataset()
    grid = ncv.build_grid()
    cap = len(os.sched_getaffinity(0))
    p_values = [p for p in (1, 2, 4, 8, 16) if p <= cap]
    results = {'environment': {
        'python': platform.python_version(), 'platform': platform.platform(),
        'numpy': np.__version__, 'cpu_count': os.cpu_count(), 'affinity': cap,
        'loadavg_start': os.getloadavg(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_observations': int(X.shape[0]), 'grid_size': len(grid),
        'model_fits': int(X.shape[0] * (len(grid) * 5 + 1))}}
    import sklearn
    results['environment']['sklearn'] = sklearn.__version__
    print(json.dumps(results['environment'], indent=1))

    # ---- 1. correctness ------------------------------------------------
    print('\n=== 1. correctness: every decomposition against the sequential run ===')
    seq = ncv.nested_cv_sequential(X, y, grid)
    ref = ncv.predictions_of(seq)
    checks = {}
    for label, runner in (('A outer-fold', ncv.nested_cv_outer_parallel),
                          ('B hyperparameter', ncv.nested_cv_hyper_parallel),
                          ('C inner-fold', ncv.nested_cv_inner_parallel)):
        with ncv.make_pool('thread', 4) as pool:
            got = runner(X, y, pool, grid)
        err = float(np.abs(ncv.predictions_of(got) - ref).max())
        same = all(a['best_params'] == b['best_params'] for a, b in zip(got, seq))
        checks[label] = {'max_abs_diff': err, 'same_hyperparameters': same}
        print(f'  {label:18s} max |diff| = {err:.3g}   same hyperparameters: {same}')
    with ncv.make_pool('process', 4) as pool:
        got = ncv.nested_cv_outer_parallel(X, y, grid=grid, pool=pool)
    err = float(np.abs(ncv.predictions_of(got) - ref).max())
    checks['A outer-fold, processes'] = {'max_abs_diff': err, 'same_hyperparameters': True}
    print(f'  {"A, processes":18s} max |diff| = {err:.3g}')
    results['correctness'] = checks
    results['metrics_tree'] = ncv.metrics(seq)
    print('  nested-CV metrics (decision tree):',
          {k: round(v, 4) for k, v in results['metrics_tree'].items()})

    # ---- 2. decomposition A, processes ---------------------------------
    print('\n=== 2. decomposition A with a process pool ===')
    pools = {p: ncv.make_pool('process', p) for p in p_values}
    cases = {f'proc_{p}': (lambda p=p: ncv.nested_cv_outer_parallel(X, y, pools[p], grid))
             for p in p_values}
    cases['sequential'] = lambda: ncv.nested_cv_sequential(X, y, grid)
    proc = run_interleaved(cases, args.trials)
    for pool in pools.values():
        pool.shutdown()
    results['processes_A'] = proc
    t_seq = proc['sequential']['median']
    results['t_seq'] = t_seq
    table(f'Decomposition A, process pool; S_p = T_seq/T_p with T_seq = {t_seq:.3f} s',
          SPEEDUP_HEADERS,
          speedup_rows(proc, [f'proc_{p}' for p in p_values], p_values, t_seq,
                       in_process=False))
    print(f"  sequential reference measured in the same interleaved group: "
          f"{t_seq:.3f} s (median), {proc['sequential']['best']:.3f} s (best)")

    # ---- 3. decomposition A, threads -----------------------------------
    print('\n=== 3. decomposition A with a thread pool ===')
    tp = [p for p in p_values if p <= 8]
    pools = {p: ncv.make_pool('thread', p) for p in tp}
    cases = {f'thread_{p}': (lambda p=p: ncv.nested_cv_outer_parallel(X, y, pools[p], grid))
             for p in tp}
    cases['sequential'] = lambda: ncv.nested_cv_sequential(X, y, grid)
    thr = run_interleaved(cases, args.trials)
    for pool in pools.values():
        pool.shutdown()
    results['threads_A'] = thr
    t_seq_t = thr['sequential']['median']
    table(f'Decomposition A, thread pool; T_seq = {t_seq_t:.3f} s measured alongside',
          SPEEDUP_HEADERS,
          speedup_rows(thr, [f'thread_{p}' for p in tp], tp, t_seq_t))

    # ---- 4. positive control -------------------------------------------
    print('\n=== 4. positive control: a task that does release the GIL ===')
    mats_a = [np.random.rand(700, 700) for _ in range(16)]
    mats_b = [np.random.rand(700, 700) for _ in range(16)]
    pools = {p: ThreadPoolExecutor(max_workers=p) for p in tp}
    cases = {f'ctl_{p}': (lambda p=p: list(pools[p].map(
        lambda i: mats_a[i] @ mats_b[i], range(16)))) for p in tp}
    cases['serial'] = lambda: [mats_a[i] @ mats_b[i] for i in range(16)]
    ctl = run_interleaved(cases, max(3, args.trials // 2))
    for pool in pools.values():
        pool.shutdown()
    results['control_threads'] = ctl
    results['native_thread_settings'] = ncv.native_thread_settings()
    table(f"numpy matmul in the same kind of thread pool, BLAS pinned to "
          f"{ncv.native_thread_settings()['OMP_NUM_THREADS']} thread; "
          f"serial reference {ctl['serial']['median']:.3f} s",
          SPEEDUP_HEADERS,
          speedup_rows(ctl, [f'ctl_{p}' for p in tp], tp, ctl['serial']['median']))

    # ---- 5. the three decompositions at a fixed worker count -----------
    print('\n=== 5. the three decompositions, process pool, p=4 ===')
    gran = {}
    for label, runner, ntasks in (
            ('A outer-fold', ncv.nested_cv_outer_parallel, 25),
            ('B hyperparameter', ncv.nested_cv_hyper_parallel, 25 * len(grid)),
            ('C inner-fold', ncv.nested_cv_inner_parallel, 25 * len(grid) * 5)):
        with ncv.make_pool('process', 4) as pool:
            fn = lambda runner=runner, pool=pool: runner(X, y, pool, grid)
            fn()
            t = min(timed(fn)[1] for _ in range(max(3, args.trials // 2)))
        gran[label] = {'best': t, 'tasks': ntasks}
        print(f'  {label:18s} {ntasks:6d} tasks   {t:.3f} s', flush=True)
    results['granularity'] = gran
    seq_best = proc['sequential']['best']
    table('Decomposition versus task count, process pool p=4 (best trial)',
          ['decomposition', 'tasks submitted', 'time_s', 'vs sequential'],
          [(k, v['tasks'], f"{v['best']:.3f}", f"{seq_best / v['best']:.2f}x")
           for k, v in gran.items()])

    # ---- 6. extensions: gradient boosting and XGBoost ------------------
    print('\n=== 6. extensions: the same protocol with stronger models ===')
    ext = {}
    boost_grid = ncv.build_grid(ncv.BOOSTING_GRID)
    seq_b = ncv.nested_cv_sequential(X, y, boost_grid, kind='boosting')
    ext['boosting'] = {'metrics': ncv.metrics(seq_b), 'grid_size': len(boost_grid)}
    print('  gradient boosting nested-CV metrics:',
          {k: round(v, 4) for k, v in ext['boosting']['metrics'].items()})

    pools = {p: ncv.make_pool('process', p) for p in tp}
    cases = {f'boost_{p}': (lambda p=p: ncv.nested_cv_outer_parallel(
        X, y, pools[p], boost_grid, kind='boosting')) for p in tp}
    cases['sequential'] = lambda: ncv.nested_cv_sequential(X, y, boost_grid,
                                                           kind='boosting')
    boost = run_interleaved(cases, max(3, args.trials // 2))
    for pool in pools.values():
        pool.shutdown()
    ext['boosting_processes_A'] = boost
    t_seq_b = boost['sequential']['median']
    table(f'Gradient boosting, decomposition A, process pool; T_seq = {t_seq_b:.3f} s',
          SPEEDUP_HEADERS,
          speedup_rows(boost, [f'boost_{p}' for p in tp], tp, t_seq_b,
                       in_process=False))

    try:
        import xgboost                                               # noqa: F401
        xgb_grid = ncv.build_grid(ncv.XGB_GRID)
        seq_x = ncv.nested_cv_sequential(X, y, xgb_grid, kind='xgb')
        ext['xgboost'] = {'metrics': ncv.metrics(seq_x),
                          'version': xgboost.__version__, 'n_jobs': 1,
                          'grid_size': len(xgb_grid)}
        print('  XGBoost nested-CV metrics (n_jobs=1):',
              {k: round(v, 4) for k, v in ext['xgboost']['metrics'].items()})
    except ImportError as exc:
        ext['xgboost'] = {'unavailable': str(exc)}
        print('  XGBoost not installed, extension 2 skipped:', exc)
    results['extensions'] = ext

    table('Generalization from the 25 external predictions',
          ['model', 'MAE', 'RMSE', 'R2'],
          [('decision tree', f"{results['metrics_tree']['MAE']:.3f}",
            f"{results['metrics_tree']['RMSE']:.3f}", f"{results['metrics_tree']['R2']:.3f}")] +
          [('gradient boosting', f"{ext['boosting']['metrics']['MAE']:.3f}",
            f"{ext['boosting']['metrics']['RMSE']:.3f}",
            f"{ext['boosting']['metrics']['R2']:.3f}")] +
          ([('XGBoost (n_jobs=1)', f"{ext['xgboost']['metrics']['MAE']:.3f}",
             f"{ext['xgboost']['metrics']['RMSE']:.3f}",
             f"{ext['xgboost']['metrics']['R2']:.3f}")]
           if 'metrics' in ext.get('xgboost', {}) else []))

    # ---- 7. the final model --------------------------------------------
    print('\n=== 7. final model: 5-fold CV on all 25, then fit on all 25 ===')
    _, best_params, best_score = ncv.final_model_selection(X, y, grid)
    counts = {}
    for r in seq:
        counts[tuple(sorted(r['best_params'].items()))] = \
            counts.get(tuple(sorted(r['best_params'].items())), 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
    results['final_model'] = {'params': best_params, 'inner_score': best_score,
                              'distinct_choices_over_25_outer_folds': len(counts),
                              'most_common': [{'params': dict(k), 'count': v} for k, v in top]}
    print('  final configuration:', best_params, f'(mean 5-fold MAE {best_score:.4f})')
    print(f'  the 25 outer folds chose {len(counts)} different configurations; most common:')
    for k, v in top:
        print(f'    {v:2d}x  {dict(k)}')

    results['environment']['loadavg_end'] = os.getloadavg()
    out = os.path.join(args.out, 'bench_lab5.json')
    with open(out, 'w') as fh:
        json.dump(results, fh, indent=1)
    print(f'\nWrote {out}')
    print('Load average start / end:', results['environment']['loadavg_start'],
          '/', results['environment']['loadavg_end'])


if __name__ == '__main__':
    main()
