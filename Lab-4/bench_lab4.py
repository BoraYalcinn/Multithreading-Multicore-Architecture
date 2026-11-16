"""PC40 Lab04 - extended strong-scaling study on a local multicore CPU.

The notebook follows the protocol supplied with the lab and caps the thread
count at 4, which is the conservative choice for a shared Colab VM.  This
script repeats the same two experiments on the allocated local node, where
16 logical CPUs are visible, and adds the measurements the notebook does not
make: p up to 16, a workload with enough head tasks to occupy them, the cost
of the pool dispatch on its own, and a granularity sweep.

Protocol.  Every configuration is measured once per trial and the trials are
interleaved, so that a slow period of the host hits all configurations rather
than whichever one happened to run during it.  Reported values are the median
over trials of the per-trial median, plus the best trial, which is the value
least contaminated by competing load.

Usage:  python bench_lab4.py [--trials N] [--out DIR]
"""

import argparse, csv, json, math, os, platform, statistics, time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

torch.manual_seed(40)


# --------------------------------------------------------------------------
# attention kernels (identical to the notebook)
# --------------------------------------------------------------------------
def attention_dense(q, k, v, causal=False):
    d_k = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    if causal:
        len_q, len_k = scores.shape[-2], scores.shape[-1]
        future = torch.ones(len_q, len_k, dtype=torch.bool, device=scores.device).triu(1)
        scores = scores.masked_fill(future, float('-inf'))
    return torch.softmax(scores, dim=-1) @ v


def serial_heads(q, k, v):
    return torch.stack([attention_dense(q[:, h], k[:, h], v[:, h])
                        for h in range(q.shape[1])], dim=1)


def parallel_heads(q, k, v, pool):
    def one_head(h):
        return attention_dense(q[:, h], k[:, h], v[:, h])
    return torch.stack(list(pool.map(one_head, range(q.shape[1]))), dim=1)


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
def measure_once(fn, intra_threads=1, min_run_time=0.30):
    timer = benchmark.Timer(stmt='fn()', globals={'fn': fn}, num_threads=intra_threads)
    result = timer.blocked_autorange(min_run_time=min_run_time)
    return 1000 * result.median


def run_interleaved(configs, trials, min_run_time=0.30):
    """configs: list of (series, p, fn, intra_threads). Returns {(series,p): [ms,...]}."""
    samples = {(s, p): [] for s, p, _, _ in configs}
    for series, p, fn, intra in configs:          # one warm-up pass, discarded
        fn()
    for trial in range(trials):
        for series, p, fn, intra in configs:
            samples[(series, p)].append(measure_once(fn, intra, min_run_time))
        print(f'  trial {trial + 1}/{trials} done', flush=True)
    return samples


def summarize(samples):
    out = {}
    for key, values in samples.items():
        out[key] = {'median': statistics.median(values),
                    'best': min(values), 'worst': max(values),
                    'trials': len(values)}
    return out


def table(title, rows, headers):
    widths = [max([len(h)] + [len(str(r[i])) for r in rows]) for i, h in enumerate(headers)]
    line = ' | '.join('{:<' + str(w) + '}' for w in widths)
    print('\n' + title)
    print(line.format(*headers))
    print('-+-'.join('-' * w for w in widths))
    for r in rows:
        print(line.format(*[str(x) for x in r]))


# --------------------------------------------------------------------------
def environment():
    quota = 'not exposed'
    try:
        raw = open('/sys/fs/cgroup/cpu.max').read().split()
        if raw[0] != 'max':
            quota = f'{int(raw[0]) / int(raw[1]):.2f} CPU-equivalents'
    except (OSError, ValueError, IndexError):
        pass
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'platform': platform.platform(),
            'cpu_count': os.cpu_count(),
            'affinity': len(os.sched_getaffinity(0)),
            'cgroup_cpu_quota': quota,
            'loadavg_at_start': os.getloadavg(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=7)
    ap.add_argument('--out', default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    torch.set_num_threads(1)
    env = environment()
    print(json.dumps(env, indent=2))
    cap = min(env['cpu_count'], env['affinity'])
    p_values = [p for p in (1, 2, 4, 8, 16) if p <= cap]
    results = {'environment': env, 'p_values': p_values}

    # ---- workloads ------------------------------------------------------
    # A: the notebook workload, H = 4 head tasks.
    # B: the same per-head algorithm with H = 16, so p = 16 has one task each.
    workloads = {'H4': (2, 4, 192, 64), 'H16': (2, 16, 192, 64)}
    tensors = {}
    for name, (b, h, l, d) in workloads.items():
        q = torch.randn(b, h, l, d)
        tensors[name] = (q, torch.randn_like(q), torch.randn_like(q))

    # correctness before timing: the parallel path must equal the serial path
    checks = {}
    for name, (q, k, v) in tensors.items():
        ref = serial_heads(q, k, v)
        with ThreadPoolExecutor(max_workers=4) as pool:
            got = parallel_heads(q, k, v, pool)
        err = (got - ref).abs().max().item()
        assert got.shape == ref.shape and err < 1e-5, (name, err)
        sdpa = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        checks[name] = {'pool_vs_serial_max_abs_err': err,
                        'dense_vs_sdpa_max_abs_err': (ref - sdpa).abs().max().item()}
        print(f'[check] {name}: pool vs serial {err:.3g}, '
              f'dense vs SDPA {checks[name]["dense_vs_sdpa_max_abs_err"]:.3g}')
    results['correctness'] = checks

    # ---- experiment 1: explicit per-head thread pool, intra-op = 1 -------
    print('\n=== experiment 1: explicit per-head tasks (intra-op = 1) ===')
    pools = {p: ThreadPoolExecutor(max_workers=p) for p in p_values}
    configs = []
    for name, (q, k, v) in tensors.items():
        for p in p_values:
            configs.append((f'pool_{name}', p,
                            (lambda q=q, k=k, v=v, p=p: parallel_heads(q, k, v, pools[p])), 1))
    for name, (q, k, v) in tensors.items():
        configs.append((f'serial_loop_{name}', 1, (lambda q=q, k=k, v=v: serial_heads(q, k, v)), 1))
    explicit = summarize(run_interleaved(configs, args.trials))
    for p in pools.values():
        p.shutdown()

    # ---- experiment 2: vectorized attention, vary intra-op threads ------
    print('\n=== experiment 2: vectorized attention (intra-op threads) ===')
    configs = []
    for name, (q, k, v) in tensors.items():
        for p in p_values:
            configs.append((f'vector_{name}', p,
                            (lambda q=q, k=k, v=v: attention_dense(q, k, v)), p))
    vector = summarize(run_interleaved(configs, args.trials))

    # ---- experiment 3: dispatch overhead of the pool alone --------------
    print('\n=== experiment 3: pool dispatch overhead (empty tasks) ===')
    pools = {p: ThreadPoolExecutor(max_workers=p) for p in p_values}
    noop = lambda h: h
    configs = [(f'dispatch_{h}tasks', p,
                (lambda p=p, h=h: list(pools[p].map(noop, range(h)))), 1)
               for h in (4, 16) for p in p_values]
    dispatch = summarize(run_interleaved(configs, max(3, args.trials // 2), min_run_time=0.15))
    for p in pools.values():
        p.shutdown()

    # ---- experiment 4: granularity -- work per head task ----------------
    print('\n=== experiment 4: granularity (sequence length per head task) ===')
    pools = {p: ThreadPoolExecutor(max_workers=p) for p in (1, 4)}
    gran_lengths = [32, 64, 128, 256, 512]
    gran_tensors = {}
    for sl in gran_lengths:
        q = torch.randn(2, 4, sl, 64)
        gran_tensors[sl] = (q, torch.randn_like(q), torch.randn_like(q))
    configs = []
    for sl, (q, k, v) in gran_tensors.items():
        for p in (1, 4):
            configs.append((f'gran_L{sl}', p,
                            (lambda q=q, k=k, v=v, p=p: parallel_heads(q, k, v, pools[p])), 1))
    gran = summarize(run_interleaved(configs, max(3, args.trials // 2), min_run_time=0.20))
    for p in pools.values():
        p.shutdown()

    # ---- experiment 5: implementation comparison at fixed intra-op ------
    print('\n=== experiment 5: implementations at a fixed intra-op setting ===')
    q, k, v = tensors['H4']
    impls = {'serial head loop': lambda: serial_heads(q, k, v),
             'vectorized attention': lambda: attention_dense(q, k, v),
             'PyTorch SDPA': lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)}
    configs = [(f'impl_{n}', 1, fn, 1) for n, fn in impls.items()]
    impl_res = summarize(run_interleaved(configs, args.trials))

    # ---- experiment 6: sequence-length scaling and score memory ---------
    print('\n=== experiment 6: sequence length, time and score memory ===')
    lengths = [64, 128, 256, 512, 768]
    len_tensors = {}
    for sl in lengths:
        q = torch.randn(2, 4, sl, 64)
        len_tensors[sl] = (q, torch.randn_like(q), torch.randn_like(q))
    configs = [(f'len_{sl}', 1, (lambda q=q, k=k, v=v: attention_dense(q, k, v)), 1)
               for sl, (q, k, v) in len_tensors.items()]
    len_res = summarize(run_interleaved(configs, max(3, args.trials // 2), min_run_time=0.20))

    # ---- experiment 7: prefill vs cached decoding -----------------------
    print('\n=== experiment 7: prefill versus cached decoding ===')
    gen_len, gen_h, gen_d = 256, 4, 64
    q = torch.randn(1, gen_h, gen_len, gen_d)
    k, v = torch.randn_like(q), torch.randn_like(q)
    max_err = 0.0
    for t in range(1, gen_len + 1, 17):
        whole = attention_dense(q[:, :, :t], k[:, :, :t], v[:, :, :t], causal=True)[:, :, -1:]
        cached = attention_dense(q[:, :, t - 1:t], k[:, :, :t], v[:, :, :t], causal=False)
        max_err = max(max_err, (whole - cached).abs().max().item())
    configs = [('prefill', 1, (lambda: attention_dense(q, k, v, causal=True)), 1),
               ('decode_step', 1, (lambda: attention_dense(q[:, :, -1:], k, v, causal=False)), 1),
               ('decode_no_cache', 1,
                (lambda: attention_dense(q, k, v, causal=True)[:, :, -1:]), 1)]
    gen_res = summarize(run_interleaved(configs, max(3, args.trials // 2), min_run_time=0.20))
    results['kv_cache_max_abs_err'] = max_err

    # ---- experiment 8: what actually limits the pool --------------------
    # Three task kinds in the same pool, same p sweep.  They separate the
    # candidate causes: if the pool and the GIL were the limit, none of them
    # would scale; if the limit is the memory traffic of the attention kernel,
    # only the compute-bound task scales.
    print('\n=== experiment 8: control -- what limits the per-head pool ===')
    pools = {p: ThreadPoolExecutor(max_workers=p) for p in p_values}

    gemm_a = [torch.randn(1024, 1024) for _ in range(16)]
    gemm_b = [torch.randn(1024, 1024) for _ in range(16)]
    head_q = [torch.randn(2, 512, 64) for _ in range(16)]
    head_k = [torch.randn(2, 512, 64) for _ in range(16)]
    head_v = [torch.randn(2, 512, 64) for _ in range(16)]

    def task_gemm(i):                       # compute bound, high intensity
        return gemm_a[i] @ gemm_b[i]

    def task_head(i):                       # the Lab04 per-head task
        return attention_dense(head_q[i], head_k[i], head_v[i])

    def task_python(i):                     # pure Python bytecode, GIL bound
        total = 0
        for j in range(300_000):
            total += j
        return total

    ntasks = 16
    configs = []
    for name, fn in (('ctl_gemm', task_gemm), ('ctl_head', task_head),
                     ('ctl_python', task_python)):
        for p in p_values:
            configs.append((name, p,
                            (lambda p=p, fn=fn: list(pools[p].map(fn, range(ntasks)))), 1))
    control = summarize(run_interleaved(configs, max(3, args.trials // 2), min_run_time=0.40))
    for p in pools.values():
        p.shutdown()

    # arithmetic intensity of the two tensor tasks, as a rough ratio
    gemm_flops = 2 * 1024**3
    gemm_bytes = 3 * 1024 * 1024 * 4
    head_flops = 2 * 2 * 512 * 512 * 64 * 2
    head_bytes = (3 * 2 * 512 * 64 * 4            # Q, K, V
                  + 3 * 2 * 512 * 512 * 4         # scores written, read, weights written
                  + 2 * 512 * 64 * 4)             # output
    results['arithmetic_intensity'] = {
        'gemm_flops_per_byte': gemm_flops / gemm_bytes,
        'attention_head_flops_per_byte': head_flops / head_bytes}

    # ---- report ---------------------------------------------------------
    def speedup_rows(data, prefix, label):
        rows = []
        base = data[(prefix, p_values[0])]['median']
        base_best = data[(prefix, p_values[0])]['best']
        for p in p_values:
            d = data[(prefix, p)]
            rows.append((p, f"{d['median']:.3f}", f"{d['best']:.3f}", f"{d['worst']:.3f}",
                         f"{base / d['median']:.2f}", f"{base / d['median'] / p:.2f}",
                         f"{base_best / d['best']:.2f}"))
        table(label, rows, ['p', 'median_ms', 'best_ms', 'worst_ms',
                            'S_p(median)', 'E_p', 'S_p(best)'])
        return rows

    rows_pool_h4 = speedup_rows(explicit, 'pool_H4', 'Explicit per-head pool, H=4 (intra-op=1)')
    rows_pool_h16 = speedup_rows(explicit, 'pool_H16', 'Explicit per-head pool, H=16 (intra-op=1)')
    rows_vec_h4 = speedup_rows(vector, 'vector_H4', 'Vectorized attention, H=4 (intra-op sweep)')
    rows_vec_h16 = speedup_rows(vector, 'vector_H16', 'Vectorized attention, H=16 (intra-op sweep)')

    table('Serial head loop (separate implementation, for reference only)',
          [(n.replace('serial_loop_', ''), f"{d['median']:.3f}", f"{d['best']:.3f}")
           for (n, p), d in explicit.items() if n.startswith('serial_loop')],
          ['workload', 'median_ms', 'best_ms'])

    table('Thread-pool dispatch cost with empty tasks',
          [(n.replace('dispatch_', ''), p, f"{d['median'] * 1000:.1f}")
           for (n, p), d in dispatch.items()],
          ['tasks', 'workers', 'median_us'])

    gran_rows = []
    for sl in gran_lengths:
        t1 = gran[(f'gran_L{sl}', 1)]['median']
        t4 = gran[(f'gran_L{sl}', 4)]['median']
        gran_rows.append((sl, f'{t1:.3f}', f'{t1 / 4:.3f}', f'{t4:.3f}', f'{t1 / t4:.2f}'))
    table('Granularity: work per head task (H=4, pool p=1 vs p=4)', gran_rows,
          ['L', 'pool_p1_ms', 'work_per_task_ms', 'pool_p4_ms', 'S_4'])

    impl_base = impl_res[('impl_serial head loop', 1)]['median']
    table('Implementations at intra-op = 1 (optimization ratio, NOT a parallel speedup)',
          [(n.replace('impl_', ''), f"{d['median']:.3f}", f"{d['best']:.3f}",
            f"{impl_base / d['median']:.2f}x") for (n, p), d in impl_res.items()],
          ['implementation', 'median_ms', 'best_ms', 'loop / implementation'])

    len_rows = []
    prev = None
    for sl in lengths:
        t = len_res[(f'len_{sl}', 1)]['median']
        mib = 2 * 4 * sl * sl * 4 / 1024**2
        len_rows.append((sl, f'{mib:.2f}', f'{t:.3f}', '-' if prev is None else f'{t / prev:.2f}x'))
        prev = t
    table('Sequence length: score-matrix memory and latency (B=2,H=4,d_h=64, 1 thread)',
          len_rows, ['L', 'one_score_matrix_MiB', 'median_ms', 'time vs previous L'])

    pf = gen_res[('prefill', 1)]['median']
    st = gen_res[('decode_step', 1)]['median']
    nc = gen_res[('decode_no_cache', 1)]['median']
    table(f'Prefill versus decoding (L={gen_len}, H={gen_h}, d_h={gen_d}); '
          f'KV-cache max abs error {max_err:.3g}',
          [('prefill of the whole prompt', f'{pf:.4f}', '1 step'),
           ('one decoding step, KV cache', f'{st:.4f}', f'{gen_len} steps = {gen_len * st:.2f} ms'),
           ('one decoding step, no cache', f'{nc:.4f}', f'{gen_len} steps = {gen_len * nc:.2f} ms')],
          ['regime', 'median_ms', 'cost of a full sequence'])


    ctl_rows = []
    for name, label in (('ctl_gemm', 'compute-bound GEMM 1024^3'),
                        ('ctl_head', 'Lab04 attention head, L=512'),
                        ('ctl_python', 'pure Python loop (GIL-bound)')):
        base = control[(name, p_values[0])]['median']
        ctl_rows.append([label] + [f"{base / control[(name, p)]['median']:.2f}" for p in p_values])
    table('Control: same pool, same task count (16), three kinds of task -- S_p',
          ctl_rows, ['task kind'] + [f'p={p}' for p in p_values])
    ai = results['arithmetic_intensity']
    print(f"  arithmetic intensity: GEMM {ai['gemm_flops_per_byte']:.0f} FLOP/byte, "
          f"attention head {ai['attention_head_flops_per_byte']:.1f} FLOP/byte")

    # ---- persist --------------------------------------------------------
    outdir = args.out
    os.makedirs(outdir, exist_ok=True)
    flat = []
    for tag, data in (('explicit_pool', explicit), ('vectorized_intraop', vector),
                      ('dispatch', dispatch), ('granularity', gran),
                      ('implementations', impl_res), ('length_scan', len_res),
                      ('generation', gen_res), ('control', control)):
        for (series, p), d in data.items():
            flat.append({'experiment': tag, 'series': series, 'p': p,
                         'median_ms': round(d['median'], 6), 'best_ms': round(d['best'], 6),
                         'worst_ms': round(d['worst'], 6), 'trials': d['trials']})
    csv_path = os.path.join(outdir, 'bench_lab4.csv')
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)
    results['rows'] = flat
    results['loadavg_at_end'] = os.getloadavg()
    json_path = os.path.join(outdir, 'bench_lab4.json')
    with open(json_path, 'w') as fh:
        json.dump(results, fh, indent=1)
    print(f'\nWrote {csv_path} and {json_path}')
    print('Load average at start / end:', env['loadavg_at_start'], '/', results['loadavg_at_end'])


if __name__ == '__main__':
    main()
