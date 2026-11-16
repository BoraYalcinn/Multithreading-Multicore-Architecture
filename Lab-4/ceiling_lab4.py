"""PC40 Lab04 -- how much speedup can this machine give at all?

A first control experiment showed that the per-head thread pool is not limited
by the GIL, because the same pool scales on a compute-bound task.  It did not
establish *what* the limit is.  An attempt to attribute it to memory bandwidth
did not survive testing and is reported here as a rejected hypothesis.

This script measures the limit directly instead of inferring it:

  1. Machine ceiling.  A large, compute-bound GEMM -- the most parallel-friendly
     workload available -- at every thread count, with the ratio of process CPU
     time to wall time, which proves how many cores were really busy.
  2. Core inventory.  Single-thread throughput of one P-core, one E-core and one
     LP-E core, since this CPU is heterogeneous and a join waits for the slowest.
  3. Homogeneous subsets.  The same GEMM and the same Lab04 per-head pool pinned
     to P-cores only and to E-cores only, which separates the hardware ceiling
     from any property of the decomposition.
  4. Rejected hypothesis.  Arithmetic intensity swept at a fixed score-matrix
     footprint, the three kernel stages measured separately, and allocation
     separated from traffic.

Usage:  python ceiling_lab4.py [--out DIR]
"""

import argparse, json, math, os, platform, time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.utils.benchmark as benchmark

torch.set_num_threads(1)
torch.manual_seed(40)

NTASKS = 16


def attention_dense(q, k, v):
    s = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    return torch.softmax(s, dim=-1) @ v


def measure_ms(fn, min_run_time=0.40, repeats=4):
    """Best of `repeats` independent blocks: the estimator least contaminated
    by competing load on an interactive host."""
    out = []
    for _ in range(repeats):
        t = benchmark.Timer(stmt='fn()', globals={'fn': fn}, num_threads=1)
        out.append(1000 * t.blocked_autorange(min_run_time=min_run_time).median)
    return min(out)


def cpu_topology():
    def read(path):
        try:
            return open(path).read().strip()
        except OSError:
            return None

    def expand(spec):
        if not spec:
            return []
        cpus = []
        for part in spec.split(','):
            if '-' in part:
                lo, hi = part.split('-')
                cpus.extend(range(int(lo), int(hi) + 1))
            else:
                cpus.append(int(part))
        return cpus

    p = expand(read('/sys/devices/cpu_core/cpus'))
    e = expand(read('/sys/devices/cpu_atom/cpus'))
    rest = [c for c in range(os.cpu_count()) if c not in p and c not in e]
    return {'P': p, 'E': e, 'other': rest}


def gemm_throughput(n, nthreads, reps=8):
    """Returns GFLOP/s and the measured cores-busy ratio (process CPU / wall)."""
    a, b = torch.randn(n, n), torch.randn(n, n)
    torch.set_num_threads(nthreads)
    for _ in range(2):
        a @ b
    c0, w0 = time.process_time(), time.perf_counter()
    for _ in range(reps):
        a @ b
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    torch.set_num_threads(1)
    return reps * 2 * n ** 3 / 1e9 / wall, cpu / wall


def pooled(task, p, ntasks=NTASKS):
    pool = ThreadPoolExecutor(max_workers=p)
    fn = lambda: list(pool.map(task, range(ntasks)))
    fn()
    t = measure_ms(fn)
    pool.shutdown()
    return t


def table(title, headers, rows):
    widths = [max([len(h)] + [len(str(r[i])) for r in rows]) for i, h in enumerate(headers)]
    line = ' | '.join('{:<' + str(w) + '}' for w in widths)
    print('\n' + title)
    print(line.format(*headers))
    print('-+-'.join('-' * w for w in widths))
    for r in rows:
        print(line.format(*[str(x) for x in r]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    topo = cpu_topology()
    all_cpus = sorted(os.sched_getaffinity(0))
    results = {'platform': platform.platform(), 'torch': torch.__version__,
               'topology': topo, 'loadavg': os.getloadavg(),
               'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    print(json.dumps({k: results[k] for k in ('platform', 'torch', 'topology', 'loadavg')}, indent=1))

    # ---- 1. machine ceiling -------------------------------------------
    print('\n=== 1. machine ceiling: one large compute-bound GEMM (2048^3) ===')
    rows, ceiling = [], {}
    base = None
    for nt in (1, 2, 4, 6, 8, 12, 16):
        if nt > len(all_cpus):
            continue
        g, busy = gemm_throughput(2048, nt)
        base = base or g
        ceiling[nt] = {'gflops': g, 'cores_busy': busy, 'speedup': g / base}
        rows.append((nt, f'{g:.1f}', f'{g/base:.2f}', f'{busy:.2f}', f'{g/busy:.1f}'))
    table('Intra-op threads on the full machine', 
          ['threads', 'GFLOP/s', 'speedup', 'cores busy', 'GFLOP/s per busy core'], rows)
    results['ceiling_all'] = ceiling

    # ---- 2. core inventory --------------------------------------------
    print('\n=== 2. single-thread throughput of each core type ===')
    rows, inventory = [], {}
    for kind, cpus in topo.items():
        if not cpus:
            continue
        os.sched_setaffinity(0, {cpus[0]})
        g, _ = gemm_throughput(2048, 1, reps=6)
        inventory[kind] = g
        rows.append((kind, cpus[0], f'{g:.1f}'))
    os.sched_setaffinity(0, set(all_cpus))
    slowest, fastest = min(inventory.values()), max(inventory.values())
    table('One core of each type, GEMM 2048^3',
          ['core type', 'cpu', 'GFLOP/s'], rows)
    print(f'  fastest / slowest core = {fastest/slowest:.2f}x -- a join waits for the slowest worker')
    results['core_inventory'] = inventory

    # ---- 3. homogeneous subsets ---------------------------------------
    print('\n=== 3. the same work on a homogeneous subset of cores ===')
    q = [torch.randn(2, 512, 64) for _ in range(NTASKS)]
    k = [torch.randn(2, 512, 64) for _ in range(NTASKS)]
    v = [torch.randn(2, 512, 64) for _ in range(NTASKS)]
    head_task = lambda i: attention_dense(q[i], k[i], v[i])

    subsets = {'P-cores only': topo['P'], 'E-cores only': topo['E'],
               'all CPUs (heterogeneous)': all_cpus}
    rows_g, rows_h, homog = [], [], {}
    for label, cpus in subsets.items():
        if not cpus:
            continue
        os.sched_setaffinity(0, set(cpus))
        ps = [p for p in (1, 2, 4, 8) if p <= len(cpus)]
        gemm = {}
        gb = None
        for p in ps:
            g, _ = gemm_throughput(2048, p, reps=6)
            gb = gb or g
            gemm[p] = g / gb
        head = {p: pooled(head_task, p) for p in ps}
        hb = head[ps[0]]
        homog[label] = {'cpus': cpus, 'gemm_speedup': gemm,
                        'head_ms': head, 'head_speedup': {p: hb / head[p] for p in ps}}
        rows_g.append((label, len(cpus)) + tuple(f'{gemm[p]:.2f}' for p in ps) + ('',) * (4 - len(ps)))
        rows_h.append((label, len(cpus)) + tuple(f'{hb/head[p]:.2f}' for p in ps) + ('',) * (4 - len(ps)))
    os.sched_setaffinity(0, set(all_cpus))
    hdr = ['cpu set', 'cpus', 'p=1', 'p=2', 'p=4', 'p=8']
    table('Compute-bound GEMM, intra-op threads -- S_p', hdr, rows_g)
    table('Lab04 per-head pool, 16 tasks -- S_p', hdr, rows_h)
    results['homogeneous'] = homog

    # ---- 4. the rejected hypothesis ------------------------------------
    print('\n=== 4. rejected hypothesis: memory bandwidth of the dense kernel ===')
    rejected = {}

    # 4a. arithmetic intensity at a fixed score-matrix footprint
    rows = []
    for dh in (16, 64, 256):
        qq = [torch.randn(2, 512, dh) for _ in range(NTASKS)]
        kk = [torch.randn(2, 512, dh) for _ in range(NTASKS)]
        vv = [torch.randn(2, 512, dh) for _ in range(NTASKS)]
        task = lambda i, qq=qq, kk=kk, vv=vv: attention_dense(qq[i], kk[i], vv[i])
        flops = 2 * 2 * 512 * 512 * dh * 2
        byts = 3 * 2 * 512 * dh * 4 + 3 * 2 * 512 * 512 * 4 + 2 * 512 * dh * 4
        t = {p: pooled(task, p) for p in (1, 4)}
        rejected[f'intensity_dh{dh}'] = {'flops_per_byte': flops / byts, 'S_4': t[1] / t[4]}
        rows.append((dh, f'{flops/byts:.1f}', f'{t[1]/t[4]:.2f}'))
    table('4a. intensity swept, score matrix held at 512x512 (expect monotonic if bandwidth-bound)',
          ['d_h', 'FLOP/byte', 'S_4'], rows)

    # 4b. the three stages separately
    s0 = [torch.randn(2, 512, 512) for _ in range(NTASKS)]
    a0 = [torch.softmax(s, dim=-1) for s in s0]
    stages = {
        'QK^T (writes 2 MiB)': lambda i: q[i] @ k[i].transpose(-2, -1),
        'softmax (reads+writes 2 MiB)': lambda i: torch.softmax(s0[i], dim=-1),
        'A@V (reads 2 MiB)': lambda i: a0[i] @ v[i]}
    rows = []
    for name, task in stages.items():
        t = {p: pooled(task, p) for p in (1, 4)}
        rejected[name] = t[1] / t[4]
        rows.append((name, f'{t[1]/t[4]:.2f}'))
    table('4b. the stages of the kernel (expect softmax worst if bandwidth-bound)',
          ['stage', 'S_4'], rows)

    # 4c. traffic versus allocation
    src = [torch.randn(2, 512, 512) for _ in range(NTASKS)]
    dst = [torch.empty(2, 512, 512) for _ in range(NTASKS)]
    variants = {
        'elementwise add, out= preallocated': lambda i: torch.add(src[i], 1.0, out=dst[i]),
        'elementwise add, allocates output': lambda i: src[i] + 1.0,
        'allocate 2 MiB, touch one element': lambda i: torch.empty(2, 512, 512).fill_(1.0)}
    rows = []
    for name, task in variants.items():
        t = {p: pooled(task, p) for p in (1, 4)}
        rejected[name] = t[1] / t[4]
        rows.append((name, f'{t[1]/t[4]:.2f}'))
    table('4c. same 2 MiB working set, allocation separated from traffic',
          ['variant', 'S_4'], rows)
    results['rejected_bandwidth_hypothesis'] = rejected

    out = os.path.join(args.out, 'ceiling_lab4.json')
    results['loadavg_end'] = os.getloadavg()
    with open(out, 'w') as fh:
        json.dump(results, fh, indent=1)
    print(f'\nWrote {out}')
    print('Load average start / end:', results['loadavg'], '/', results['loadavg_end'])


if __name__ == '__main__':
    main()
