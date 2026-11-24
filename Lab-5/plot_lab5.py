"""Figures for the PC40 Lab 05 synthesis, built from bench_lab5.json."""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'bench_lab5.json')))


def curve(block, prefix, t_seq):
    ps = sorted(int(k.split('_')[1]) for k in block if k.startswith(prefix))
    return ps, [t_seq / block[f'{prefix}_{p}']['median'] for p in ps]


def busy(block, prefix):
    ps = sorted(int(k.split('_')[1]) for k in block if k.startswith(prefix))
    return ps, [block[f'{prefix}_{p}']['busy'] for p in ps]


# ---- Figure 1: time and speedup, processes versus threads -----------------
proc, thr = d['processes_A'], d['threads_A']
t_seq_p, t_seq_t = proc['sequential']['median'], thr['sequential']['median']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.1))

for block, prefix, t_seq, label, style, color in (
        (proc, 'proc', t_seq_p, 'Process pool', 'o-', 'tab:green'),
        (thr, 'thread', t_seq_t, 'Thread pool', 's-', 'tab:red')):
    ps = sorted(int(k.split('_')[1]) for k in block if k.startswith(prefix))
    med = [block[f'{prefix}_{p}']['median'] for p in ps]
    lo = [med[i] - block[f'{prefix}_{p}']['best'] for i, p in enumerate(ps)]
    hi = [block[f'{prefix}_{p}']['worst'] - med[i] for i, p in enumerate(ps)]
    ax1.errorbar(ps, med, yerr=[lo, hi], fmt=style, color=color, label=label,
                 ms=4, capsize=3, lw=1.3, elinewidth=0.8)
    ax2.plot(*curve(block, prefix, t_seq), style, color=color, label=label, ms=4, lw=1.3)

ax1.axhline(t_seq_p, color='k', ls=':', lw=1, label='Sequential reference')
ax1.set_yscale('log')
ax1.set_ylabel('Execution time (s), log scale')
ax1.set_title('Nested CV, decision tree: median, bars span min–max', fontsize=9.5)
ax1.legend(fontsize=7.5)

allp = sorted(int(k.split('_')[1]) for k in proc if k.startswith('proc'))
ax2.plot(allp, allp, 'k:', lw=1, label='Ideal $S_p=p$')
ax2.axhline(1.0, color='gray', lw=0.8, ls='--')
ax2.set_ylabel('$S_p = T_{\\mathrm{seq}} / T_p$')
ax2.set_ylim(0, 9)
ax2.set_title('Speedup against the sequential reference', fontsize=9.5)
ax2.legend(fontsize=7.5, loc='upper left')

for ax in (ax1, ax2):
    ax.set_xscale('log', base=2)
    ax.set_xticks(allp); ax.set_xticklabels(allp)
    ax.set_xlabel('Workers $p$')
    ax.grid(alpha=0.25, which='both')

path = os.path.join(HERE, 'L5_time_speedup.png')
fig.savefig(path, dpi=160, bbox_inches='tight')
plt.close(fig)
print('wrote', path)


# ---- Figure 2: did the workers run, and the task-count effect -------------
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.2, 3.1))

ps_t, busy_t = busy(thr, 'thread')
ps_c, busy_c = busy(d['control_threads'], 'ctl')
axa.plot(ps_c, busy_c, 'o-', color='tab:blue', ms=4,
         label='numpy matmul (releases the GIL)')
axa.plot(ps_t, busy_t, 's-', color='tab:red', ms=4,
         label='nested CV with scikit-learn')
axa.plot(ps_c, ps_c, 'k:', lw=1, label='One core per worker')
axa.set_xscale('log', base=2); axa.set_xticks(ps_c); axa.set_xticklabels(ps_c)
axa.set_xlabel('Thread-pool workers $p$')
axa.set_ylabel('Cores busy (process CPU time / wall time)')
axa.set_title('Do the threads actually run?', fontsize=9.5)
axa.set_ylim(0, 9); axa.grid(alpha=0.25); axa.legend(fontsize=7, loc='upper left')

gran = d['granularity']
labels = list(gran.keys())
tasks = [gran[k]['tasks'] for k in labels]
times = [gran[k]['best'] for k in labels]
bars = axb.bar(range(len(labels)), times,
               color=['tab:green', 'tab:olive', 'tab:red'])
axb.axhline(d['processes_A']['sequential']['best'], color='k', ls=':', lw=1,
            label='Sequential reference')
axb.set_xticks(range(len(labels)))
axb.set_xticklabels([f'{k.split()[0]}\n{n} tasks' for k, n in zip(labels, tasks)],
                    fontsize=8)
axb.set_ylabel('Execution time (s)')
axb.set_title('Same work, same 4 workers, different task size', fontsize=9.5)
for bar, t in zip(bars, times):
    axb.text(bar.get_x() + bar.get_width() / 2, t, f'{t:.2f} s',
             ha='center', va='bottom', fontsize=8)
axb.grid(alpha=0.2, axis='y'); axb.legend(fontsize=7.5)

path = os.path.join(HERE, 'L5_gil_granularity.png')
fig.savefig(path, dpi=160, bbox_inches='tight')
plt.close(fig)
print('wrote', path)
