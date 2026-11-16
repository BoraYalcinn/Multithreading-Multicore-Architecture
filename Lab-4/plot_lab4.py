"""Figures for the PC40 Lab04 synthesis, built from bench_lab4.json."""

import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, 'bench_lab4.json')))
P = data['p_values']
rows = {}
for r in data['rows']:
    rows[(r['series'], r['p'])] = r


def series(name, field='best_ms'):
    base = rows[(name, P[0])][field]
    return [base / rows[(name, p)][field] for p in P]


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


# --- Figure 1: the two decompositions, both estimators ---------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.95), sharey=True)
curves = [('pool_H16', 'Per-head pool, H=16', 'o-', 'tab:blue'),
          ('pool_H4', 'Per-head pool, H=4', 'o--', 'tab:cyan'),
          ('vector_H16', 'Vectorized intra-op, H=16', 's-', 'tab:orange'),
          ('vector_H4', 'Vectorized intra-op, H=4', 's--', 'tab:red')]
for name, label, style, color in curves:
    ax1.plot(P, series(name, 'best_ms'), style, color=color, label=label, ms=5)
ax1.plot(P, P, 'k:', lw=1, label='Ideal $S_p=p$')
ax1.set_title('Fastest of 7 interleaved trials')

for name, label, style, color in curves:
    med = series(name, 'median_ms')
    lo = [rows[(name, P[0])]['median_ms'] / rows[(name, p)]['worst_ms'] for p in P]
    hi = [rows[(name, P[0])]['median_ms'] / rows[(name, p)]['best_ms'] for p in P]
    ax2.errorbar(P, med, yerr=[[m - l for m, l in zip(med, lo)],
                               [h - m for m, h in zip(med, hi)]],
                 fmt=style, color=color, label=label, ms=5, capsize=3, lw=1.2, elinewidth=0.8)
ax2.plot(P, P, 'k:', lw=1, label='Ideal $S_p=p$')
ax2.set_title('Median of 7 trials, bars span min–max')

for ax in (ax1, ax2):
    ax.set_xscale('log', base=2); ax.set_xticks(P); ax.set_xticklabels(P)
    ax.set_xlabel('Configured workers / intra-op threads $p$')
    ax.grid(alpha=0.25)
ax1.set_ylabel('$S_p^{\\mathrm{fixed,rel}} = T_1/T_p$')
ax1.set_ylim(0, 4.0)   # the ideal line leaves the panel; the data would be unreadable otherwise
ax1.legend(fontsize=7.5, loc='upper left')
save(fig, 'L4_speedup.png')


# --- Figure 2: the control experiment --------------------------------------
fig, ax = plt.subplots(figsize=(4.4, 2.95))
for name, label, style, color in (
        ('ctl_gemm', 'Compute-bound GEMM $1024^3$', 'o-', 'tab:green'),
        ('ctl_head', 'Lab04 attention head, $L$=512', 's-', 'tab:blue'),
        ('ctl_python', 'Pure Python loop', '^-', 'tab:red')):
    ax.plot(P, series(name, 'best_ms'), style, color=color, label=label, ms=5)
ax.plot(P, P, 'k:', lw=1, label='Ideal $S_p=p$')
ax.set_xscale('log', base=2); ax.set_xticks(P); ax.set_xticklabels(P)
ax.set_xlabel('Pool workers $p$ (16 tasks, intra-op = 1)')
ax.set_ylabel('$S_p^{\\mathrm{fixed,rel}}$')
ax.set_title('Same pool, three kinds of task')
ax.set_ylim(0, 6.2); ax.grid(alpha=0.25); ax.legend(fontsize=8, loc='upper left')
save(fig, 'L4_control.png')


# --- Figure 3: sequence-length scaling -------------------------------------
lengths = sorted(int(s.split('_')[1]) for s, p in rows if s.startswith('len_'))
times = [rows[(f'len_{sl}', 1)]['median_ms'] for sl in lengths]
fig, ax = plt.subplots(figsize=(5.2, 3.9))
ax.plot(lengths, times, 'o-', color='tab:purple', label='Measured, 1 intra-op thread')
ref = [times[0] * (sl / lengths[0]) ** 2 for sl in lengths]
ax.plot(lengths, ref, 'k:', lw=1, label='$\\propto L^2$ from the first point')
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xticks(lengths); ax.set_xticklabels(lengths)
ax.set_xlabel('Sequence length $L$')
ax.set_ylabel('Latency (ms)')
ax.set_title('Dense attention, $B$=2, $H$=4, $d_h$=64')
ax.grid(alpha=0.25, which='both'); ax.legend(fontsize=8)
for sl, t in zip(lengths, times):
    mib = 2 * 4 * sl * sl * 4 / 1024**2
    ax.annotate(f'{mib:.1f} MiB', (sl, t), textcoords='offset points',
                xytext=(4, -10), fontsize=6.5, color='dimgray')
save(fig, 'L4_length.png')


# --- Figure 4: prefill, decoding and the KV cache ---------------------------
L_gen = 256
pf = rows[('prefill', 1)]['median_ms']
st = rows[('decode_step', 1)]['median_ms']
nc = rows[('decode_no_cache', 1)]['median_ms']
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9, 3.6))
axa.bar(['Prefill\n256 positions', 'Decode step\nKV cache', 'Decode step\nno cache'], [pf, st, nc],
        color=['tab:blue', 'tab:green', 'tab:red'])
axa.set_ylabel('Latency (ms)')
axa.set_title('Cost of one step')
for i, val in enumerate([pf, st, nc]):
    axa.text(i, val, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
axa.grid(alpha=0.2, axis='y')

axb.bar(['Prefill', f'{L_gen} steps\nKV cache', f'{L_gen} steps\nno cache'],
        [pf, L_gen * st, L_gen * nc], color=['tab:blue', 'tab:green', 'tab:red'])
axb.set_yscale('log'); axb.set_ylabel('Latency (ms, log scale)')
axb.set_title(f'Generating {L_gen} tokens')
for i, val in enumerate([pf, L_gen * st, L_gen * nc]):
    axb.text(i, val, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
axb.grid(alpha=0.2, axis='y')
save(fig, 'L4_generation.png')


# --- Figure 5: pool dispatch overhead --------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for tasks, color in (('4tasks', 'tab:cyan'), ('16tasks', 'tab:blue')):
    ax.plot(P, [rows[(f'dispatch_{tasks}', p)]['median_ms'] * 1000 for p in P],
            'o-', color=color, label=f'{tasks[:-5]} empty tasks', ms=5)
ax.set_xscale('log', base=2); ax.set_xticks(P); ax.set_xticklabels(P)
ax.set_xlabel('Pool workers $p$')
ax.set_ylabel('Dispatch + join (µs)')
ax.set_title('Cost of pool.map with no work in the tasks')
ax.grid(alpha=0.25); ax.legend(fontsize=8)
save(fig, 'L4_dispatch.png')
