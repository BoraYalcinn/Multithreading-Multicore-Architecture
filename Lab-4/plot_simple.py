"""The two plots the synthesis needs: execution time and speedup versus p."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'bench_lab4.json')))
P = d['p_values']
R = {(r['series'], r['p']): r for r in d['rows']}

series = [('pool_H16',   'Per-head pool, H=16',       'o-',  'tab:blue'),
          ('pool_H4',    'Per-head pool, H=4',        'o--', 'tab:cyan'),
          ('vector_H16', 'Vectorized intra-op, H=16', 's-',  'tab:orange'),
          ('vector_H4',  'Vectorized intra-op, H=4',  's--', 'tab:red')]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.1))

# left: execution time, median with min-max bars
for name, label, style, color in series:
    med = [R[(name, p)]['median_ms'] for p in P]
    lo = [med[i] - R[(name, p)]['best_ms'] for i, p in enumerate(P)]
    hi = [R[(name, p)]['worst_ms'] - med[i] for i, p in enumerate(P)]
    ax1.errorbar(P, med, yerr=[lo, hi], fmt=style, color=color, label=label,
                 ms=4, capsize=3, lw=1.3, elinewidth=0.8)
ax1.set_yscale('log')
ax1.set_ylabel('Execution time (ms), log scale')
ax1.set_title('Execution time: median, bars span min–max', fontsize=9.5)

# right: speedup, each series on its own T1
for name, label, style, color in series:
    t1 = R[(name, P[0])]['median_ms']
    ax2.plot(P, [t1 / R[(name, p)]['median_ms'] for p in P], style,
             color=color, label=label, ms=4, lw=1.3)
ax2.plot(P, P, 'k:', lw=1, label='Ideal $S_p = p$')
ax2.set_ylabel('$S_p^{\\mathrm{fixed,rel}} = T_1 / T_p$')
ax2.set_ylim(0, 4.2)
ax2.set_title('Speedup, each curve on its own $T_1$', fontsize=9.5)

for ax in (ax1, ax2):
    ax.set_xscale('log', base=2); ax.set_xticks(P); ax.set_xticklabels(P)
    ax.set_xlabel('Workers / intra-op threads $p$')
    ax.grid(alpha=0.25, which='both')
ax1.legend(fontsize=6.8, loc='lower left')

path = os.path.join(HERE, 'L4_time_speedup.png')
fig.savefig(path, dpi=160, bbox_inches='tight')
print('wrote', path)
