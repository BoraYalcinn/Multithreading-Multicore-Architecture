"""The machine-ceiling figure, built from ceiling_lab4.json."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'ceiling_lab4.json')))

figa, axa = plt.subplots(figsize=(4.6, 3.0))
figb, axb = plt.subplots(figsize=(4.4, 3.0))

# --- left: the ceiling of the whole machine, on the friendliest workload -----
c = d['ceiling_all']
ps = sorted(int(p) for p in c)
sp = [c[str(p)]['speedup'] for p in ps]
busy = [c[str(p)]['cores_busy'] for p in ps]
axa.plot(ps, ps, 'k:', lw=1, label='Ideal $S_p=p$')
axa.plot(ps, busy, '^-', color='tab:gray', ms=4, label='Cores actually busy (CPU/wall)')
axa.plot(ps, sp, 'o-', color='tab:green', ms=5, label='Speedup obtained')
axa.annotate(f'{sp[-1]:.2f}$\\times$ from {busy[-1]:.1f} busy cores',
             (ps[-1], sp[-1]), textcoords='offset points', xytext=(-104, 14),
             fontsize=7.5, color='tab:red',
             arrowprops=dict(arrowstyle='->', color='tab:red', lw=0.8))
axa.set_xscale('log', base=2); axa.set_xticks(ps); axa.set_xticklabels(ps)
axa.set_xlabel('Intra-op threads $p$')
axa.set_ylabel('$S_p$  /  cores busy')
axa.set_title('Machine ceiling: compute-bound GEMM $2048^3$', fontsize=9)
axa.set_ylim(0, 17); axa.grid(alpha=0.25); axa.legend(fontsize=7, loc='upper left')

# --- right: homogeneous subsets separate hardware from decomposition --------
h = d['homogeneous']
styles = {'P-cores only': ('tab:blue', 'o'), 'E-cores only': ('tab:orange', 's'),
          'all CPUs (heterogeneous)': ('tab:red', '^')}
for label, rec in h.items():
    color, mark = styles[label]
    ps2 = sorted(int(p) for p in rec['gemm_speedup'])
    axb.plot(ps2, [rec['gemm_speedup'][str(p)] for p in ps2], mark + '-', color=color,
             ms=4, lw=1.4, label=f'GEMM, {label}')
    axb.plot(ps2, [rec['head_speedup'][str(p)] for p in ps2], mark + '--', color=color,
             ms=4, lw=1.4, alpha=0.75, label=f'Per-head pool, {label}')
axb.plot([1, 2, 4, 8], [1, 2, 4, 8], 'k:', lw=1, label='Ideal $S_p=p$')
axb.set_xscale('log', base=2); axb.set_xticks([1, 2, 4, 8]); axb.set_xticklabels([1, 2, 4, 8])
axb.set_xlabel('Workers / intra-op threads $p$')
axb.set_ylabel('$S_p$')
axb.set_title('Homogeneous cores: hardware vs decomposition', fontsize=9)
axb.set_ylim(0, 8.6); axb.grid(alpha=0.25); axb.legend(fontsize=5.8, loc='upper left', ncol=1)

for fig, name in ((figa, 'L4_ceiling.png'), (figb, 'L4_homog.png')):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    print('wrote', path)
