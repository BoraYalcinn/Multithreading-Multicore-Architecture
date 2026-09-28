#!/usr/bin/env python3
"""Part J - reads results_J.csv, prints median table (P, Tp, Sp, Ep),
writes LaTeX tables and the two required plots per batch size:
  J_time_bs<B>.png     execution time vs thread count
  J_speedup_bs<B>.png  speedup vs thread count, with ideal Sp = P
T1 = the 1-worker run of the SAME parallel structure (as the lab asks).
"""
import csv
from collections import defaultdict
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

times = defaultdict(list)
with open("results_J.csv") as f:
    for r in csv.DictReader(f):
        times[(r["mode"], int(r["bs"]), int(r["P"]))].append(float(r["time"]))
med = {k: median(v) for k, v in times.items()}

seq = med.get(("sequential", 64, 1))
if seq:
    print(f"Original sequential baseline (median of {len(times[('sequential', 64, 1)])}): "
          f"{seq:.4f} s\n")

modes = [m for m in ("create", "pool") if any(k[0] == m for k in med)]
bss = sorted({k[1] for k in med if k[0] != "sequential"})
label = {"create": "create/join per batch (G)", "pool": "persistent pool (K)"}

tex = []
for bs in bss:
    fig_t, ax_t = plt.subplots(figsize=(6, 4))
    fig_s, ax_s = plt.subplots(figsize=(6, 4))
    maxP = 1
    for mode in modes:
        Ps = sorted(k[2] for k in med if k[0] == mode and k[1] == bs)
        if not Ps:
            continue
        T1 = med[(mode, bs, 1)]
        Tp = [med[(mode, bs, P)] for P in Ps]
        Sp = [T1 / t for t in Tp]
        Ep = [s / P for s, P in zip(Sp, Ps)]
        maxP = max(maxP, max(Ps))

        print(f"== {label[mode]}, batch = {bs} ==")
        print(f"{'P':>3} {'Tp (s)':>9} {'Sp':>6} {'Ep':>6}   runs")
        for P, t, s, e in zip(Ps, Tp, Sp, Ep):
            runs = ", ".join(f"{x:.3f}" for x in times[(mode, bs, P)])
            print(f"{P:>3} {t:9.4f} {s:6.2f} {e:6.2f}   [{runs}]")
        print()

        tex.append(f"% {label[mode]}, batch = {bs}")
        tex.append(r"\begin{tabular}{rrrr}\hline")
        tex.append(r"$P$ & $T_P$ (s) & $S_P$ & $E_P$ \\ \hline")
        for P, t, s, e in zip(Ps, Tp, Sp, Ep):
            tex.append(f"{P} & {t:.3f} & {s:.2f} & {e:.2f} \\\\")
        tex.append(r"\hline\end{tabular}" + "\n")

        ax_t.plot(Ps, Tp, "o-", label=label[mode])
        ax_s.plot(Ps, Sp, "o-", label=label[mode])

    if seq and bs == 64:
        ax_t.axhline(seq, ls=":", c="gray", label="original sequential")
    ideal = [p for p in (1, 2, 4, 8, 16) if p <= maxP]
    ax_s.plot(ideal, ideal, "k--", label="ideal $S_P = P$")
    for ax, yl, name in ((ax_t, "Training time (s, median)", "time"),
                         (ax_s, "Speedup $S_P = T_1/T_P$", "speedup")):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16][: [1, 2, 4, 8, 16].index(maxP) + 1])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Threads $P$")
        ax.set_ylabel(yl)
        ax.set_title(f"MNIST NN, batch size {bs}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig_t.tight_layout(); fig_t.savefig(f"J_time_bs{bs}.png", dpi=200)
    fig_s.tight_layout(); fig_s.savefig(f"J_speedup_bs{bs}.png", dpi=200)
    print(f"saved J_time_bs{bs}.png, J_speedup_bs{bs}.png")

with open("J_tables.tex", "w") as f:
    f.write("\n".join(tex))
print("saved J_tables.tex")
