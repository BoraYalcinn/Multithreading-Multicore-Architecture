#!/usr/bin/env python3
"""PC40 Lab 3 - Part E/F analysis. Reads bench_E.csv and writes:
  KPI_E.md                 compact KPI tables (median, min-max, S_T, E_T, throughput, extra memory)
  E_infer_time.png         B=500 inference: time vs T (both decompositions, create and pool)
  E_infer_speedup.png      B=500 inference: speedup vs T + ideal
  E_BltT_speedup.png       B=1 < T: v1 (image) and v2 (rows) speedup curves on the same plot
  E_conv_vs_full.png       convolution alone vs complete inference (create)
  E_train_time.png / E_train_speedup.png   training, batch 32 and 256
and prints the Amdahl fraction phi measured from the T=1 training phase times (Part F).
S_T uses t_1 = the one-worker run of the same mechanism; S_seq uses the sequential reference.
"""
import csv, re
from collections import defaultdict
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("bench_E.csv")))
G = defaultdict(list)          # key -> list of rows
for r in rows:
    mech = "pool" if r["notes"].startswith("pool") else ("create" if r["notes"].startswith("create") else "ref")
    key = (r["version"], r["mode"], r["decomposition"], r["timing_scope"], int(r["images"]), int(r["batch"]),
           mech, int(r["workers_T"]))
    G[key].append(r)


def med(key):
    return median(float(r["seconds"]) for r in G[key])


def spread(key):
    v = [float(r["seconds"]) for r in G[key]]
    return min(v), max(v)


def extra_mb(key):
    m = re.search(r"extra_bytes=(\d+)", G[key][0]["notes"])
    return int(m.group(1)) / 1e6 if m else 0.0


def series(mode, decomp, scope, images, batch, mech):
    Ts = sorted(k[7] for k in G if k[0] == "final" and k[1:7] == (mode, decomp, scope, images, batch, mech))
    return Ts, [("final", mode, decomp, scope, images, batch, mech, T) for T in Ts]


def seq_key(mode, images, batch, scope):
    for k in G:
        if k[0] == "sequential" and k[1] == mode and k[4] == images and k[5] == batch:
            return k


NRUN = max(len(v) for v in G.values())
REP = {int(r["images"]): r["repeats"] for r in rows if r["mode"] == "infer"}
md = [f"# Part E – KPI tables (median of {NRUN} independent runs)\n"]
for k in G:
    if k[0] == "sequential":
        lo, hi = spread(k)
        md.append(f"- sequential reference, {k[1]} images={k[4]} batch={k[5]}: median {med(k):.4f} s "
                  f"[{lo:.4f}–{hi:.4f}]")
md.append("")


def table(title, mode, decomp, scope, images, batch, mech, conv_series=None):
    Ts, keys = series(mode, decomp, scope, images, batch, mech)
    if not Ts:
        return None
    t1 = med(keys[0])
    sk = seq_key(mode, images, batch, scope)
    tseq = med(sk) if sk and scope != "conv" else None
    work = images * int(G[keys[0]][0]["repeats"]) if mode == "infer" else images
    md.append(f"## {title}\n")
    head = "| T | conv median (s) | total median (s) | min–max (s) | S_T = t1/tT | E_T | S vs sequential | throughput (img/s) | extra memory (MB) |"
    md.append(head)
    md.append("|---|---|---|---|---|---|---|---|---|")
    for T, k in zip(Ts, keys):
        tT = med(k); lo, hi = spread(k); S = t1 / tT
        conv = "–"
        if conv_series:
            ck = ("final", mode, decomp, "conv", images, batch, "create", T)
            if ck in G:
                conv = f"{med(ck):.4f}"
        sseq = f"{tseq / tT:.2f}" if tseq else "–"
        md.append(f"| {T} | {conv} | {tT:.4f} | {lo:.4f}–{hi:.4f} | {S:.2f} | {S / T:.2f} | {sseq} | "
                  f"{work / tT:,.0f} | {extra_mb(k):.2f} |")
    md.append("")
    return Ts, [t1 / med(k) for k in keys], [med(k) for k in keys]


def ideal(ax, Ts):
    ax.plot(Ts, Ts, "k--", label="ideal $S_T = T$")


def style(ax, ylabel, title):
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Workers $T$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


# ---- inference, B = 500
fig_t, at = plt.subplots(figsize=(6, 4)); fig_s, asp = plt.subplots(figsize=(6, 4))
for d, name in (("image", "v1 image"), ("rows", "v2 rows")):
    for mech in ("create", "pool"):
        res = table(f"Inference B=500 (synthetic, {REP.get(500)} repeats) – {name}, workers {mech}", "infer", d, "full",
                    500, 500, mech, conv_series=(mech == "create"))
        if res:
            Ts, S, t = res
            at.plot(Ts, t, "o-", label=f"{name}, {mech}")
            asp.plot(Ts, S, "o-", label=f"{name}, {mech}")
sk = seq_key("infer", 500, 500, "full")
if sk:
    at.axhline(med(sk), ls=":", c="gray", label="sequential reference")
ideal(asp, [1, 2, 4, 8, 16])
style(at, f"Time (s, median of {NRUN})", f"Inference, B=500 images × {REP.get(500)} repeats")
style(asp, "Speedup $t_1/t_T$", f"Inference, B=500 images × {REP.get(500)} repeats")
fig_t.tight_layout(); fig_t.savefig("E_infer_time.png", dpi=200)
fig_s.tight_layout(); fig_s.savefig("E_infer_speedup.png", dpi=200)

# ---- convolution alone vs full (create)
fig, ax = plt.subplots(figsize=(6, 4))
for d, name in (("image", "v1 image"), ("rows", "v2 rows")):
    for scope in ("conv", "full"):
        Ts, keys = series("infer", d, scope, 500, 500, "create")
        if Ts:
            t1 = med(keys[0])
            ax.plot(Ts, [t1 / med(k) for k in keys], "o-" if scope == "full" else "s--", label=f"{name}, {scope}")
ideal(ax, [1, 2, 4, 8, 16])
style(ax, "Speedup $t_1/t_T$", "Convolution alone vs complete inference (B=500)")
fig.tight_layout(); fig.savefig("E_conv_vs_full.png", dpi=200)

# ---- B < T : one image
fig, ax = plt.subplots(figsize=(6, 4))
for d, name in (("image", "v1 image"), ("rows", "v2 rows")):
    for mech in ("create", "pool"):
        res = table(f"Inference B=1 < T (one image, {REP.get(1)} repeats) – {name}, workers {mech}", "infer", d, "full",
                    1, 1, mech)
        if res:
            Ts, S, _ = res
            ax.plot(Ts, S, "o-", label=f"{name}, {mech}")
ideal(ax, [1, 2, 4, 8, 16])
style(ax, "Speedup $t_1/t_T$", "B = 1 image < T workers")
fig.tight_layout(); fig.savefig("E_BltT_speedup.png", dpi=200)

# ---- training
fig_t, at = plt.subplots(figsize=(6, 4)); fig_s, asp = plt.subplots(figsize=(6, 4))
phi_lines = []
for b in (32, 256):
    imgs = sorted({k[4] for k in G if k[0] == "final" and k[1] == "train" and k[5] == b})
    if not imgs:
        continue
    n = imgs[0]                                    # number of training images
    res = table(f"Training, MNIST {n} images, batch {b}, image ownership (create per batch)", "train", "image",
                "epoch_loop", n, b, "create")
    Ts, S, t = res
    at.plot(Ts, t, "o-", label=f"batch {b}")
    asp.plot(Ts, S, "o-", label=f"batch {b}")
    sk = seq_key("train", n, b, "epoch_loop")
    if sk:
        at.axhline(med(sk), ls=":", c="C0" if b == 32 else "C1", label=f"sequential, batch {b}")
    phase = lambda key, name: median(float(re.search(name + r"=([\d.e-]+)", r["notes"]).group(1)) for r in G[key])
    # Part F: phi from the T = 1 phase times (median over runs)
    k1 = ("final", "train", "image", "epoch_loop", n, b, "create", 1)
    par, red, upd, tot = phase(k1, "par"), phase(k1, "red"), phase(k1, "upd"), med(k1)
    phi = par / tot
    phi_lines.append(f"- batch {b}: T=1 total {tot:.4f} s = parallel region {par:.4f} + reduction {red:.4f} + "
                     f"update {upd:.4f} + other {tot - par - red - upd:.4f} → **φ = {phi:.3f}**")
    for T, Smeas in zip(Ts, S):
        kT = ("final", "train", "image", "epoch_loop", n, b, "create", T)
        model = 1 / ((1 - phi) + phi / T)
        phi_lines.append(f"  - T={T}: measured S={Smeas:.2f}, Amdahl model S={model:.2f}; "
                         f"parallel region {phase(kT, 'par'):.4f} s, reduction {phase(kT, 'red'):.4f} s")
ideal(asp, [1, 2, 4, 8, 16])
style(at, f"Epoch time (s, median of {NRUN})", "Training (MNIST), image ownership")
style(asp, "Speedup $t_1/t_T$", "Training (MNIST), image ownership")
fig_t.tight_layout(); fig_t.savefig("E_train_time.png", dpi=200)
fig_s.tight_layout(); fig_s.savefig("E_train_speedup.png", dpi=200)

md.append("## Part F – Amdahl fraction measured from the T=1 training phases\n")
md += phi_lines
open("KPI_E.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
print("\nsaved KPI_E.md and E_*.png")
