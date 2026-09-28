#!/usr/bin/env python3
"""Part J - scalability benchmark. Writes results_J.csv.

For every mode (create, pool), batch size and P in {1,2,4,8,16}: RUNS runs.
Also runs the original sequential baseline (reported separately).

Usage:  python3 bench_J.py                       (MNIST in ../data, bs = 64)
        python3 bench_J.py --bs 32 64 128 256    (optional batch-size sweep)
        python3 bench_J.py --synthetic           (quick smoke test)
"""
import csv, os, re, subprocess, sys

RUNS = 3
THREADS = [1, 2, 4, 8, 16]
MODES = ["create", "pool"]

args = sys.argv[1:]
batch_sizes = [64]
if "--bs" in args:
    i = args.index("--bs")
    j = i + 1
    while j < len(args) and args[j].isdigit():
        j += 1
    batch_sizes = [int(x) for x in args[i + 1:j]]
    del args[i:j]
data = args if args else ["../data"]


def seconds(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return float(re.search(r"Training time: ([\d.eE+-]+)", out).group(1))


# Record the machine (Part J asks for physical/logical cores).
lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
info = {k.strip(): v.strip() for k, v in
        (l.split(":", 1) for l in lscpu.splitlines() if ":" in l)}
print("CPU:", info.get("Model name"))
print("Logical CPUs:", os.cpu_count(),
      "| Threads/core:", info.get("Thread(s) per core"),
      "| Cores/socket:", info.get("Core(s) per socket"),
      "| Sockets:", info.get("Socket(s)"))
with open("machine_J.txt", "w") as f:
    f.write(lscpu)

rows = []
for r in range(RUNS):
    t = seconds(["./nn_sequential", *data])
    rows.append(["sequential", 64, 1, r + 1, t])
    print(f"sequential run {r + 1}: {t:.4f} s")

for bs in batch_sizes:
    for mode in MODES:
        for P in THREADS:
            for r in range(RUNS):
                t = seconds(["./nn_IJ", *data, "--threads", str(P),
                             "--mode", mode, "--bs", str(bs)])
                rows.append([mode, bs, P, r + 1, t])
                print(f"{mode:6} bs={bs:3} P={P:2} run {r + 1}: {t:.4f} s", flush=True)

with open("results_J.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mode", "bs", "P", "run", "time"])
    w.writerows(rows)
print("\nSaved results_J.csv and machine_J.txt  ->  now run: python3 plot_J.py")
