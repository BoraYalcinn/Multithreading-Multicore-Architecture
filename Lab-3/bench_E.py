#!/usr/bin/env python3
"""PC40 Lab 3 - Part E benchmark driver. Writes every run to bench_E.csv
(columns of benchmark_template.csv) and the machine description to machine_E.txt.

Each measurement is a separate invocation of the program (5 independent runs per
configuration, as required by Section 2.4). Workloads:
  infer large : --synthetic --test 500 --repeats 100        (B=500 images, ~1 s sequential)
  infer B<T   : --synthetic --test 1   --repeats 30000      (one image, ~0.6-1 s sequential)
  train       : MNIST --train 10000 --epochs 1, batch 32 and batch 256
Usage: python3 bench_E.py            (full study, several minutes)
       python3 bench_E.py --quick    (2 runs, smaller repeats: only to test the pipeline)
"""
import os, re, subprocess, sys

QUICK = "--quick" in sys.argv
RUNS = 2 if QUICK else 5
THREADS = [1, 2, 4, 8, 16]
LARGE = ["--synthetic", "--mode", "infer", "--test", "500", "--filters", "8", "--seed", "42",
         "--repeats", "10" if QUICK else "100"]
SMALL = ["--synthetic", "--mode", "infer", "--test", "1", "--filters", "8", "--seed", "42",
         "--repeats", "2000" if QUICK else "30000"]
TRAIN_N = "2000" if QUICK else "10000"
TRAIN = ["--data", "data/mnist", "--mode", "train", "--train", TRAIN_N, "--test", "100",
         "--epochs", "1", "--filters", "8", "--seed", "42", "--lr", "0.05"]
CSV = "bench_E.csv"
HEADER = ("language,version,decomposition,dataset,mode,images,batch,repeats,workers_T,run_index,"
          "timing_scope,seconds,images_per_s,checksum_or_final_loss,notes\n")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def seq_infer(args, label):
    for r in range(RUNS):
        out = run(["./cnn_seq", *args])
        sec = float(re.search(r"total_s=([\d.]+)", out).group(1))
        ck = re.search(r"checksum=([\d.]+)", out).group(1)
        n = int(args[args.index("--test") + 1]); rep = int(args[args.index("--repeats") + 1])
        with open(CSV, "a") as f:
            f.write(f"C++,sequential,none,synthetic,infer,{n},{n},{rep},1,{r+1},full,{sec},{n*rep/sec},{ck},reference\n")
        print(f"seq   infer {label:5} run {r+1}: {sec:.4f} s  checksum={ck}", flush=True)


def seq_train(batch):
    for r in range(RUNS):
        out = run(["./cnn_seq", *TRAIN, "--batch", str(batch)])
        m = re.search(r"train_loss=([\d.]+).*time_s=([\d.]+)", out)
        loss, sec = m.group(1), float(m.group(2))
        with open(CSV, "a") as f:
            f.write(f"C++,sequential,none,MNIST,train,{TRAIN_N},{batch},1,1,{r+1},epoch_loop,{sec},"
                    f"{int(TRAIN_N)/sec},{loss},reference\n")
        print(f"seq   train b={batch:<4} run {r+1}: {sec:.4f} s  loss={loss}", flush=True)


def final(args, extra, label):
    for T in THREADS:
        for r in range(RUNS):
            out = run(["./cnn_final", *args, *extra, "--threads", str(T), "--output-csv", CSV])
            sec = re.search(r"(?:total_s|time_s)=([\d.]+)", out).group(1)
            tail = re.search(r"(checksum=[\d.]+|final_train_loss=[\d.]+)", out).group(1)
            print(f"final {label:28} T={T:2} run {r+1}: {float(sec):.4f} s  {tail}", flush=True)


if not (os.path.exists("cnn_seq") and os.path.exists("cnn_final")):
    sys.exit("build cnn_seq and cnn_final first")
with open(CSV, "w") as f:
    f.write(HEADER)
with open("machine_E.txt", "w") as f:
    f.write(run(["lscpu"]) + "\n" + run(["g++", "--version"]))

# 1) sequential references
seq_infer(LARGE, "B=500")
seq_infer(SMALL, "B=1")
for b in (32, 256):
    seq_train(b)
# 2) inference, large workload: both decompositions, create and pool, full scope
for d in ("image", "rows"):
    for pool in ([], ["--pool"]):
        final(LARGE, ["--decomp", d, *pool], f"infer B=500 {d} {'pool' if pool else 'create'}")
# 3) convolution alone (create), large workload
for d in ("image", "rows"):
    final(LARGE, ["--decomp", d, "--scope", "conv"], f"conv  B=500 {d} create")
# 4) B < T: one image
for d in ("image", "rows"):
    for pool in ([], ["--pool"]):
        final(SMALL, ["--decomp", d, *pool], f"infer B=1 {d} {'pool' if pool else 'create'}")
# 5) training (image ownership, create per batch), MNIST
for b in (32, 256):
    final(TRAIN, ["--batch", str(b)], f"train b={b}")
print(f"\nSaved {CSV} and machine_E.txt -> now run: python3 plot_E.py")
