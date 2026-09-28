#!/usr/bin/env python3
"""Part I - correctness validation.

Runs nn_IJ for P = 1, 2, 4 and compares every P against P = 1:
  epoch losses, final test accuracy, weight checksum (sum and sum|w|).

Usage:  python3 check_I.py                 (MNIST in ../data, mode create)
        python3 check_I.py --synthetic
        python3 check_I.py --mode pool
"""
import re, subprocess, sys

BIN = "./nn_IJ"
THREADS = [1, 2, 4]

# Tolerances (explained in the report):
TOL_LOSS_REL = 1e-4   # epoch loss: relative difference
TOL_ACC_PP   = 0.1    # test accuracy: absolute, in percentage points (= 2 of 2000 test images)
TOL_SUM_REL  = 1e-4   # checksum: relative difference

args = sys.argv[1:]
mode = "create"
if "--mode" in args:
    i = args.index("--mode"); mode = args[i + 1]; del args[i:i + 2]
data = args if args else ["../data"]


def run(P):
    out = subprocess.run([BIN, *data, "--threads", str(P), "--mode", mode],
                         capture_output=True, text=True, check=True).stdout
    return {
        "loss": [float(x) for x in re.findall(r"loss=([\d.eE+-]+)", out)],
        "acc":  float(re.search(r"Test accuracy: ([\d.]+)", out).group(1)),
        "sum":  float(re.search(r"sum=([\d.eE+-]+)", out).group(1)),
        "abs":  float(re.search(r"abs_sum=([\d.eE+-]+)", out).group(1)),
    }


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


res = {P: run(P) for P in THREADS}
ref = res[1]
ok_all = True

print(f"mode = {mode}\n")
print(f"{'P':>2} | {'loss ep1':>12} {'loss ep2':>12} {'loss ep3':>12} | "
      f"{'test acc':>8} | {'checksum sum':>18} | {'checksum |w|':>18}")
print("-" * 100)
for P, r in res.items():
    L = " ".join(f"{x:12.9f}" for x in r["loss"])
    print(f"{P:>2} | {L} | {r['acc']:7.2f}% | {r['sum']:18.12f} | {r['abs']:18.12f}")

print("\nDifferences vs P=1:")
print(f"{'P':>2} | {'max rel loss':>12} | {'acc diff (pp)':>13} | "
      f"{'rel sum':>10} | {'rel |w|':>10} | verdict")
print("-" * 75)
for P, r in res.items():
    if P == 1:
        continue
    dl = max(rel(a, b) for a, b in zip(r["loss"], ref["loss"]))
    da = abs(r["acc"] - ref["acc"])
    ds = rel(r["sum"], ref["sum"])
    dw = rel(r["abs"], ref["abs"])
    ok = dl <= TOL_LOSS_REL and da <= TOL_ACC_PP and ds <= TOL_SUM_REL and dw <= TOL_SUM_REL
    ok_all &= ok
    print(f"{P:>2} | {dl:12.3e} | {da:13.2f} | {ds:10.3e} | {dw:10.3e} | "
          f"{'PASS' if ok else 'FAIL'}")

print(f"\nTolerances: loss rel <= {TOL_LOSS_REL:g}, test acc <= {TOL_ACC_PP} pp, "
      f"checksum rel <= {TOL_SUM_REL:g}")
print("ALL PASS" if ok_all else "SOMETHING FAILED -> likely a bug")
sys.exit(0 if ok_all else 1)
