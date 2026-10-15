#!/usr/bin/env python3
# PC40 Lab 03 J.Gaber (gaber@utbm.fr)
# sequential CNN reference (v6)

import argparse
import math
import os
import struct
import time
import csv
H = W = 28
KH = KW = 3
OH = OW = 26
PH = PW = 13
CLASSES = 10

class RNG:

    def __init__(self, seed):
        self.state = seed & 4294967295

    def u32(self):
        self.state = 1664525 * self.state + 1013904223 & 4294967295
        return self.state

    def uniform(self, a=-1.0, b=1.0):
        return a + (b - a) * (self.u32() / 4294967296.0)

    def randint(self, n):
        return self.u32() % n

class Sample:
    __slots__ = ('x', 'y')

    def __init__(self, x, y):
        self.x = x
        self.y = y

class Params:

    def __init__(self, F, rng):
        self.F = F
        self.K = [rng.uniform(-0.08, 0.08) for _ in range(F * 9)]
        self.b = [0.0] * F
        n = F * PH * PW
        self.W = [rng.uniform(-0.05, 0.05) for _ in range(CLASSES * n)]
        self.c = [0.0] * CLASSES

class Grad:

    def __init__(self, F):
        self.F = F
        self.K = [0.0] * (F * 9)
        self.b = [0.0] * F
        self.W = [0.0] * (CLASSES * F * PH * PW)
        self.c = [0.0] * CLASSES

    def zero(self):
        self.K[:] = [0.0] * len(self.K)
        self.b[:] = [0.0] * len(self.b)
        self.W[:] = [0.0] * len(self.W)
        self.c[:] = [0.0] * len(self.c)

class Workspace:

    def __init__(self, F):
        self.F = F
        self.S = [0.0] * (F * OH * OW)
        self.A = [0.0] * (F * OH * OW)
        self.P = [0.0] * (F * PH * PW)
        self.win = [0] * (F * PH * PW)
        self.z = [0.0] * CLASSES
        self.p = [0.0] * CLASSES

def sidx(f, r, q):
    return (f * OH + r) * OW + q

def pidx(f, r, q):
    return (f * PH + r) * PW + q

def forward(x, p, w):
    F = p.F
    for f in range(F):
        kb = f * 9
        for r in range(OH):
            for q in range(OW):
                acc = p.b[f]
                for u in range(3):
                    xb = (r + u) * W + q
                    kb2 = kb + u * 3
                    acc += p.K[kb2] * x[xb] + p.K[kb2 + 1] * x[xb + 1] + p.K[kb2 + 2] * x[xb + 2]
                ii = sidx(f, r, q)
                w.S[ii] = acc
                w.A[ii] = acc if acc > 0 else 0.0
    for f in range(F):
        for r in range(PH):
            for q in range(PW):
                br = 2 * r
                bq = 2 * q
                best = -1e+300
                besti = 0
                for dr in range(2):
                    for dq in range(2):
                        ii = sidx(f, br + dr, bq + dq)
                        val = w.A[ii]
                        if val > best:
                            best = val
                            besti = ii
                pi = pidx(f, r, q)
                w.P[pi] = best
                w.win[pi] = besti
    N = F * PH * PW
    for k in range(CLASSES):
        acc = p.c[k]
        off = k * N
        for j in range(N):
            acc += p.W[off + j] * w.P[j]
        w.z[k] = acc
    m = max(w.z)
    den = 0.0
    for k in range(CLASSES):
        w.p[k] = math.exp(w.z[k] - m)
        den += w.p[k]
    for k in range(CLASSES):
        w.p[k] /= den
    return w

def loss(w, y):
    m = max(w.z)
    den = sum((math.exp(v - m) for v in w.z))
    return -(w.z[y] - m) + math.log(den)

def backward_accumulate(x, y, p, w, g):
    F = p.F
    N = F * PH * PW
    dz = w.p[:]
    dz[y] -= 1.0
    dP = [0.0] * N
    for k in range(CLASSES):
        off = k * N
        d = dz[k]
        g.c[k] += d
        for j in range(N):
            g.W[off + j] += d * w.P[j]
            dP[j] += p.W[off + j] * d
    dA = [0.0] * (F * OH * OW)
    for j, v in enumerate(dP):
        dA[w.win[j]] += v
    for f in range(F):
        kb = f * 9
        for r in range(OH):
            for q in range(OW):
                ii = sidx(f, r, q)
                ds = dA[ii] if w.S[ii] > 0 else 0.0
                g.b[f] += ds
                for u in range(3):
                    xb = (r + u) * W + q
                    ko = kb + u * 3
                    g.K[ko] += ds * x[xb]
                    g.K[ko + 1] += ds * x[xb + 1]
                    g.K[ko + 2] += ds * x[xb + 2]

def sgd_update(p, g, lr, B):
    a = lr / B
    for i in range(len(p.K)):
        p.K[i] -= a * g.K[i]
    for i in range(len(p.b)):
        p.b[i] -= a * g.b[i]
    for i in range(len(p.W)):
        p.W[i] -= a * g.W[i]
    for i in range(len(p.c)):
        p.c[i] -= a * g.c[i]

def synthetic(n, seed):
    rng = RNG(seed)
    out = []
    centers = [(4, 4), (4, 12), (4, 20), (10, 7), (10, 17), (16, 4), (16, 12), (16, 20), (22, 8), (22, 18)]
    for i in range(n):
        y = i % 10
        x = [0.0] * (H * W)
        cr, cc = centers[y]
        for r in range(H):
            for q in range(W):
                x[r * W + q] = 0.03 * rng.uniform(0, 1)
        for r in range(max(0, cr - 2), min(H, cr + 3)):
            for q in range(max(0, cc - 2), min(W, cc + 3)):
                x[r * W + q] = 0.75 + 0.25 * rng.uniform(0, 1)
        out.append(Sample(x, y))
    return out

def read_be32(f):
    return struct.unpack('>I', f.read(4))[0]

def load_mnist(path, train, n):
    ip = os.path.join(path, 'train-images-idx3-ubyte' if train else 't10k-images-idx3-ubyte')
    lp = os.path.join(path, 'train-labels-idx1-ubyte' if train else 't10k-labels-idx1-ubyte')
    with open(ip, 'rb') as fi, open(lp, 'rb') as fl:
        if read_be32(fi) != 2051 or read_be32(fl) != 2049:
            raise ValueError('bad MNIST magic')
        ni = read_be32(fi)
        rows = read_be32(fi)
        cols = read_be32(fi)
        nl = read_be32(fl)
        if rows != 28 or cols != 28 or ni != nl:
            raise ValueError('bad MNIST dimensions')
        n = min(n, ni)
        out = []
        for _ in range(n):
            raw = fi.read(784)
            lab = fl.read(1)
            out.append(Sample([b / 255.0 for b in raw], lab[0]))
        return out

def evaluate(data, p):
    w = Workspace(p.F)
    ok = 0
    ls = 0.0
    for s in data:
        forward(s.x, p, w)
        ls += loss(w, s.y)
        ok += max(range(10), key=lambda k: w.p[k]) == s.y
    return (ls / len(data), ok / len(data))

def train_epoch(data, p, batch, lr):
    w = Workspace(p.F)
    g = Grad(p.F)
    total = 0.0
    for lo in range(0, len(data), batch):
        hi = min(len(data), lo + batch)
        g.zero()
        subtotal = 0.0
        for i in range(lo, hi):
            s = data[i]
            forward(s.x, p, w)
            subtotal += loss(w, s.y)
            backward_accumulate(s.x, s.y, p, w, g)
        sgd_update(p, g, lr, hi - lo)
        total += subtotal
    return total / len(data)

def self_test():
    """Three fixtures: a hand-computed convolution, forward() against a naive
    direct implementation, and the pooling tie rule (first maximum, row-major)."""
    ok = True
    # 1. Hand-computed 2 x 2 convolution on a 4 x 4 image.
    x = [1, 2, 0, 1, 3, 1, 2, 2, 0, 1, 3, 1, 2, 2, 1, 0]
    K = [1, 0, -1, 2]
    exp = [0, 5, 2, 5, 6, 1, 2, 1, 2]
    got = []
    for r in range(3):
        for q in range(3):
            got.append(sum((K[u * 2 + v] * x[(r + u) * 4 + q + v] for u in range(2) for v in range(2))))
    e1 = max((abs(a - b) for a, b in zip(got, exp)))
    ok &= e1 <= 1e-12
    print(f'self-test 1 hand convolution      max_error={e1:.3e}')
    # 2. Compare the forward pass with a direct implementation.
    rng = RNG(42)
    p = Params(2, rng)
    p.b = [0.01, -0.02]
    p.c = [0.001 * k for k in range(CLASSES)]
    s = synthetic(1, 7)[0]
    w = Workspace(2)
    forward(s.x, p, w)
    N = 2 * PH * PW
    eS = eP = 0.0
    z = [0.0] * CLASSES
    P = [0.0] * N
    for f in range(2):
        for r in range(OH):
            for q in range(OW):
                acc = p.b[f]
                for u in range(3):
                    for v in range(3):
                        acc += p.K[f * 9 + u * 3 + v] * s.x[(r + u) * W + q + v]
                eS = max(eS, abs(acc - w.S[sidx(f, r, q)]))
        for r in range(PH):
            for q in range(PW):
                m = max((max(0.0, w.S[sidx(f, 2 * r + a, 2 * q + b)]) for a in range(2) for b in range(2)))
                P[pidx(f, r, q)] = m
                eP = max(eP, abs(m - w.P[pidx(f, r, q)]))
    for k in range(CLASSES):
        z[k] = p.c[k] + sum((p.W[k * N + j] * P[j] for j in range(N)))
    ez = max((abs(a - b) for a, b in zip(z, w.z)))
    es = abs(sum(w.p) - 1.0)
    e2 = max(eS, eP, ez, es)
    ok &= e2 <= 1e-12
    print(
        f'self-test 2 forward vs naive      max_error={e2:.3e} '
        f'(S {eS:.1e}, P {eP:.1e}, z {ez:.1e}, sum(p)-1 {es:.1e})'
    )
    # 3. Verify the first-maximum tie rule in max-pooling.
    p.b = [0.0, 0.0]
    forward([0.0] * (H * W), p, w)
    bad = sum((
        1
        for f in range(2)
        for r in range(PH)
        for q in range(PW)
        if w.win[pidx(f, r, q)] != sidx(f, 2 * r, 2 * q)
    ))
    ok &= bad == 0
    print(f'self-test 3 pooling tie rule      wrong_winners={bad}')
    print('SELF-TEST', 'PASS' if ok else 'FAIL')
    if not ok:
        raise SystemExit(2)

def gradcheck():
    rng = RNG(42)
    p = Params(2, rng)
    data = synthetic(1, 7)
    s = data[0]
    w = Workspace(2)
    g = Grad(2)
    forward(s.x, p, w)
    g.zero()
    backward_accumulate(s.x, s.y, p, w, g)
    checks = [('K', 0), ('K', 7), ('b', 0), ('b', 1), ('W', 0), ('W', 169), ('W', 2000), ('c', 3)]
    eps = 1e-06
    worst = 0.0
    for name, i in checks:
        arr = getattr(p, name)
        ga = getattr(g, name)[i]
        old = arr[i]
        arr[i] = old + eps
        forward(s.x, p, w)
        lp = loss(w, s.y)
        arr[i] = old - eps
        forward(s.x, p, w)
        lm = loss(w, s.y)
        arr[i] = old
        gn = (lp - lm) / (2 * eps)
        e = abs(ga - gn)
        worst = max(worst, e)
        print(f'gradcheck {name}[{i}] analytic={ga:.8e} numeric={gn:.8e} abs_error={e:.3e}')
    print(f"GRADCHECK {('PASS' if worst < 1e-05 else 'FAIL')} max_error={worst:.3e}")
    if worst >= 1e-05:
        raise SystemExit(3)

def parse():
    ap = argparse.ArgumentParser(description='PC40 Lab 03 sequential CNN reference')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--gradcheck', action='store_true')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--data', default='data/mnist')
    ap.add_argument('--mode', choices=['train', 'infer'], default='train')
    ap.add_argument('--train', type=int, default=2000)
    ap.add_argument('--test', type=int, default=500)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--filters', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--output-csv', default='')
    a = ap.parse_args()
    if any((getattr(a, name) <= 0 for name in ('train', 'test', 'epochs', 'batch', 'filters', 'repeats'))):
        ap.error('--train, --test, --epochs, --batch, --filters and --repeats must all be positive')
    if not math.isfinite(a.lr) or a.lr <= 0:
        ap.error('--lr must be finite and positive')
    return a

def main():
    a = parse()
    if a.self_test:
        self_test()
        return
    if a.gradcheck:
        gradcheck()
        return
    rng = RNG(a.seed)
    p = Params(a.filters, rng)
    if a.synthetic:
        tr = synthetic(a.train, a.seed + 1)
        te = synthetic(a.test, a.seed + 2)
    else:
        tr = load_mnist(a.data, True, a.train)
        te = load_mnist(a.data, False, a.test)
    if a.mode == 'train':
        for e in range(a.epochs):
            t0 = time.perf_counter()
            ml = train_epoch(tr, p, a.batch, a.lr)
            dt = time.perf_counter() - t0
            tl, acc = evaluate(te, p)
            print(f'epoch={e + 1} train_loss={ml:.6f} test_loss={tl:.6f} accuracy={100 * acc:.2f}% time_s={dt:.6f}')
    else:
        w = Workspace(p.F)
        for s in te[:min(10, len(te))]:
            forward(s.x, p, w)
        t0 = time.perf_counter()
        checksum = 0.0
        for _ in range(a.repeats):
            for s in te:
                forward(s.x, p, w)
                checksum += w.p[0]
        dt = time.perf_counter() - t0
        print(
            f'infer images={len(te)} repeats={a.repeats} total_s={dt:.6f} '
            f'images_per_s={len(te) * a.repeats / dt:.3f} checksum={checksum:.9f}'
        )
        if a.output_csv:
            with open(a.output_csv, 'w', newline='') as f:
                cw = csv.writer(f)
                cw.writerow(['language', 'images', 'repeats', 'seconds', 'images_per_s', 'checksum'])
                cw.writerow(['Python', len(te), a.repeats, dt, len(te) * a.repeats / dt, checksum])
if __name__ == '__main__':
    main()
