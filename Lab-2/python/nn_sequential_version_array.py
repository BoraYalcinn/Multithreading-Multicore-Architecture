#PC40: Parallel Computing, Fall 2026 (A26), J.Gaber, gaber@utbm.fr

#!/usr/bin/env python3
"""PC40 - Sequential baseline, array.array variant (to be tested on CPython 3.13t / 3.14t).

Identical algorithm and identical numerical results to nn_sequential.py.
The ONLY difference: the model parameters, the input images and the gradient
accumulators are stored in array.array('d') (contiguous C doubles) instead of
Python lists of float objects.

Why this matters for the free-threaded build
--------------------------------------------
In a list of floats, every element is a separate Python object with its own
reference count. When P threads read the same weights W1[j*H+k] in a tight
loop, each read from a thread that does not own the object performs an
*atomic* increment/decrement on that object's shared reference counter
(biased reference counting). Many threads hitting the same 50 890 objects
means cache-line bouncing on the refcounts, which can destroy scaling for
reasons unrelated to the synchronisation lessons of the lab.

array.array('d') stores raw doubles. Indexing creates a fresh float object
owned by the reading thread; no shared reference count is touched. The
reads of the model remain genuinely read-only shared memory.

Run:  python3 nn_sequential_array.py ../data [--limit N]
      python3 nn_sequential_array.py --synthetic
The program prints whether the interpreter is free-threaded.
"""
import math
import random
import struct
import sys
import sysconfig
import time
from array import array

D, H, C = 784, 64, 10


# ---------------------------------------------------------------------------
# [1] Data loading
# ---------------------------------------------------------------------------
def load_idx(ip, lp, limit):
    """Load at most `limit` samples. Each image is an array('d') of 784 doubles."""
    with open(ip, 'rb') as fi, open(lp, 'rb') as fl:
        magic, n, r, c = struct.unpack('>IIII', fi.read(16))
        ml, nl = struct.unpack('>II', fl.read(8))
        assert magic == 2051 and ml == 2049
        n = min(n, nl, limit)
        xs, ys = [], []
        for _ in range(n):
            xs.append(array('d', (b / 255.0 for b in fi.read(r * c))))
            ys.append(fl.read(1)[0])
    return xs, ys


def synthetic(n):
    """Synthetic dataset for smoke tests: class y lights up 20 fixed pixels."""
    random.seed(1)
    xs, ys = [], []
    for i in range(n):
        y = i % 10
        x = array('d', (random.gauss(0, .03) for _ in range(D)))
        for k in range(20):
            x[(y * 73 + k) % D] += 1
        xs.append(x)
        ys.append(y)
    return xs, ys


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class NN:
    # Parameters. Layout: W1[j*H+k] connects input j to hidden k,
    #                     W2[k*C+c] connects hidden k to output c.

    def __init__(self):
        # [2] Model initialisation: deterministic pseudo-random weights (seed 42).
        random.seed(42)
        self.W1 = array('d', (random.gauss(0, .05) for _ in range(D * H)))
        self.b1 = array('d', [0.] * H)
        self.W2 = array('d', (random.gauss(0, .05) for _ in range(H * C)))
        self.b2 = array('d', [0.] * C)

    @staticmethod
    def new_grads():
        """Fresh zeroed gradient accumulators (g1, gb1, g2, gb2).

        In the multithreaded version each worker should own ONE such set.
        """
        return (array('d', bytes(8 * D * H)), array('d', bytes(8 * H)),
                array('d', bytes(8 * H * C)), array('d', bytes(8 * C)))

    def sample_grad(self, x, label, g1, gb1, g2, gb2):
        """Forward + backward pass for ONE sample. Never modifies the model."""
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2

        # ---- [3] Forward pass ---------------------------------------------
        h = []
        for k in range(H):
            s = b1[k]
            for j in range(D):
                s += x[j] * W1[j * H + k]
            h.append(max(0., s))

        logits = []
        for c in range(C):
            s = b2[c]
            for k in range(H):
                s += h[k] * W2[k * C + c]
            logits.append(s)

        m = max(logits)
        p = [math.exp(v - m) for v in logits]
        z = sum(p)
        p = [v / z for v in p]

        loss = -math.log(max(p[label], 1e-8))
        pred = max(range(C), key=p.__getitem__)

        # ---- [4] Backward pass --------------------------------------------
        dl = p[:]
        dl[label] -= 1

        for k in range(H):
            for c in range(C):
                g2[k * C + c] += h[k] * dl[c]
        for c in range(C):
            gb2[c] += dl[c]

        dh = [sum(W2[k * C + c] * dl[c] for c in range(C)) if h[k] > 0 else 0.
              for k in range(H)]

        for j in range(D):
            xj = x[j]
            for k in range(H):
                g1[j * H + k] += xj * dh[k]
        for k in range(H):
            gb1[k] += dh[k]

        return loss, pred == label

    def train(self, xs, ys, epochs=3, bs=64, lr=.08):
        """Training loop: epochs x mini-batches."""
        for ep in range(epochs):
            loss = 0.
            ok = 0

            for s in range(0, len(xs), bs):
                e = min(s + bs, len(xs))
                n = e - s
                g1, gb1, g2, gb2 = self.new_grads()

                # ---- [5] Mini-batch accumulation --------------------------
                # THIS is the loop to parallelise.
                for i in range(s, e):
                    l, c = self.sample_grad(xs[i], ys[i], g1, gb1, g2, gb2)
                    loss += l
                    ok += c

                # ---- [6] Weight update ------------------------------------
                a = lr / n
                W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
                for i in range(D * H):
                    W1[i] -= a * g1[i]
                for i in range(H):
                    b1[i] -= a * gb1[i]
                for i in range(H * C):
                    W2[i] -= a * g2[i]
                for i in range(C):
                    b2[i] -= a * gb2[i]

            print(f"Epoch {ep + 1} loss={loss / len(xs):.4f} train_acc={100 * ok / len(xs):.2f}%")

    # ---- [7] Evaluation ---------------------------------------------------
    def predict(self, x):
        h = [max(0., self.b1[k] + sum(x[j] * self.W1[j * H + k] for j in range(D)))
             for k in range(H)]
        o = [self.b2[c] + sum(h[k] * self.W2[k * C + c] for k in range(H))
             for c in range(C)]
        return max(range(C), key=o.__getitem__)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def interpreter_info():
    ft = sysconfig.get_config_var('Py_GIL_DISABLED')
    gil = getattr(sys, '_is_gil_enabled', lambda: True)()
    return (f"Python {sys.version.split()[0]} | free-threaded build: {bool(ft)}"
            f" | GIL currently enabled: {gil}")


def main():
    argv = sys.argv[1:]
    syn = '--synthetic' in argv
    limit = 3000
    if '--limit' in argv:
        limit = int(argv[argv.index('--limit') + 1])

    print(interpreter_info())
    if syn:
        tr = synthetic(500)
        te = synthetic(200)
    else:
        root = next((a for a in argv if not a.startswith('--') and not a.isdigit()), '../data')
        try:
            tr = load_idx(root + '/train-images-idx3-ubyte', root + '/train-labels-idx1-ubyte', limit)
            te = load_idx(root + '/t10k-images-idx3-ubyte', root + '/t10k-labels-idx1-ubyte', 500)
        except FileNotFoundError:
            print('Run data/download_mnist.py or use --synthetic')
            return 1

    nn = NN()
    t = time.perf_counter()
    nn.train(*tr)
    sec = time.perf_counter() - t

    acc = 100 * sum(nn.predict(x) == y for x, y in zip(*te)) / len(te[0])
    print(f"Test accuracy: {acc:.2f}%\nTraining time: {sec:.3f} s")


if __name__ == '__main__':
    raise SystemExit(main() or 0)
