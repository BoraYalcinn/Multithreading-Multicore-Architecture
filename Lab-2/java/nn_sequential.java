// PC40: Parallel Computing, Fall 2026 (A26), J.Gaber, gaber@utbm.fr
// Sequential baseline, keep this file unchanged.
//
// Small fully-connected neural network for MNIST:
//     784 inputs -> 64 hidden neurons (ReLU) -> 10 outputs (softmax)
// trained by mini-batch stochastic gradient descent (SGD). No external libraries.
//
// Regions students must identify in Part A:
//   [1] Data loading          load(), synthetic()
//   [2] Model initialisation  NN()
//   [3] Forward pass          first half of NN.sampleGrad()
//   [4] Backward pass         second half of NN.sampleGrad()
//   [5] Mini-batch loop       inner loop of NN.train()   <-- parallel target
//   [6] Weight update         end of the batch in NN.train()
//   [7] Evaluation            NN.predict(), NN.accuracy()
//
// Build:  javac nn_sequential.java
// Run:    java nn_sequential ../data [--limit N]  (MNIST IDX files)
//         java nn_sequential --synthetic [--limit N] (smoke test)
//
// IMPORTANT FOR PERFORMANCE MEASUREMENTS:
//   The JVM uses Just-In-Time (JIT) compilation. The first execution can include
//   class loading and compilation overhead. This program therefore performs a
//   short, untimed warm-up before creating the measured network. Use the same
//   JVM, dataset size, number of epochs, and command-line options when comparing
//   the sequential and multithreaded versions.

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Random;

public class nn_sequential {

    static final int D = 784, H = 64, C = 10;

    // -----------------------------------------------------------------------
    // [1] Data loading
    // -----------------------------------------------------------------------
    static class Data {
        float[][] x;   // x[i] = 784 pixels of sample i, normalised to [0,1]
        int[] y;       // labels in 0..9

        Data(float[][] x, int[] y) {
            this.x = x;
            this.y = y;
        }
    }

    // Read a 32-bit big-endian integer (IDX header format).
    static int be32(DataInputStream in) throws IOException {
        return in.readInt();
    }

    // Load at most `limit` samples from an IDX image file and its label file.
    static Data load(String ip, String lp, int limit) throws IOException {
        try (DataInputStream fi = new DataInputStream(new BufferedInputStream(new FileInputStream(ip)));
             DataInputStream fl = new DataInputStream(new BufferedInputStream(new FileInputStream(lp)))) {
            if (be32(fi) != 2051 || be32(fl) != 2049) throw new IOException("Bad IDX magic");
            int ni = be32(fi), nl = be32(fl), r = be32(fi), c = be32(fi);
            int n = Math.min(Math.min(ni, nl), limit);
            float[][] x = new float[n][r * c];
            int[] y = new int[n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < r * c; j++) x[i][j] = fi.readUnsignedByte() / 255f;
                y[i] = fl.readUnsignedByte();
            }
            return new Data(x, y);
        }
    }

    // Synthetic dataset for smoke tests: class y lights up 20 fixed pixels.
    static Data synthetic(int n) {
        Random g = new Random(1);
        float[][] x = new float[n][D];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            y[i] = i % 10;
            for (int j = 0; j < D; j++) x[i][j] = (float) (g.nextGaussian() * .03);
            for (int k = 0; k < 20; k++) x[i][(y[i] * 73 + k) % D] += 1;
        }
        return new Data(x, y);
    }

    // -----------------------------------------------------------------------
    // The model
    // -----------------------------------------------------------------------
    static class NN {
        // Parameters. Layout: W1[j*H+k] connects input j to hidden k,
        //                     W2[k*C+c] connects hidden k to output c.
        float[] W1 = new float[D * H], b1 = new float[H];
        float[] W2 = new float[H * C], b2 = new float[C];

        // [2] Model initialisation: deterministic pseudo-random weights (seed 42).
        NN() {
            Random g = new Random(42);
            for (int i = 0; i < W1.length; i++) W1[i] = (float) (g.nextGaussian() * .05);
            for (int i = 0; i < W2.length; i++) W2[i] = (float) (g.nextGaussian() * .05);
        }

        // Forward + backward pass for ONE sample (x, label).
        // Reads:   W1, b1, W2, b2 (the model)             -> shared, read-only
        // Adds to: g1, gb1, g2, gb2 (gradient accumulators),
        //          ok[0] (correct count)                   -> whoever owns them
        // Returns: the sample loss.
        // The method never modifies the model.
        float sampleGrad(float[] x, int label,
                         float[] g1, float[] gb1, float[] g2, float[] gb2,
                         int[] ok) {
            // ---- [3] Forward pass -----------------------------------------
            float[] h = new float[H], p = new float[C];

            // Hidden layer: h = ReLU(W1^T x + b1)
            for (int k = 0; k < H; k++) {
                float s = b1[k];
                for (int j = 0; j < D; j++) s += x[j] * W1[j * H + k];
                h[k] = Math.max(0, s);
            }

            // Output layer: logits stored in p, track max for stability
            float m = -Float.MAX_VALUE;
            for (int c = 0; c < C; c++) {
                float s = b2[c];
                for (int k = 0; k < H; k++) s += h[k] * W2[k * C + c];
                p[c] = s;
                m = Math.max(m, s);
            }

            // Softmax (shifted by the max) and prediction
            float z = 0;
            for (int c = 0; c < C; c++) {
                p[c] = (float) Math.exp(p[c] - m);
                z += p[c];
            }
            int pred = 0;
            for (int c = 0; c < C; c++) {
                p[c] /= z;
                if (p[c] > p[pred]) pred = c;
            }
            if (pred == label) ok[0]++;
            float loss = -(float) Math.log(Math.max(p[label], 1e-8));

            // ---- [4] Backward pass ----------------------------------------
            // dL/dlogit = prob - onehot(label), computed in place in p
            p[label] -= 1;

            // Output layer gradients
            for (int k = 0; k < H; k++)
                for (int c = 0; c < C; c++) g2[k * C + c] += h[k] * p[c];
            for (int c = 0; c < C; c++) gb2[c] += p[c];

            // Back-propagate to the hidden layer through ReLU
            float[] dh = new float[H];
            for (int k = 0; k < H; k++) {
                for (int c = 0; c < C; c++) dh[k] += W2[k * C + c] * p[c];
                if (h[k] <= 0) dh[k] = 0;
            }

            // Hidden layer gradients (the most expensive loop: 784 x 64)
            for (int j = 0; j < D; j++)
                for (int k = 0; k < H; k++) g1[j * H + k] += x[j] * dh[k];
            for (int k = 0; k < H; k++) gb1[k] += dh[k];

            return loss;
        }

        // Training loop: epochs x mini-batches.
        void train(Data d, int epochs, int bs, float lr) {
            for (int ep = 0; ep < epochs; ep++) {
                float loss = 0;      // epoch loss
                int[] ok = {0};      // epoch correct count

                for (int s = 0; s < d.x.length; s += bs) {
                    int e = Math.min(s + bs, d.x.length);   // last batch may be shorter
                    int n = e - s;

                    // Batch gradient accumulators, zeroed for every batch.
                    float[] g1 = new float[W1.length], gb1 = new float[H];
                    float[] g2 = new float[W2.length], gb2 = new float[C];

                    // ---- [5] Mini-batch accumulation ----------------------
                    // Each sample contributes independently to the same
                    // accumulators. THIS is the loop to parallelise.
                    for (int i = s; i < e; i++)
                        loss += sampleGrad(d.x[i], d.y[i], g1, gb1, g2, gb2, ok);

                    // ---- [6] Weight update --------------------------------
                    // Once per batch, AFTER the full batch gradient is known:
                    // W <- W - lr * G / n
                    float a = lr / n;
                    for (int i = 0; i < W1.length; i++) W1[i] -= a * g1[i];
                    for (int i = 0; i < H; i++) b1[i] -= a * gb1[i];
                    for (int i = 0; i < W2.length; i++) W2[i] -= a * g2[i];
                    for (int i = 0; i < C; i++) b2[i] -= a * gb2[i];
                }

                System.out.printf("Epoch %d loss=%.4f train_acc=%.2f%%%n",
                                  ep + 1, loss / d.x.length, 100.0 * ok[0] / d.x.length);
            }
        }

        // ---- [7] Evaluation -------------------------------------------------
        // Forward pass only (argmax of logits == argmax of probabilities).
        int predict(float[] x) {
            float[] h = new float[H], o = new float[C];
            for (int k = 0; k < H; k++) {
                float s = b1[k];
                for (int j = 0; j < D; j++) s += x[j] * W1[j * H + k];
                h[k] = Math.max(0, s);
            }
            for (int c = 0; c < C; c++) {
                float s = b2[c];
                for (int k = 0; k < H; k++) s += h[k] * W2[k * C + c];
                o[c] = s;
            }
            int p = 0;
            for (int c = 1; c < C; c++) if (o[c] > o[p]) p = c;
            return p;
        }

        double accuracy(Data d) {
            int ok = 0;
            for (int i = 0; i < d.x.length; i++) if (predict(d.x[i]) == d.y[i]) ok++;
            return 100.0 * ok / d.x.length;
        }
    }

    // -----------------------------------------------------------------------
    // Command-line helpers
    // -----------------------------------------------------------------------
    static int intOption(String[] args, String option, int defaultValue) {
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals(option)) {
                if (i + 1 >= args.length)
                    throw new IllegalArgumentException("Missing value after " + option);
                return Integer.parseInt(args[i + 1]);
            }
        }
        return defaultValue;
    }

    static String dataRoot(String[] args) {
        for (int i = 0; i < args.length; i++) {
            String v = args[i];
            if (v.equals("--limit")) { i++; continue; }
            if (!v.startsWith("--")) return v;
        }
        return "../data";
    }

    // Short untimed run used only to let the JVM/JIT execute the hot methods
    // before the real timing starts. A fresh NN is created afterwards, so the
    // measured training always starts from the deterministic seed-42 weights.
    static void warmUp() {
        Data w = synthetic(64);
        NN n = new NN();
        n.train(w, 1, 64, .08f);
    }

    // -----------------------------------------------------------------------
    // main
    // -----------------------------------------------------------------------
    public static void main(String[] a) throws Exception {
        boolean syn = false;
        for (String v : a) if (v.equals("--synthetic")) syn = true;

        // --limit controls the training-set size in BOTH real and synthetic mode.
        // This makes C++/Java/Python experiments easier to reproduce and compare.
        int defaultTrain = syn ? 2000 : 12000;
        int trainLimit = intOption(a, "--limit", defaultTrain);
        int testLimit = syn ? Math.min(500, Math.max(100, trainLimit / 4)) : 2000;
        String root = dataRoot(a);

        Data tr, te;
        try {
            tr = syn ? synthetic(trainLimit)
                     : load(root + "/train-images-idx3-ubyte",
                            root + "/train-labels-idx1-ubyte", trainLimit);
            te = syn ? synthetic(testLimit)
                     : load(root + "/t10k-images-idx3-ubyte",
                            root + "/t10k-labels-idx1-ubyte", testLimit);
        } catch (IOException e) {
            System.err.println(e + "\nRun data/download_mnist.py or use --synthetic");
            return;
        }

        System.out.printf("Dataset: %s | train=%d | test=%d | architecture=%d->%d->%d%n",
                          syn ? "synthetic" : "MNIST", tr.x.length, te.x.length, D, H, C);

        // JVM warm-up is deliberately OUTSIDE the measured interval.
        // It reduces the risk that JIT compilation dominates the first timing.
        System.out.println("JVM warm-up (not timed)...");
        warmUp();

        // Fresh model: warm-up must not change the initial state of the measured run.
        NN nn = new NN();
        System.out.println("Measured sequential training...");
        long t = System.nanoTime();
        nn.train(tr, /*epochs=*/3, /*batch size=*/64, /*learning rate=*/.08f);
        double sec = (System.nanoTime() - t) / 1e9;

        System.out.printf("Test accuracy: %.2f%%%nTraining time: %.3f s%n", nn.accuracy(te), sec);
        System.out.println("Use this time as T1 only when TP is measured with the same JVM and options.");
    }}
