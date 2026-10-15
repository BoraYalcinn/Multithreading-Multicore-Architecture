// PC40 Lab 03 J.Gaber (gaber@utbm.fr)
// sequential CNN reference (v6)
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class CNNSequential {
    static final int H = 28, W = 28, OH = 26, OW = 26, PH = 13, PW = 13, N_CLASSES = 10;
    static class RNG {
        long s;
        RNG(long x) {
            s = x & 0xffffffffL;
        }
        long u32() {
            s = (1664525L * s + 1013904223L) & 0xffffffffL;
            return s;
        }
        double uniform(double a, double b) {
            return a + (b - a) * (u32() / 4294967296.0);
        }
    }

    static class Sample {
        double[] x;
        int y;
        Sample(double[] x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    static class Params {
        int F;
        double[] K, b, Wd, c;
        Params(int f, RNG r) {
            F = f;
            K = new double[f * 9];
            b = new double[f];
            Wd = new double[N_CLASSES * f * PH * PW];
            c = new double[N_CLASSES];
            for (int i = 0; i < K.length; i++)
                K[i] = r.uniform(-.08, .08);
            for (int i = 0; i < Wd.length; i++)
                Wd[i] = r.uniform(-.05, .05);
        }
    }

    static class Grad {
        int F;
        double[] K, b, Wd, c;
        Grad(int f) {
            F = f;
            K = new double[f * 9];
            b = new double[f];
            Wd = new double[N_CLASSES * f * PH * PW];
            c = new double[N_CLASSES];
        }
        void zero() {
            Arrays.fill(K, 0);
            Arrays.fill(b, 0);
            Arrays.fill(Wd, 0);
            Arrays.fill(c, 0);
        }
    }

    static class WS {
        int F;
        double[] S, A, P, z, p;
        int[] win;
        WS(int f) {
            F = f;
            S = new double[f * OH * OW];
            A = new double[f * OH * OW];
            P = new double[f * PH * PW];
            z = new double[N_CLASSES];
            p = new double[N_CLASSES];
            win = new int[f * PH * PW];
        }
    }

    static int si(int f, int r, int q) {
        return (f * OH + r) * OW + q;
    }

    static int pi(int f, int r, int q) {
        return (f * PH + r) * PW + q;
    }

    static void forward(double[] x, Params p, WS w) {
        int F = p.F;
        for (int f = 0; f < F; f++) {
            int kb = f * 9;
            for (int r = 0; r < OH; r++)
                for (int q = 0; q < OW; q++) {
                    double a = p.b[f];
                    for (int u = 0; u < 3; u++) {
                        int xb = (r + u) * W + q, ko = kb + u * 3;
                        a += p.K[ko] * x[xb] + p.K[ko + 1] * x[xb + 1] + p.K[ko + 2] * x[xb + 2];
                    }
                    int ii = si(f, r, q);
                    w.S[ii] = a;
                    w.A[ii] = a > 0 ? a : 0;
                }
        }
        for (int f = 0; f < F; f++)
            for (int r = 0; r < PH; r++)
                for (int q = 0; q < PW; q++) {
                    int br = 2 * r, bq = 2 * q, besti = 0;
                    double best = -1e300;
                    for (int dr = 0; dr < 2; dr++)
                        for (int dq = 0; dq < 2; dq++) {
                            int ii = si(f, br + dr, bq + dq);
                            if (w.A[ii] > best) {
                                best = w.A[ii];
                                besti = ii;
                            }
                        }
                    int j = pi(f, r, q);
                    w.P[j] = best;
                    w.win[j] = besti;
                }
        int N = F * PH * PW;
        for (int k = 0; k < N_CLASSES; k++) {
            double a = p.c[k];
            int off = k * N;
            for (int j = 0; j < N; j++)
                a += p.Wd[off + j] * w.P[j];
            w.z[k] = a;
        }
        double m = w.z[0], den = 0;
        for (int k = 1; k < N_CLASSES; k++)
            if (w.z[k] > m)
                m = w.z[k];
        for (int k = 0; k < N_CLASSES; k++) {
            w.p[k] = Math.exp(w.z[k] - m);
            den += w.p[k];
        }
        for (int k = 0; k < N_CLASSES; k++)
            w.p[k] /= den;
    }

    static double loss(WS w, int y) {
        double m = w.z[0], d = 0;
        for (int k = 1; k < N_CLASSES; k++)
            if (w.z[k] > m)
                m = w.z[k];
        for (double v : w.z)
            d += Math.exp(v - m);
        return - (w.z[y] - m) + Math.log(d);
    }

    static void backward_accumulate(double[] x, int y, Params p, WS w, Grad g) {
        int F = p.F, N = F * PH * PW;
        double[] dz = w.p.clone(), dP = new double[N], dA = new double[F * OH * OW];
        dz[y] -= 1;
        for (int k = 0; k < N_CLASSES; k++) {
            int off = k * N;
            double d = dz[k];
            g.c[k] += d;
            for (int j = 0; j < N; j++) {
                g.Wd[off + j] += d * w.P[j];
                dP[j] += p.Wd[off + j] * d;
            }
        }
        for (int j = 0; j < N; j++)
            dA[w.win[j]] += dP[j];
        for (int f = 0; f < F; f++) {
            int kb = f * 9;
            for (int r = 0; r < OH; r++)
                for (int q = 0; q < OW; q++) {
                    int ii = si(f, r, q);
                    double ds = w.S[ii] > 0 ? dA[ii] : 0;
                    g.b[f] += ds;
                    for (int u = 0; u < 3; u++) {
                        int xb = (r + u) * W + q, ko = kb + u * 3;
                        g.K[ko] += ds * x[xb];
                        g.K[ko + 1] += ds * x[xb + 1];
                        g.K[ko + 2] += ds * x[xb + 2];
                    }
                }
        }
    }

    static void sgd_update(Params p, Grad g, double lr, int B) {
        double a = lr / B;
        for (int i = 0; i < p.K.length; i++)
            p.K[i] -= a * g.K[i];
        for (int i = 0; i < p.b.length; i++)
            p.b[i] -= a * g.b[i];
        for (int i = 0; i < p.Wd.length; i++)
            p.Wd[i] -= a * g.Wd[i];
        for (int i = 0; i < p.c.length; i++)
            p.c[i] -= a * g.c[i];
    }

    static List<Sample> synthetic(int n, long seed) {
        RNG r = new RNG(seed);
        int[][] cc = { { 4, 4 }, { 4, 12 }, { 4, 20 }, { 10, 7 }, { 10, 17 }, { 16, 4 }, { 16, 12 }, { 16,
            20 }, { 22, 8 }, { 22, 18 } };
        ArrayList<Sample> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int y = i % 10;
            double[] x = new double[H * W];
            for (int j = 0; j < x.length; j++)
                x[j] = .03 * r.uniform(0, 1);
            int cr = cc[y][0], cq = cc[y][1];
            for (int a = Math.max(0, cr - 2); a < Math.min(H, cr + 3); a++)
                for (int b = Math.max(0, cq - 2); b < Math.min(W, cq + 3); b++)
                    x[a * W + b] = .75 +.25 * r.uniform(0, 1);
            out.add(new Sample(x, y));
        }
        return out;
    }

    static int be32(DataInputStream d) throws IOException {
        return d.readInt();
    }

    static List<Sample> loadMNIST(String dir, boolean tr, int n) throws IOException {
        String im = tr ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte",
            la = tr ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte";
        try (DataInputStream fi = new DataInputStream(
                 new BufferedInputStream(Files.newInputStream(Paths.get(dir, im))));
             DataInputStream fl = new DataInputStream(
                 new BufferedInputStream(Files.newInputStream(Paths.get(dir, la))))) {
            if (be32(fi) != 2051 || be32(fl) != 2049)
                throw new IOException("bad MNIST magic");
            int ni = be32(fi), rows = be32(fi), cols = be32(fi), nl = be32(fl);
            if (rows != 28 || cols != 28 || ni != nl)
                throw new IOException("bad MNIST dimensions");
            n = Math.min(n, ni);
            ArrayList<Sample> out = new ArrayList<>();
            byte[] buf = new byte[784];
            for (int i = 0; i < n; i++) {
                fi.readFully(buf);
                int y = fl.readUnsignedByte();
                double[] x = new double[784];
                for (int j = 0; j < 784; j++)
                    x[j] = (buf[j] & 255) / 255.0;
                out.add(new Sample(x, y));
            }
            return out;
        }
    }

    static double[] evaluate(List<Sample> d, Params p) {
        WS w = new WS(p.F);
        double L = 0;
        int ok = 0;
        for (Sample s : d) {
            forward(s.x, p, w);
            L += loss(w, s.y);
            int best = 0;
            for (int k = 1; k < N_CLASSES; k++)
                if (w.p[k] > w.p[best])
                    best = k;
            if (best == s.y)
                ok++;
        }
        return new double[] { L / d.size(), ok / (double) d.size() };
    }

    static double train_epoch(List<Sample> d, Params p, int batch, double lr) {
        WS w = new WS(p.F);
        Grad g = new Grad(p.F);
        double tot = 0;
        for (int lo = 0; lo < d.size(); lo += batch) {
            int hi = Math.min(d.size(), lo + batch);
            g.zero();
            for (int i = lo; i < hi; i++) {
                Sample s = d.get(i);
                forward(s.x, p, w);
                tot += loss(w, s.y);
                backward_accumulate(s.x, s.y, p, w, g);
            }
            sgd_update(p, g, lr, hi - lo);
        }
        return tot / d.size();
    }
    // Three fixtures: a hand-computed convolution, forward() against a naive direct
    // implementation, and the pooling tie rule (first maximum in row-major order).
    static void selfTest() {
        boolean ok = true;
        {
            double[] x = { 1, 2, 0, 1, 3, 1, 2, 2, 0, 1, 3, 1, 2, 2, 1, 0 }, K = { 1, 0,-1, 2 }, ex = { 0,
                5, 2, 5, 6, 1, 2, 1, 2 };
            double e = 0;
            int z = 0;
            for (int r = 0; r < 3; r++)
                for (int q = 0; q < 3; q++) {
                    double a = 0;
                    for (int u = 0; u < 2; u++)
                        for (int v = 0; v < 2; v++)
                            a += K[u * 2 + v] * x[(r + u) * 4 + q + v];
                    e = Math.max(e, Math.abs(a - ex[z++]));
                }
            ok &= e <= 1e-12;
            System.out.printf(Locale.US, "self-test 1 hand convolution      max_error=%.3e%n", e);
        }
        RNG rg = new RNG(42);
        Params p = new Params(2, rg);
        p.b[0] = 0.01;
        p.b[1] = -0.02;
        for (int k = 0; k < N_CLASSES; k++)
            p.c[k] = 0.001 * k;
        double[] x = synthetic(1, 7).get(0).x;
        WS w = new WS(2);
        forward(x, p, w);
        int N = 2 * PH * PW;
        {
            double eS = 0, eP = 0;
            double[] P = new double[N];
            for (int f = 0; f < 2; f++) {
                for (int r = 0; r < OH; r++)
                    for (int q = 0; q < OW; q++) {
                        double a = p.b[f];
                        for (int u = 0; u < 3; u++)
                            for (int v = 0; v < 3; v++)
                                a += p.K[f * 9 + u * 3 + v] * x[(r + u) * W + q + v];
                        eS = Math.max(eS, Math.abs(a - w.S[si(f, r, q)]));
                    }
                for (int r = 0; r < PH; r++)
                    for (int q = 0; q < PW; q++) {
                        double m = 0;
                        for (int a = 0; a < 2; a++)
                            for (int b = 0; b < 2; b++)
                                m = Math.max(m, Math.max(0.0, w.S[si(f, 2 * r + a, 2 * q + b)]));
                        P[pi(f, r, q)] = m;
                        eP = Math.max(eP, Math.abs(m - w.P[pi(f, r, q)]));
                    }
            }
            double ez = 0, sp = 0;
            for (int k = 0; k < N_CLASSES; k++) {
                double a = p.c[k];
                for (int j = 0; j < N; j++)
                    a += p.Wd[k * N + j] * P[j];
                ez = Math.max(ez, Math.abs(a - w.z[k]));
                sp += w.p[k];
            }
            double es = Math.abs(sp - 1.0);
            double e = Math.max(Math.max(eS, eP), Math.max(ez, es));
            ok &= e <= 1e-12;
            System.out.printf(
                Locale.US,
                "self-test 2 forward vs naive      max_error=%.3e (S %.1e, P %.1e, z %.1e, sum(p)-1 %.1e)%n",
                e, eS, eP, ez, es);
        }
        {
            p.b[0] = 0;
            p.b[1] = 0;
            forward(new double[H * W], p, w);
            int bad = 0;
            for (int f = 0; f < 2; f++)
                for (int r = 0; r < PH; r++)
                    for (int q = 0; q < PW; q++)
                        if (w.win[pi(f, r, q)] != si(f, 2 * r, 2 * q))
                            bad++;
            ok &= bad == 0;
            System.out.printf("self-test 3 pooling tie rule      wrong_winners=%d%n", bad);
        }
        System.out.println("SELF-TEST " + (ok ? "PASS" : "FAIL"));
        if (!ok)
            System.exit(2);
    }

    static void gradcheck() {
        RNG r = new RNG(42);
        Params p = new Params(2, r);
        Sample s = synthetic(1, 7).get(0);
        WS w = new WS(2);
        Grad g = new Grad(2);
        forward(s.x, p, w);
        g.zero();
        backward_accumulate(s.x, s.y, p, w, g);
        String[] ns = { "K", "K", "b", "b", "W", "W", "W", "c" };
        int[] is = { 0, 7, 0, 1, 0, 169, 2000, 3 };
        double eps = 1e-6, worst = 0;
        for (int t = 0; t < ns.length; t++) {
            double[] a, gg;
            switch (ns[t]) {
                case "K" :
                a = p.K;
                gg = g.K;
                break;
                case "b" :
                a = p.b;
                gg = g.b;
                break;
                case "W" :
                a = p.Wd;
                gg = g.Wd;
                break;
                default:
                a = p.c;
                gg = g.c;
            }
            int i = is[t];
            double old = a[i], ga = gg[i];
            a[i] = old + eps;
            forward(s.x, p, w);
            double lp = loss(w, s.y);
            a[i] = old - eps;
            forward(s.x, p, w);
            double lm = loss(w, s.y);
            a[i] = old;
            double gn = (lp - lm) / (2 * eps), e = Math.abs(ga - gn);
            worst = Math.max(worst, e);
            System.out.printf(Locale.US, "gradcheck %s[%d] analytic=%.8e numeric=%.8e abs_error=%.3e%n",
                ns[t], i, ga, gn, e);
        }
        System.out.printf(Locale.US, "GRADCHECK %s max_error=%.3e%n", worst < 1e-5 ? "PASS" : "FAIL", worst);
        if (worst >= 1e-5)
            System.exit(3);
    }

    static class Opt {
        boolean st, gc, syn;
        String data = "data/mnist", mode = "train", csv = "";
        int train = 2000, test = 500, epochs = 3, batch = 32, F = 8, seed = 42, repeats = 10;
        double lr = .05;
    }

    static String optionValue(String[] args, int pos, String option) {
        if (pos >= args.length)
            throw new IllegalArgumentException("missing value for " + option);
        return args[pos];
    }
    public static void main(String[] args) throws Exception {
        Opt o = new Opt();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--help")) {
                System.out.println(
                    "PC40 Lab 03 sequential CNN reference\n"
                    + "--self-test --gradcheck --synthetic --data DIR --mode train|infer "
                    + "--train N --test N --epochs N --batch N --filters N "
                    + "--seed N --lr X --repeats N --output-csv FILE");
                return;
            }
            else if (a.equals("--self-test"))
                o.st = true;
            else if (a.equals("--gradcheck"))
                o.gc = true;
            else if (a.equals("--synthetic"))
                o.syn = true;
            else if (a.equals("--data"))
                o.data = optionValue(args,++i, a);
            else if (a.equals("--mode"))
                o.mode = optionValue(args,++i, a);
            else if (a.equals("--train"))
                o.train = Integer.parseInt(optionValue(args,++i, a));
            else if (a.equals("--test"))
                o.test = Integer.parseInt(optionValue(args,++i, a));
            else if (a.equals("--epochs"))
                o.epochs = Integer.parseInt(optionValue(args,++i, a));
            else if (a.equals("--batch"))
                o.batch = Integer.parseInt(optionValue(args,++i, a));
            else if (a.equals("--filters"))
                o.F = Integer.parseInt(optionValue(args,++i, a));
            else if (a.equals("--seed"))
                o.seed = Integer.parseInt(optionValue(args,++i, a));
            else if (a.equals("--lr"))
                o.lr = Double.parseDouble(optionValue(args,++i, a));
            else if (a.equals("--repeats"))
                o.repeats = Integer.parseInt(optionValue(args,++i, a));
            else if (a.equals("--output-csv"))
                o.csv = optionValue(args,++i, a);
            else
                throw new IllegalArgumentException("unknown option " + a);
        }
        if (!o.mode.equals("train") &&!o.mode.equals("infer"))
            throw new IllegalArgumentException("--mode must be train or infer");
        if (o.train <= 0 || o.test <= 0 || o.epochs <= 0 || o.batch <= 0 || o.F <= 0 || o.repeats <= 0)
            throw new IllegalArgumentException(
                "--train, --test, --epochs, --batch, --filters and --repeats must all be positive");
        if (!Double.isFinite(o.lr) || o.lr <= 0)
            throw new IllegalArgumentException("--lr must be finite and positive");
        if (o.st) {
            selfTest();
            return;
        }
        if (o.gc) {
            gradcheck();
            return;
        }
        RNG r = new RNG(o.seed);
        Params p = new Params(o.F, r);
        List<Sample> tr = o.syn ? synthetic(o.train, o.seed + 1) : loadMNIST(o.data, true, o.train),
            te = o.syn ? synthetic(o.test, o.seed + 2) : loadMNIST(o.data, false, o.test);
        if (o.mode.equals("train")) {
            for (int e = 0; e < o.epochs; e++) {
                long t = System.nanoTime();
                double ml = train_epoch(tr, p, o.batch, o.lr), sec = (System.nanoTime() - t) / 1e9;
                double[] ev = evaluate(te, p);
                System.out.printf(Locale.US,
                    "epoch=%d train_loss=%.6f test_loss=%.6f accuracy=%.2f%% time_s=%.6f%n", e + 1, ml,
                    ev[0], 100 * ev[1], sec);
            }
        }
        else {
            WS w = new WS(p.F);
            for (int i = 0; i < Math.min(10, te.size()); i++)
                forward(te.get(i).x, p, w);
            long t = System.nanoTime();
            double sum = 0;
            for (int rr = 0; rr < o.repeats; rr++)
                for (Sample s : te) {
                    forward(s.x, p, w);
                    sum += w.p[0];
                }
            double sec = (System.nanoTime() - t) / 1e9;
            System.out.printf(
                Locale.US,
                "infer images=%d repeats=%d total_s=%.6f images_per_s=%.3f checksum=%.9f%n",
                te.size(), o.repeats, sec, te.size() * o.repeats / sec, sum);
            if (!o.csv.isEmpty()) {
                try (PrintWriter pw = new PrintWriter(o.csv)) {
                    pw.println("language,images,repeats,seconds,images_per_s,checksum");
                    pw.printf(Locale.US, "Java,%d,%d,%.12f,%.12f,%.12f%n", te.size(), o.repeats, sec,
                        te.size() * o.repeats / sec, sum);
                }
            }
        }
    }
}
