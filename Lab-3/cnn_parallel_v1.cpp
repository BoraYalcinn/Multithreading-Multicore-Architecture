// PC40 Lab 03 - Part B, version v1: IMAGE OWNERSHIP (parallel forward / inference)
// Based on cnn_sequential.cpp (v7). forward(), loss(), backward and training are unchanged.
// Only the inference benchmark is parallel:
//   worker t owns images [floor(t*M/T), floor((t+1)*M/T)) and runs the whole forward
//   pass for them with its OWN one-image workspace. Workers are created and joined
//   once per repetition (no persistent pool).
// Build: g++ -O2 -std=c++17 -Wall -Wextra -pedantic -pthread cnn_parallel_v1.cpp -o cnn_v1
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace std;
constexpr int H = 28, W = 28, OH = 26, OW = 26, PH = 13, PW = 13, N_CLASSES = 10;
struct RNG {
    uint32_t s;
    explicit RNG(uint32_t x) : s(x) {
    }
    uint32_t u32() {
        s = 1664525u * s + 1013904223u;
        return s;
    }
    double uniform(double a = -1, double b = 1) {
        return a + (b - a) * (double(u32()) / 4294967296.0);
    }
};

struct Sample {
    vector<double> x;
    int y;
};

struct Params {
    int F;
    vector<double> K, b, Wd, c;
    Params(int f, RNG & r) : F(f), K(f * 9), b(f, 0), Wd(N_CLASSES * f * PH * PW), c(N_CLASSES, 0) {
        for (auto & v : K)
            v = r.uniform(-.08, .08);
        for (auto & v : Wd)
            v = r.uniform(-.05, .05);
    }
};

struct Grad {
    int F;
    vector<double> K, b, Wd, c;
    Grad(int f) : F(f), K(f * 9), b(f), Wd(N_CLASSES * f * PH * PW), c(N_CLASSES) {
    }
    void zero() {
        fill(K.begin(), K.end(), 0);
        fill(b.begin(), b.end(), 0);
        fill(Wd.begin(), Wd.end(), 0);
        fill(c.begin(), c.end(), 0);
    }
};

struct WS {
    int F;
    vector<double> S, A, P, z, p;
    vector<int> win;
    WS(int f) : F(f), S(f * OH * OW), A(f * OH * OW), P(f * PH * PW), z(N_CLASSES), p(N_CLASSES),
        win(f * PH * PW) {
    }
};
inline int si(int f, int r, int q) {
    return (f * OH + r) * OW + q;
}
inline int pi(int f, int r, int q) {
    return (f * PH + r) * PW + q;
}

void forward(const vector<double> & x, const Params & p, WS & w) {
    int F = p.F;
    for (int f = 0; f < F;++f) {
        int kb = f * 9;
        for (int r = 0; r < OH;++r)
            for (int q = 0; q < OW;++q) {
                double a = p.b[f];
                for (int u = 0; u < 3;++u) {
                    int xb = (r + u) * W + q, ko = kb + u * 3;
                    a += p.K[ko] * x[xb] + p.K[ko + 1] * x[xb + 1] + p.K[ko + 2] * x[xb + 2];
                }
                int ii = si(f, r, q);
                w.S[ii] = a;
                w.A[ii] = a > 0 ? a : 0;
            }
    }
    for (int f = 0; f < F;++f)
        for (int r = 0; r < PH;++r)
            for (int q = 0; q < PW;++q) {
                int br = 2 * r, bq = 2 * q, besti = 0;
                double best = -1e300;
                for (int dr = 0; dr < 2;++dr)
                    for (int dq = 0; dq < 2;++dq) {
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
    for (int k = 0; k < N_CLASSES;++k) {
        double a = p.c[k];
        int off = k * N;
        for (int j = 0; j < N;++j)
            a += p.Wd[off + j] * w.P[j];
        w.z[k] = a;
    }
    double m = * max_element(w.z.begin(), w.z.end()), d = 0;
    for (int k = 0; k < N_CLASSES;++k) {
        w.p[k] = exp(w.z[k] - m);
        d += w.p[k];
    }
    for (double & v : w.p)
        v /= d;
}

double loss(const WS & w, int y) {
    double m = * max_element(w.z.begin(), w.z.end()), d = 0;
    for (double v : w.z)
        d += exp(v - m);
    return -(w.z[y] - m) + log(d);
}

void backward_accumulate(const vector<double> & x, int y, const Params & p, const WS & w, Grad & g) {
    int F = p.F, N = F * PH * PW;
    vector<double> dz = w.p, dP(N, 0), dA(F * OH * OW, 0);
    dz[y] -= 1;
    for (int k = 0; k < N_CLASSES;++k) {
        int off = k * N;
        double d = dz[k];
        g.c[k] += d;
        for (int j = 0; j < N;++j) {
            g.Wd[off + j] += d * w.P[j];
            dP[j] += p.Wd[off + j] * d;
        }
    }
    for (int j = 0; j < N;++j)
        dA[w.win[j]] += dP[j];
    for (int f = 0; f < F;++f) {
        int kb = f * 9;
        for (int r = 0; r < OH;++r)
            for (int q = 0; q < OW;++q) {
                int ii = si(f, r, q);
                double ds = w.S[ii] > 0 ? dA[ii] : 0;
                g.b[f] += ds;
                for (int u = 0; u < 3;++u) {
                    int xb = (r + u) * W + q, ko = kb + u * 3;
                    g.K[ko] += ds * x[xb];
                    g.K[ko + 1] += ds * x[xb + 1];
                    g.K[ko + 2] += ds * x[xb + 2];
                }
            }
    }
}

void sgd_update(Params & p, const Grad & g, double lr, int B) {
    double a = lr / B;
    for (size_t i = 0; i < p.K.size();++i)
        p.K[i] -= a * g.K[i];
    for (size_t i = 0; i < p.b.size();++i)
        p.b[i] -= a * g.b[i];
    for (size_t i = 0; i < p.Wd.size();++i)
        p.Wd[i] -= a * g.Wd[i];
    for (size_t i = 0; i < p.c.size();++i)
        p.c[i] -= a * g.c[i];
}

vector<Sample> synthetic(int n, uint32_t seed) {
    RNG r(seed);
    int cc[10][2] = { { 4, 4 }, { 4, 12 }, { 4, 20 }, { 10, 7 }, { 10, 17 }, { 16, 4 }, { 16, 12 }, { 16,
        20 }, { 22, 8 }, { 22, 18 } };
    vector<Sample> v;
    v.reserve(n);
    for (int i = 0; i < n;++i) {
        Sample s;
        s.y = i % 10;
        s.x.resize(H * W);
        for (double & z : s.x)
            z = .03 * r.uniform(0, 1);
        int cr = cc[s.y][0], cq = cc[s.y][1];
        for (int a = max(0, cr - 2); a < min(H, cr + 3);++a)
            for (int b = max(0, cq - 2); b < min(W, cq + 3);++b)
                s.x[a * W + b] = .75 +.25 * r.uniform(0, 1);
        v.push_back(move(s));
    }
    return v;
}
uint32_t be32(ifstream & f) {
    unsigned char b[4];
    f.read((char *) b, 4);
    return (uint32_t(b[0]) << 24) |(uint32_t(b[1]) << 16) |(uint32_t(b[2]) << 8) | b[3];
}

vector<Sample> load_mnist(const string & dir, bool tr, int n) {
    string ip = dir + "/" + (tr ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte"),
        lp = dir + "/" + (tr ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte");
    ifstream fi(ip, ios::binary), fl(lp, ios::binary);
    if (!fi ||!fl)
        throw runtime_error("cannot open MNIST");
    if (be32(fi) != 2051 || be32(fl) != 2049)
        throw runtime_error("bad MNIST magic");
    int ni = be32(fi), rows = be32(fi), cols = be32(fi), nl = be32(fl);
    if (rows != 28 || cols != 28 || ni != nl)
        throw runtime_error("bad MNIST dimensions");
    n = min(n, ni);
    vector<Sample> v;
    v.reserve(n);
    for (int i = 0; i < n;++i) {
        Sample s;
        s.x.resize(784);
        unsigned char buf[784], lab;
        fi.read((char *) buf, 784);
        fl.read((char *) & lab, 1);
        for (int j = 0; j < 784;++j)
            s.x[j] = buf[j] / 255.0;
        s.y = lab;
        v.push_back(move(s));
    }
    return v;
}

pair<double, double> evaluate(const vector<Sample> & d, const Params & p) {
    WS w(p.F);
    double L = 0;
    int ok = 0;
    for (auto & s : d) {
        forward(s.x, p, w);
        L += loss(w, s.y);
        ok += int(max_element(w.p.begin(), w.p.end()) - w.p.begin()) == s.y;
    }
    return {
        L / d.size(), double(ok) / d.size()
    };
}

double train_epoch(const vector<Sample> & d, Params & p, int batch, double lr) {
    WS w(p.F);
    Grad g(p.F);
    double tot = 0;
    for (int lo = 0; lo <(int) d.size(); lo += batch) {
        int hi = min((int) d.size(), lo + batch);
        g.zero();
        for (int i = lo; i < hi;++i) {
            forward(d[i].x, p, w);
            tot += loss(w, d[i].y);
            backward_accumulate(d[i].x, d[i].y, p, w, g);
        }
        sgd_update(p, g, lr, hi - lo);
    }
    return tot / d.size();
}
// Three fixtures: a hand-computed convolution, forward() against a naive direct
// implementation, and the pooling tie rule (first maximum in row-major order).
void selftest() {
    bool ok = true;
    {
        double x[16] = { 1, 2, 0, 1, 3, 1, 2, 2, 0, 1, 3, 1, 2, 2, 1, 0 }, K[4] = { 1, 0,-1, 2 },
            ex[9] = { 0, 5, 2, 5, 6, 1, 2, 1, 2 };
        double e = 0;
        int z = 0;
        for (int r = 0; r < 3;++r)
            for (int q = 0; q < 3;++q) {
                double a = 0;
                for (int u = 0; u < 2;++u)
                    for (int v = 0; v < 2;++v)
                        a += K[u * 2 + v] * x[(r + u) * 4 + q + v];
                e = max(e, fabs(a - ex[z++]));
            }
        ok &= e <= 1e-12;
        cout << "self-test 1 hand convolution      max_error=" << scientific << setprecision(3) << e << "\n";
    }
    RNG r(42);
    Params p(2, r);
    p.b = { 0.01,-0.02 };
    for (int k = 0; k < N_CLASSES;++k)
        p.c[k] = 0.001 * k;
    auto d = synthetic(1, 7);
    const auto & x = d[0].x;
    WS w(2);
    forward(x, p, w);
    int N = 2 * PH * PW;
    {
        double eS = 0, eP = 0;
        vector<double> P(N), z(N_CLASSES);
        for (int f = 0; f < 2;++f) {
            for (int rr = 0; rr < OH;++rr)
                for (int q = 0; q < OW;++q) {
                    double a = p.b[f];
                    for (int u = 0; u < 3;++u)
                        for (int v = 0; v < 3;++v)
                            a += p.K[f * 9 + u * 3 + v] * x[(rr + u) * W + q + v];
                    eS = max(eS, fabs(a - w.S[si(f, rr, q)]));
                }
            for (int rr = 0; rr < PH;++rr)
                for (int q = 0; q < PW;++q) {
                    double m = 0;
                    for (int a = 0; a < 2;++a)
                        for (int b = 0; b < 2;++b)
                            m = max(m, max(0.0, w.S[si(f, 2 * rr + a, 2 * q + b)]));
                    P[pi(f, rr, q)] = m;
                    eP = max(eP, fabs(m - w.P[pi(f, rr, q)]));
                }
        }
        double ez = 0, sp = 0;
        for (int k = 0; k < N_CLASSES;++k) {
            double a = p.c[k];
            for (int j = 0; j < N;++j)
                a += p.Wd[k * N + j] * P[j];
            ez = max(ez, fabs(a - w.z[k]));
            sp += w.p[k];
        }
        double es = fabs(sp - 1.0);
        double e = max(max(eS, eP), max(ez, es));
        ok &= e <= 1e-12;
        cout << "self-test 2 forward vs naive      max_error=" << e << " (S " << setprecision(1) << eS
            << ", P " << eP << ", z " << ez << ", sum(p)-1 " << es << ")\n" << setprecision(3);
    }
    {
        p.b = { 0.0, 0.0 };
        vector<double> zero(H * W, 0.0);
        forward(zero, p, w);
        int bad = 0;
        for (int f = 0; f < 2;++f)
            for (int rr = 0; rr < PH;++rr)
                for (int q = 0; q < PW;++q)
                    if (w.win[pi(f, rr, q)] != si(f, 2 * rr, 2 * q))
                        ++bad;
        ok &= bad == 0;
        cout << "self-test 3 pooling tie rule      wrong_winners=" << bad << "\n";
    }
    cout << "SELF-TEST " <<(ok ? "PASS" : "FAIL") << "\n";
    if (!ok)
        exit(2);
}

void gradcheck() {
    RNG r(42);
    Params p(2, r);
    auto d = synthetic(1, 7);
    WS w(2);
    Grad g(2);
    forward(d[0].x, p, w);
    g.zero();
    backward_accumulate(d[0].x, d[0].y, p, w, g);
    struct Q {
        string n;
        int i;
    };
    vector<Q> qs = { { "K", 0 }, { "K", 7 }, { "b", 0 }, { "b", 1 }, { "W", 0 }, { "W", 169 }, { "W",
        2000 }, { "c", 3 } };
    double eps = 1e-6, worst = 0;
    for (auto q : qs) {
        vector<double> * a, * gg;
        if (q.n == "K") {
            a = & p.K;
            gg = & g.K;
        }
        else if (q.n == "b") {
            a = & p.b;
            gg = & g.b;
        }
        else if (q.n == "W") {
            a = & p.Wd;
            gg = & g.Wd;
        }
        else {
            a = & p.c;
            gg = & g.c;
        }
        double old = (* a)[q.i], ga = (* gg)[q.i];
        (* a)[q.i] = old + eps;
        forward(d[0].x, p, w);
        double lp = loss(w, d[0].y);
        (* a)[q.i] = old - eps;
        forward(d[0].x, p, w);
        double lm = loss(w, d[0].y);
        (* a)[q.i] = old;
        double gn = (lp - lm) / (2 * eps), e = fabs(ga - gn);
        worst = max(worst, e);
        cout << "gradcheck " << q.n << "[" << q.i << "] analytic=" << scientific << setprecision(8) << ga
            << " numeric=" << gn << " abs_error=" << setprecision(3) << e << "\n";
    }
    cout << "GRADCHECK " <<(worst < 1e-5 ? "PASS" : "FAIL") << " max_error=" << scientific
        << setprecision(3) << worst << "\n";
    if (worst >= 1e-5)
        exit(3);
}

// Parallel forward with IMAGE OWNERSHIP: worker t runs forward() on images
// [floor(t*M/T), floor((t+1)*M/T)) with its own workspace ws[t] and writes the class-0
// probability of image i into prob0[i] (one writer per slot). Returns after all workers joined.
void forward_images_parallel(const vector<Sample> & te, const Params & p, vector<WS> & ws,
                             vector<double> & prob0, int T) {
    const int M = (int) te.size();
    vector<thread> workers;
    workers.reserve(T);
    for (int tid = 0; tid < T;++tid)
        workers.emplace_back([&, tid] {
            const int lo = int((1LL * tid * M) / T);       // Equation (2)
            const int hi = int((1LL * (tid + 1) * M) / T);
            WS & w = ws[tid];                              // private workspace
            for (int i = lo; i < hi;++i) {
                forward(te[i].x, p, w);                    // p, te: shared read-only
                prob0[i] = w.p[0];                         // exactly one writer per slot
            }
        });
    for (auto & th : workers)
        th.join();                                         // all images done and visible
}

struct Opt {
    bool st = false, gc = false, syn = false;
    string data = "data/mnist", mode = "train", csv = "";
    int train = 2000, test = 500, epochs = 3, batch = 32, F = 8, seed = 42, repeats = 10;
    int threads = 1;   // T = number of worker threads for the parallel inference
    double lr = .05;
};

int run(int ac, char * * av) {
    Opt o;
    for (int i = 1; i < ac;++i) {
        string a = av[i];
        auto val = [&]() {
            if (i + 1 >= ac)
                throw runtime_error("missing value");
            return string(av[++i]);
        };
        if (a == "--help") {
            cout << "PC40 Lab 03 sequential CNN reference\n"
                    "--self-test --gradcheck --synthetic --data DIR --mode train|infer "
                    "--train N --test N --epochs N --batch N --filters N "
                    "--seed N --lr X --repeats N --output-csv FILE --threads T\n";
            return 0;
        }
        else if (a == "--self-test")
            o.st = true;
        else if (a == "--gradcheck")
            o.gc = true;
        else if (a == "--synthetic")
            o.syn = true;
        else if (a == "--data")
            o.data = val();
        else if (a == "--mode")
            o.mode = val();
        else if (a == "--train")
            o.train = stoi(val());
        else if (a == "--test")
            o.test = stoi(val());
        else if (a == "--epochs")
            o.epochs = stoi(val());
        else if (a == "--batch")
            o.batch = stoi(val());
        else if (a == "--filters")
            o.F = stoi(val());
        else if (a == "--seed")
            o.seed = stoi(val());
        else if (a == "--lr")
            o.lr = stod(val());
        else if (a == "--repeats")
            o.repeats = stoi(val());
        else if (a == "--output-csv")
            o.csv = val();
        else if (a == "--threads")
            o.threads = stoi(val());
        else
            throw runtime_error("unknown option " + a);
    }
    if (o.mode != "train" && o.mode != "infer")
        throw runtime_error("--mode must be train or infer");
    if (o.train <= 0 || o.test <= 0 || o.epochs <= 0 || o.batch <= 0 || o.F <= 0 || o.repeats <= 0 || o.threads <= 0)
        throw runtime_error("--train, --test, --epochs, --batch, --filters, --repeats and --threads must all be positive");
    if (!isfinite(o.lr) || o.lr <= 0)
        throw runtime_error("--lr must be finite and positive");
    if (o.st) {
        selftest();
        return 0;
    }
    if (o.gc) {
        gradcheck();
        return 0;
    }
    RNG r(o.seed);
    Params p(o.F, r);
    vector<Sample> tr = o.syn ? synthetic(o.train, o.seed + 1) : load_mnist(o.data, true, o.train),
        te = o.syn ? synthetic(o.test, o.seed + 2) : load_mnist(o.data, false, o.test);
    if (o.mode == "train") {
        for (int e = 0; e < o.epochs;++e) {
            auto t = chrono::steady_clock::now();
            double ml = train_epoch(tr, p, o.batch, o.lr);
            double sec = chrono::duration < double>(chrono::steady_clock::now() - t).count();
            auto[tl, acc] = evaluate(te, p);
            cout << fixed << setprecision(6) << "epoch=" << e + 1 << " train_loss=" << ml << " test_loss="
                << tl << " accuracy=" << 100 * acc << "% time_s=" << sec << "\n";
        }
    }
    else {
        const int T = o.threads;
        const int M = (int) te.size();                 // work items = images
        // Everything is allocated BEFORE the workers start: one workspace per worker,
        // and one output slot per image (class-0 probability). No resizing later.
        vector<WS> ws(T, WS(p.F));
        vector<double> prob0(M, 0.0);
        for (int i = 0; i < min(10, M);++i)            // untimed warm-up (as in the reference)
            forward(te[i].x, p, ws[0]);
        auto t = chrono::steady_clock::now();
        double sum = 0;
        for (int rr = 0; rr < o.repeats;++rr) {
            forward_images_parallel(te, p, ws, prob0, T);          // workers created + joined inside
            for (int i = 0; i < M;++i)                             // same summation order as the reference
                sum += prob0[i];
        }
        double sec = chrono::duration < double>(chrono::steady_clock::now() - t).count();
        cout << fixed << setprecision(6) << "infer threads=" << o.threads << " images=" << te.size() << " repeats=" << o.repeats
            << " total_s=" << sec << " images_per_s=" << te.size() * o.repeats / sec << " checksum="
            << setprecision(9) << sum << "\n";
        if (!o.csv.empty()) {
            ofstream f(o.csv);
            f << setprecision(12) << "language,images,repeats,seconds,images_per_s,checksum\nC++,"
                << te.size() << "," << o.repeats << "," << sec << "," << te.size() * o.repeats / sec << ","
                << sum << "\n";
        }
    }
    return 0;
}

int main(int ac, char * * av) {
    try {
        return run(ac, av);
    }
    catch (const exception & e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
}