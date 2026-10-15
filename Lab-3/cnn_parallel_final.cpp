// PC40 Lab 03 - Part E: cnn_parallel_final (copy of the validated cnn_parallel_train + measurement)
// Same numerical code as the validated versions. Adds:
//   --mode infer  --decomp image|rows  --scope full|conv  [--pool]  --runs N
//       image = v1 image ownership, rows = v2 row ownership (3 phases + barriers)
//       full  = complete forward inference (+ ordered checksum, same scope as the reference)
//       conv  = convolution (+ReLU) phase only, labelled separately
//       default: workers created and joined at every repetition (as v1/v2)
//       --pool : one persistent team of T workers (main thread = worker 0), barriers separate
//                the repetitions; its setup (thread creation) is timed and reported separately
//   --mode train  --runs N : each run re-initializes the parameters OUTSIDE the timer, then times
//       the epoch loop and reports the time spent in: parallel region (create -> join),
//       reduction, update.  phi = parallel / total (used in Part F).
//   --output-csv FILE : appends one row per run (columns of benchmark_template.csv)
// Build: g++ -O2 -std=c++17 -Wall -Wextra -pedantic -pthread cnn_parallel_final.cpp -o cnn_final
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
#include <condition_variable>
#include <mutex>
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

// Reusable barrier for C++17 (PDF Appendix A, Listing 12). std::barrier needs C++20.
class Barrier {
    std::mutex m;
    std::condition_variable cv;
    const int n;
    int count;
    unsigned long generation = 0;
public:
    explicit Barrier(int n) : n(n), count(n) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(m);
        const unsigned long g = generation;
        if (--count == 0) {          // last arrival opens the barrier
            ++generation;
            count = n;
            cv.notify_all();
        } else {
            cv.wait(lock, [&] { return g != generation; });
        }
    }
};

// Batch-wide buffers for row ownership (offset = n * per-image size + si / pi).
struct BatchWS {
    int F, B;
    vector<double> S, A, P, z, p;
    vector<int> win;
    BatchWS(int f, int b) : F(f), B(b), S((size_t) b * f * OH * OW), A((size_t) b * f * OH * OW),
        P((size_t) b * f * PH * PW), z((size_t) b * N_CLASSES), p((size_t) b * N_CLASSES),
        win((size_t) b * f * PH * PW) {
    }
};

// Phase 1: one convolution row (n, f, r). Same expression as forward().
void conv_row(const vector<double> & x, const Params & p, BatchWS & w, int n, int f, int r) {
    const size_t base = (size_t) n * p.F * OH * OW;
    const int kb = f * 9;
    for (int q = 0; q < OW;++q) {
        double a = p.b[f];                                   // private accumulator
        for (int u = 0; u < 3;++u) {
            int xb = (r + u) * W + q, ko = kb + u * 3;
            a += p.K[ko] * x[xb] + p.K[ko + 1] * x[xb + 1] + p.K[ko + 2] * x[xb + 2];
        }
        size_t ii = base + si(f, r, q);
        w.S[ii] = a;                                         // exactly one writer
        w.A[ii] = a > 0 ? a : 0;
    }
}

// Phase 2: one pooled row (n, f, r). Same comparison (strict '>') as forward().
// The winner is stored as the one-image si offset, exactly like the reference.
void pool_row(BatchWS & w, int F, int n, int f, int r) {
    const size_t sbase = (size_t) n * F * OH * OW, pbase = (size_t) n * F * PH * PW;
    for (int q = 0; q < PW;++q) {
        int br = 2 * r, bq = 2 * q, besti = 0;
        double best = -1e300;
        for (int dr = 0; dr < 2;++dr)
            for (int dq = 0; dq < 2;++dq) {
                int ii = si(f, br + dr, bq + dq);
                if (w.A[sbase + ii] > best) {
                    best = w.A[sbase + ii];
                    besti = ii;
                }
            }
        size_t j = pbase + pi(f, r, q);
        w.P[j] = best;
        w.win[j] = besti;
    }
}

// Phase 3: dense layer + softmax for image n. Same loops as forward().
void dense_softmax(const Params & p, BatchWS & w, int n) {
    const int N = p.F * PH * PW;
    const double * P = &w.P[(size_t) n * N];
    double * z = &w.z[(size_t) n * N_CLASSES];
    double * pr = &w.p[(size_t) n * N_CLASSES];
    for (int k = 0; k < N_CLASSES;++k) {
        double a = p.c[k];
        int off = k * N;
        for (int j = 0; j < N;++j)
            a += p.Wd[off + j] * P[j];
        z[k] = a;
    }
    double m = * max_element(z, z + N_CLASSES), d = 0;
    for (int k = 0; k < N_CLASSES;++k) {
        pr[k] = exp(z[k] - m);
        d += pr[k];
    }
    for (int k = 0; k < N_CLASSES;++k)
        pr[k] /= d;
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
// One synchronous mini-batch at fixed theta: images [lo, hi) are split among T workers
// (image ownership). Each worker zeroes its own G[t] and local_loss[t], then runs the reference
// forward(), loss() and backward_accumulate() (which ADDS into G[t]). After join, the main thread
// reduces G[0] + G[1] + ... + G[T-1] into g in increasing worker order. theta is NOT modified here.
// Returns the batch loss sum (local losses added in worker order).
double batch_gradient_parallel(const vector<Sample> & d, int lo, int hi, const Params & p,
                               vector<WS> & ws, vector<Grad> & G, vector<double> & local_loss,
                               Grad & g, int T) {
    const int B = hi - lo;
    vector<thread> workers;
    workers.reserve(T);
    for (int tid = 0; tid < T;++tid)
        workers.emplace_back([&, tid] {
            const int a = lo + int((1LL * tid * B) / T);           // Equation (2)
            const int b = lo + int((1LL * (tid + 1) * B) / T);
            G[tid].zero();                                         // gradients zeroed per batch
            local_loss[tid] = 0;
            for (int i = a; i < b;++i) {
                forward(d[i].x, p, ws[tid]);                       // reads theta_tau only
                local_loss[tid] += loss(ws[tid], d[i].y);
                backward_accumulate(d[i].x, d[i].y, p, ws[tid], G[tid]);   // ADDS into G[tid]
            }
            // no worker calls sgd_update()
        });
    for (auto & th : workers)
        th.join();                                                 // all gradients at theta_tau done
    g.zero();                                                      // fixed-order serial reduction
    double L = 0;
    for (int t = 0; t < T;++t) {
        for (size_t i = 0; i < g.K.size();++i)  g.K[i]  += G[t].K[i];
        for (size_t i = 0; i < g.b.size();++i)  g.b[i]  += G[t].b[i];
        for (size_t i = 0; i < g.Wd.size();++i) g.Wd[i] += G[t].Wd[i];
        for (size_t i = 0; i < g.c.size();++i)  g.c[i]  += G[t].c[i];
        L += local_loss[t];
    }
    return L;
}

// Parallel version of train_epoch(). Same batch boundaries, same image order, same update count.
double train_epoch_parallel(const vector<Sample> & d, Params & p, int batch, double lr, int T) {
    // Allocated once per epoch, BEFORE any worker starts (the reference also allocates
    // its WS and Grad inside train_epoch, so the timing scope matches).
    vector<WS> ws(T, WS(p.F));                 // one private workspace per worker
    vector<Grad> G(T, Grad(p.F));              // one private full-network gradient per worker
    vector<double> local_loss(T, 0.0);         // one private loss sum per worker
    Grad g(p.F);                               // reduced gradient (main thread only)
    double tot = 0;
    for (int lo = 0; lo <(int) d.size(); lo += batch) {
        const int hi = min((int) d.size(), lo + batch);
        tot += batch_gradient_parallel(d, lo, hi, p, ws, G, local_loss, g, T);
        sgd_update(p, g, lr, hi - lo);         // exactly ONE update, divides by the actual B once
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

// Parallel forward with ROW OWNERSHIP over the whole batch (3 phases, barriers between them).
//   phase 1: M1 = B*F*26 convolution rows (n,f,r) -> S, A
//   phase 2: M2 = B*F*13 pooled rows      (n,f,r) -> P, win
//   phase 3: M3 = B images                 n      -> z, p
// Every item is assigned with Equation (2). Returns after all workers joined.
void forward_rows_parallel(const vector<Sample> & te, const Params & p, BatchWS & bw, int T) {
    const int B = (int) te.size();
    const long long M1 = 1LL * B * p.F * OH, M2 = 1LL * B * p.F * PH, M3 = B;
    auto chunk = [T](long long M, int tid, long long & lo, long long & hi) {
        lo = (tid * M) / T;                                       // Equation (2)
        hi = ((tid + 1) * M) / T;
    };
    Barrier bar(T);
    vector<thread> workers;
    workers.reserve(T);
    for (int tid = 0; tid < T;++tid)
        workers.emplace_back([&, tid] {
            long long lo, hi;
            chunk(M1, tid, lo, hi);                               // phase 1: conv rows
            for (long long a = lo; a < hi;++a) {
                int n = int(a / (p.F * OH)), f = int((a / OH) % p.F), r = int(a % OH);
                conv_row(te[n].x, p, bw, n, f, r);
            }
            bar.arrive_and_wait();                                // all of S, A complete
            chunk(M2, tid, lo, hi);                               // phase 2: pooled rows
            for (long long a = lo; a < hi;++a) {
                int n = int(a / (p.F * PH)), f = int((a / PH) % p.F), r = int(a % PH);
                pool_row(bw, p.F, n, f, r);
            }
            bar.arrive_and_wait();                                // all of P, win complete
            chunk(M3, tid, lo, hi);                               // phase 3: images
            for (long long n = lo; n < hi;++n)
                dense_softmax(p, bw, int(n));
        });
    for (auto & th : workers)
        th.join();                                                // all probabilities complete
}

// Convolution + ReLU of ONE image only (first loop of forward(), copied unchanged).
// Used for the "convolution alone" timing scope with image ownership.
void conv_only(const vector<double> & x, const Params & p, WS & w) {
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
}

// ------------------------------- Part E: inference benchmark --------------------------------
struct InferResult { double sec, checksum, setup_sec; };

// One measured invocation: `repeats` passes over the image set, same timing scope as the
// reference (the ordered checksum summation is inside the timer).
InferResult infer_bench(const vector<Sample> & te, const Params & p, int T, bool rows, bool conv_scope,
                        bool pool, int repeats, vector<WS> & ws, vector<double> & slot, BatchWS & bw) {
    const int B = (int) te.size();
    const int NS = p.F * OH * OW;
    // the work of worker tid for one repetition; `bar` separates the row phases
    auto body = [&](int tid, Barrier & bar) {
        if (!rows) {                                          // image ownership (v1)
            const int lo = int((1LL * tid * B) / T), hi = int((1LL * (tid + 1) * B) / T);
            for (int i = lo; i < hi;++i) {
                if (conv_scope) { conv_only(te[i].x, p, ws[tid]); slot[i] = ws[tid].S[NS - 1]; }
                else            { forward(te[i].x, p, ws[tid]);   slot[i] = ws[tid].p[0]; }
            }
            return;
        }
        long long lo, hi;                                     // row ownership (v2)
        const long long M1 = 1LL * B * p.F * OH, M2 = 1LL * B * p.F * PH, M3 = B;
        lo = (tid * M1) / T; hi = ((tid + 1) * M1) / T;
        for (long long a = lo; a < hi;++a)
            conv_row(te[a / (p.F * OH)].x, p, bw, int(a / (p.F * OH)), int((a / OH) % p.F), int(a % OH));
        if (conv_scope) return;
        bar.arrive_and_wait();
        lo = (tid * M2) / T; hi = ((tid + 1) * M2) / T;
        for (long long a = lo; a < hi;++a)
            pool_row(bw, p.F, int(a / (p.F * PH)), int((a / PH) % p.F), int(a % PH));
        bar.arrive_and_wait();
        lo = (tid * M3) / T; hi = ((tid + 1) * M3) / T;
        for (long long n = lo; n < hi;++n)
            dense_softmax(p, bw, int(n));
    };
    auto value = [&](int n) -> double {                       // the consumed result of image n
        if (!rows) return slot[n];
        return conv_scope ? bw.S[(size_t) n * NS + NS - 1] : bw.p[(size_t) n * N_CLASSES];
    };
    double sum = 0, setup = 0;
    chrono::steady_clock::time_point t0, t1;
    if (!pool) {                                              // create + join at every repetition
        t0 = chrono::steady_clock::now();
        for (int rr = 0; rr < repeats;++rr) {
            Barrier bar(T);
            vector<thread> workers;
            workers.reserve(T);
            for (int tid = 0; tid < T;++tid)
                workers.emplace_back([&, tid] { body(tid, bar); });
            for (auto & th : workers)
                th.join();
            for (int n = 0; n < B;++n)                        // same order as the reference
                sum += value(n);
        }
        t1 = chrono::steady_clock::now();
    } else {                                                  // persistent team, main = worker 0
        Barrier bar(T);
        auto team = [&](int tid) {
            bar.arrive_and_wait();                            // start together
            for (int rr = 0; rr < repeats;++rr) {
                body(tid, bar);
                bar.arrive_and_wait();                        // all results of this repetition done
                if (tid == 0)
                    for (int n = 0; n < B;++n)
                        sum += value(n);
                bar.arrive_and_wait();                        // nobody overwrites before the sum is done
            }
        };
        auto s0 = chrono::steady_clock::now();
        vector<thread> helpers;
        helpers.reserve(T - 1);
        for (int tid = 1; tid < T;++tid)
            helpers.emplace_back(team, tid);
        setup = chrono::duration < double>(chrono::steady_clock::now() - s0).count();
        t0 = chrono::steady_clock::now();
        team(0);
        t1 = chrono::steady_clock::now();
        for (auto & th : helpers)
            th.join();
    }
    return { chrono::duration < double>(t1 - t0).count(), sum, setup };
}

// ------------------------------- Part E: timed training epoch -------------------------------
struct TrainTimes { double par = 0, red = 0, upd = 0; };

// Same algorithm as train_epoch_parallel (validated in Part D), with phase timers.
double train_epoch_timed(const vector<Sample> & d, Params & p, int batch, double lr, int T, TrainTimes & tt) {
    vector<WS> ws(T, WS(p.F));
    vector<Grad> G(T, Grad(p.F));
    vector<double> local_loss(T, 0.0);
    Grad g(p.F);
    double tot = 0;
    for (int lo = 0; lo <(int) d.size(); lo += batch) {
        const int hi = min((int) d.size(), lo + batch), B = hi - lo;
        auto a0 = chrono::steady_clock::now();
        vector<thread> workers;                               // parallel region: create .. join
        workers.reserve(T);
        for (int tid = 0; tid < T;++tid)
            workers.emplace_back([&, tid] {
                const int a = lo + int((1LL * tid * B) / T), b = lo + int((1LL * (tid + 1) * B) / T);
                G[tid].zero();
                local_loss[tid] = 0;
                for (int i = a; i < b;++i) {
                    forward(d[i].x, p, ws[tid]);
                    local_loss[tid] += loss(ws[tid], d[i].y);
                    backward_accumulate(d[i].x, d[i].y, p, ws[tid], G[tid]);
                }
            });
        for (auto & th : workers)
            th.join();
        auto a1 = chrono::steady_clock::now();
        g.zero();                                             // serial fixed-order reduction
        for (int t = 0; t < T;++t) {
            for (size_t i = 0; i < g.K.size();++i)  g.K[i]  += G[t].K[i];
            for (size_t i = 0; i < g.b.size();++i)  g.b[i]  += G[t].b[i];
            for (size_t i = 0; i < g.Wd.size();++i) g.Wd[i] += G[t].Wd[i];
            for (size_t i = 0; i < g.c.size();++i)  g.c[i]  += G[t].c[i];
            tot += local_loss[t];
        }
        auto a2 = chrono::steady_clock::now();
        sgd_update(p, g, lr, B);                              // one update per batch
        auto a3 = chrono::steady_clock::now();
        tt.par += chrono::duration < double>(a1 - a0).count();
        tt.red += chrono::duration < double>(a2 - a1).count();
        tt.upd += chrono::duration < double>(a3 - a2).count();
    }
    return tot / d.size();
}

struct Opt {
    bool st = false, gc = false, syn = false;
    string data = "data/mnist", mode = "train", csv = "";
    int train = 2000, test = 500, epochs = 3, batch = 32, F = 8, seed = 42, repeats = 10;
    int threads = 1;   // T = number of worker threads
    int runs = 1;      // independent measured runs
    bool pool = false;
    string decomp = "image", scope = "full";
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
                    "--seed N --lr X --repeats N --output-csv FILE --threads T --runs N "
                    "--decomp image|rows --scope full|conv --pool\n";
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
        else if (a == "--runs")
            o.runs = stoi(val());
        else if (a == "--pool")
            o.pool = true;
        else if (a == "--decomp")
            o.decomp = val();
        else if (a == "--scope")
            o.scope = val();
        else
            throw runtime_error("unknown option " + a);
    }
    if (o.mode != "train" && o.mode != "infer")
        throw runtime_error("--mode must be train or infer");
    if (o.decomp != "image" && o.decomp != "rows")
        throw runtime_error("--decomp must be image or rows");
    if (o.scope != "full" && o.scope != "conv")
        throw runtime_error("--scope must be full or conv");
    if (o.runs <= 0)
        throw runtime_error("--runs must be positive");
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
    const string dataset = o.syn ? "synthetic" : "MNIST";
    ofstream csv;
    if (!o.csv.empty()) {
        ifstream probe(o.csv);
        bool fresh = !probe.good() || probe.peek() == ifstream::traits_type::eof();
        csv.open(o.csv, ios::app);
        if (fresh)
            csv << "language,version,decomposition,dataset,mode,images,batch,repeats,workers_T,run_index,"
                   "timing_scope,seconds,images_per_s,checksum_or_final_loss,notes\n";
    }
    const int T = o.threads;
    if (o.mode == "train") {
        const size_t extra = (size_t) T * (sizeof(double) * (2 * p.F * OH * OW + p.F * PH * PW + 2 * N_CLASSES)
                             + sizeof(int) * p.F * PH * PW + sizeof(double) * 13610);   // per-worker WS + Grad
        for (int run = 0; run < o.runs;++run) {
            RNG rr(o.seed);
            Params pr(o.F, rr);                               // re-initialized OUTSIDE the timer
            TrainTimes tt;
            double ml = 0;
            auto t = chrono::steady_clock::now();
            for (int e = 0; e < o.epochs;++e)
                ml = train_epoch_timed(tr, pr, o.batch, o.lr, T, tt);
            double sec = chrono::duration < double>(chrono::steady_clock::now() - t).count();
            cout << fixed << setprecision(6) << "train threads=" << T << " run=" << run + 1 << " epochs=" << o.epochs
                << " time_s=" << sec << " parallel_s=" << tt.par << " reduction_s=" << tt.red << " update_s=" << tt.upd
                << " final_train_loss=" << ml << "\n";
            if (csv.is_open())
                csv << setprecision(9) << "C++,final,image," << dataset << ",train," << tr.size() << "," << o.batch << ","
                    << o.epochs << "," << T << "," << run + 1 << ",epoch_loop," << sec << ","
                    << tr.size() * o.epochs / sec << "," << ml << ",create par=" << tt.par << " red=" << tt.red
                    << " upd=" << tt.upd << " extra_bytes=" << extra << "\n";
        }
        auto[tl, acc] = evaluate(te, p);
        (void) tl; (void) acc;
    }
    else {
        const bool rows = o.decomp == "rows", conv_scope = o.scope == "conv";
        const int B = (int) te.size();
        // everything allocated before any worker starts
        vector<WS> ws(rows ? 0 : T, WS(p.F));
        vector<double> slot(B, 0.0);
        BatchWS bw(p.F, rows ? B : 0);
        size_t extra = rows ? (size_t) B * (sizeof(double) * (2 * p.F * OH * OW + p.F * PH * PW + 2 * N_CLASSES)
                                            + sizeof(int) * p.F * PH * PW)
                            : (size_t) T * (sizeof(double) * (2 * p.F * OH * OW + p.F * PH * PW + 2 * N_CLASSES)
                                            + sizeof(int) * p.F * PH * PW) + sizeof(double) * B;
        {
            WS w(p.F);                                        // untimed warm-up (as in the reference)
            for (int i = 0; i < min(10, B);++i)
                forward(te[i].x, p, w);
            infer_bench(te, p, T, rows, conv_scope, o.pool, 1, ws, slot, bw);   // untimed warm-up of my code
        }
        for (int run = 0; run < o.runs;++run) {
            InferResult r = infer_bench(te, p, T, rows, conv_scope, o.pool, o.repeats, ws, slot, bw);
            cout << fixed << setprecision(6) << "infer decomp=" << o.decomp << " scope=" << o.scope
                << " workers=" << (o.pool ? "pool" : "create") << " threads=" << T << " run=" << run + 1
                << " images=" << B << " repeats=" << o.repeats << " total_s=" << r.sec
                << " images_per_s=" << B * o.repeats / r.sec << " setup_s=" << r.setup_sec
                << " checksum=" << setprecision(9) << r.checksum << "\n";
            if (csv.is_open())
                csv << setprecision(12) << "C++,final," << o.decomp << "," << dataset << ",infer," << B << ","
                    << B << "," << o.repeats << "," << T << "," << run + 1 << "," << o.scope << "," << r.sec << ","
                    << B * o.repeats / r.sec << "," << r.checksum << "," << (o.pool ? "pool" : "create")
                    << " setup_s=" << r.setup_sec << " extra_bytes=" << extra << "\n";
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
