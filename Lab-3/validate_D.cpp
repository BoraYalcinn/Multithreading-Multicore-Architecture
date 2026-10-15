// PC40 Lab 03 - Part D: correctness validation.
// Compiles the UNCHANGED sequential reference and my three parallel sources into one program
// (each in its own namespace) and compares them on small deterministic cases:
//   forward  (v1 image ownership, v2 row ownership): S, A, P, winners, z, p -> EXACT equality
//   training (cnn_parallel_train): G_K, G_b, G_W, G_c and loss at the same theta, before the
//            division -> |a-b| <= eps_abs + eps_rel*max(|a|,|b|) (Eq. 8), NaN/Inf fail;
//            then one sgd_update -> max parameter difference; then whole epochs -> loss sequence.
// Every comparison is also written to validation.csv (columns of validation_template.csv).
// Build: g++ -O2 -std=c++17 -Wall -Wextra -pthread validate_D.cpp -o validate_D
// Run:   ./validate_D                 (synthetic data)
//        ./validate_D data/mnist      (also uses the first MNIST images)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// The standard headers above are already included, so the #includes inside the files
// below do nothing; each file's own code lands in its own namespace.
namespace seq {
#define main seq_main
#include "cnn_sequential.cpp"
#undef main
}
namespace v1 {
#define main v1_main
#include "cnn_parallel_v1.cpp"
#undef main
}
namespace v2 {
#define main v2_main
#include "cnn_parallel_v2.cpp"
#undef main
}
namespace tr {
#define main tr_main
#include "cnn_parallel_train.cpp"
#undef main
}

using std::vector;
using std::string;

static const double EPS_ABS = 1e-10, EPS_REL = 1e-8;   // starting values from the lab (Eq. 8)
static std::ofstream csv;
static int n_fail = 0;

// Convert the samples of one namespace to another (same layout: vector<double> x, int y).
template <class To, class From> vector<To> conv_samples(const vector<From> & v) {
    vector<To> out;
    for (auto & s : v) out.push_back(To{ s.x, s.y });
    return out;
}

struct Cmp {
    double max_abs = 0, max_rel = 0;
    long first_fail = -1, non_finite = 0, n = 0;
    template <class T> void exact(const T & a, const T & b, long idx) {
        ++n;
        if constexpr (std::is_floating_point_v<T>) {
            if (!std::isfinite(a) || !std::isfinite(b)) { ++non_finite; if (first_fail < 0) first_fail = idx; return; }
            double d = std::fabs(a - b);
            max_abs = std::max(max_abs, d);
            double m = std::max(std::fabs(a), std::fabs(b));
            if (m > 0) max_rel = std::max(max_rel, d / m);
        }
        if (!(a == b) && first_fail < 0) first_fail = idx;
    }
    void tol(double a, double b, long idx) {
        ++n;
        if (!std::isfinite(a) || !std::isfinite(b)) { ++non_finite; if (first_fail < 0) first_fail = idx; return; }
        double d = std::fabs(a - b), m = std::max(std::fabs(a), std::fabs(b));
        max_abs = std::max(max_abs, d);
        if (m > 0) max_rel = std::max(max_rel, d / m);
        if (d > EPS_ABS + EPS_REL * m && first_fail < 0) first_fail = idx;
    }
    bool ok() const { return first_fail < 0 && non_finite == 0; }
};

static void record(const string & version, const string & decomp, const string & qty, int images, int batch,
                   int T, const string & partition, const string & rule, const Cmp & c, const string & notes = "") {
    bool ok = c.ok();
    if (!ok) ++n_fail;
    std::printf("  %-6s %-22s B=%-3d T=%-3d %-30s max_abs=%-10.3e max_rel=%-10.3e first_fail=%-6ld %s\n",
                version.c_str(), qty.c_str(), images, T, partition.c_str(), c.max_abs, c.max_rel, c.first_fail,
                ok ? "PASS" : "FAIL");
    csv << version << "," << decomp << "," << qty << "," << images << "," << batch << "," << T << ",\"" << partition
        << "\"," << rule << "," << std::scientific << std::setprecision(3) << c.max_abs << "," << c.max_rel << ","
        << c.first_fail << "," << c.non_finite << "," << (ok ? "PASS" : "FAIL") << "," << notes << "\n";
}

static string partition_of(int M, int T) {          // e.g. "3+3+4" (sizes of the T chunks)
    string s;
    for (int t = 0; t < T; ++t) {
        int lo = int((1LL * t * M) / T), hi = int((1LL * (t + 1) * M) / T);
        if (t) s += "+";
        s += std::to_string(hi - lo);
    }
    return s.size() > 28 ? s.substr(0, 25) + "..." : s;
}

// ---------------------------------------------------------------------------------------------
// FORWARD: v1 (image ownership) and v2 (row ownership) vs the sequential forward(), exact.
static void check_forward(const vector<seq::Sample> & data, const seq::Params & P, int T, const string & tag) {
    const int B = (int) data.size(), F = P.F;
    const int NS = F * seq::OH * seq::OW, NP = F * seq::PH * seq::PW, NC = seq::N_CLASSES;
    // reference values, one image at a time
    vector<double> S, A, Pp, z, pr; vector<int> win;
    seq::WS w(F);
    for (auto & s : data) {
        seq::forward(s.x, P, w);
        S.insert(S.end(), w.S.begin(), w.S.end());   A.insert(A.end(), w.A.begin(), w.A.end());
        Pp.insert(Pp.end(), w.P.begin(), w.P.end()); win.insert(win.end(), w.win.begin(), w.win.end());
        z.insert(z.end(), w.z.begin(), w.z.end());   pr.insert(pr.end(), w.p.begin(), w.p.end());
    }
    const string part = partition_of(B, T);
    // v1: same parameters, forward_images_parallel -> prob0 (exact). The other arrays live in the
    // per-worker workspaces; v1 calls the unchanged forward(), so prob0 is what leaves the workers.
    {
        v1::RNG dummy(1); v1::Params p1(F, dummy);
        p1.K = P.K; p1.b = P.b; p1.Wd = P.Wd; p1.c = P.c;
        auto d1 = conv_samples<v1::Sample>(data);
        vector<v1::WS> ws(T, v1::WS(F));
        vector<double> prob0(B, -1.0);                // sentinel: an unwritten slot would fail
        v1::forward_images_parallel(d1, p1, ws, prob0, T);
        Cmp c; for (int n = 0; n < B; ++n) c.exact(prob0[n], pr[(size_t) n * NC], n);
        record("v1", "image", "p[class0] " + tag, B, B, T, part, "exact", c);
    }
    // v2: every intermediate array is batch-wide -> compare all of them exactly.
    {
        v2::RNG dummy(1); v2::Params p2(F, dummy);
        p2.K = P.K; p2.b = P.b; p2.Wd = P.Wd; p2.c = P.c;
        auto d2 = conv_samples<v2::Sample>(data);
        v2::BatchWS bw(F, B);
        std::fill(bw.S.begin(), bw.S.end(), -777.0);  // sentinels: any unwritten output fails
        std::fill(bw.P.begin(), bw.P.end(), -777.0);
        std::fill(bw.win.begin(), bw.win.end(), -1);
        v2::forward_rows_parallel(d2, p2, bw, T);
        const string rp = partition_of(B * F * seq::OH, T);
        Cmp cS, cA, cP, cW, cz, cp;
        for (size_t i = 0; i < S.size(); ++i) { cS.exact(bw.S[i], S[i], i); cA.exact(bw.A[i], A[i], i); }
        for (size_t i = 0; i < Pp.size(); ++i) { cP.exact(bw.P[i], Pp[i], i); cW.exact(bw.win[i], win[i], i); }
        for (size_t i = 0; i < z.size(); ++i) { cz.exact(bw.z[i], z[i], i); cp.exact(bw.p[i], pr[i], i); }
        record("v2", "rows", "S " + tag, B, B, T, rp, "exact", cS);
        record("v2", "rows", "A " + tag, B, B, T, rp, "exact", cA);
        record("v2", "rows", "P " + tag, B, B, T, rp, "exact", cP);
        record("v2", "rows", "winner " + tag, B, B, T, rp, "exact", cW);
        record("v2", "rows", "z " + tag, B, B, T, rp, "exact", cz);
        record("v2", "rows", "p " + tag, B, B, T, rp, "exact", cp);
        (void) NS; (void) NP;
    }
}

// ---------------------------------------------------------------------------------------------
// ONE TRAINING STEP: gradients at the same theta (before division), loss, then one update.
static void check_step(const vector<seq::Sample> & data, const seq::Params & P0, int T, double lr, const string & tag) {
    const int B = (int) data.size(), F = P0.F;
    // sequential reference: exactly the body of seq::train_epoch for one batch
    seq::Params ps = P0; seq::WS ws(F); seq::Grad gs(F);
    gs.zero(); double Ls = 0;
    for (auto & s : data) { seq::forward(s.x, ps, ws); Ls += seq::loss(ws, s.y); seq::backward_accumulate(s.x, s.y, ps, ws, gs); }
    // parallel
    tr::RNG dummy(1); tr::Params pt(F, dummy);
    pt.K = P0.K; pt.b = P0.b; pt.Wd = P0.Wd; pt.c = P0.c;
    auto dt = conv_samples<tr::Sample>(data);
    vector<tr::WS> wt(T, tr::WS(F)); vector<tr::Grad> G(T, tr::Grad(F)); vector<double> ll(T); tr::Grad gt(F);
    double Lt = tr::batch_gradient_parallel(dt, 0, B, pt, wt, G, ll, gt, T);
    const string part = partition_of(B, T), rule = "abs1e-10+rel1e-8";
    Cmp cK, cb, cW, cc, cL;
    for (size_t i = 0; i < gs.K.size(); ++i)  cK.tol(gt.K[i], gs.K[i], i);
    for (size_t i = 0; i < gs.b.size(); ++i)  cb.tol(gt.b[i], gs.b[i], i);
    for (size_t i = 0; i < gs.Wd.size(); ++i) cW.tol(gt.Wd[i], gs.Wd[i], i);
    for (size_t i = 0; i < gs.c.size(); ++i)  cc.tol(gt.c[i], gs.c[i], i);
    cL.tol(Lt, Ls, 0);
    record("train", "image", "G_K " + tag, B, B, T, part, rule, cK);
    record("train", "image", "G_b " + tag, B, B, T, part, rule, cb);
    record("train", "image", "G_W " + tag, B, B, T, part, rule, cW);
    record("train", "image", "G_c " + tag, B, B, T, part, rule, cc);
    record("train", "image", "batch loss " + tag, B, B, T, part, rule, cL);
    // one update with the same lr and the same B
    seq::sgd_update(ps, gs, lr, B);
    tr::sgd_update(pt, gt, lr, B);
    Cmp cu;
    long k = 0;
    for (size_t i = 0; i < ps.K.size(); ++i)  cu.tol(pt.K[i], ps.K[i], k++);
    for (size_t i = 0; i < ps.b.size(); ++i)  cu.tol(pt.b[i], ps.b[i], k++);
    for (size_t i = 0; i < ps.Wd.size(); ++i) cu.tol(pt.Wd[i], ps.Wd[i], k++);
    for (size_t i = 0; i < ps.c.size(); ++i)  cu.tol(pt.c[i], ps.c[i], k++);
    record("train", "image", "params after 1 update " + tag, B, B, T, part, rule, cu);
}

// ---------------------------------------------------------------------------------------------
// SEVERAL STEPS: whole epochs, sequential train_epoch vs train_epoch_parallel (loss per epoch).
static void check_epochs(const vector<seq::Sample> & data, const seq::Params & P0, int batch, double lr, int epochs, int T) {
    seq::Params ps = P0;
    tr::RNG dummy(1); tr::Params pt(P0.F, dummy);
    pt.K = P0.K; pt.b = P0.b; pt.Wd = P0.Wd; pt.c = P0.c;
    auto dt = conv_samples<tr::Sample>(data);
    Cmp cl, cp;
    for (int e = 0; e < epochs; ++e) cl.tol(tr::train_epoch_parallel(dt, pt, batch, lr, T), seq::train_epoch(data, ps, batch, lr), e);
    long k = 0;
    for (size_t i = 0; i < ps.Wd.size(); ++i) cp.tol(pt.Wd[i], ps.Wd[i], k++);
    for (size_t i = 0; i < ps.K.size(); ++i)  cp.tol(pt.K[i], ps.K[i], k++);
    const int updates = epochs * (((int) data.size() + batch - 1) / batch);
    const string note = std::to_string(updates) + " updates in both";
    record("train", "image", "epoch losses (" + std::to_string(epochs) + " ep)", (int) data.size(), batch, T,
           "per batch " + partition_of(batch, T), "abs1e-10+rel1e-8", cl, note);
    record("train", "image", "params after epochs", (int) data.size(), batch, T,
           "per batch " + partition_of(batch, T), "abs1e-10+rel1e-8", cp, note);
}

int main(int argc, char ** argv) {
    csv.open("validation.csv");
    csv << "version,decomposition,quantity,images,batch,workers_T,partition,comparison_rule,max_abs_error,"
           "max_rel_error,first_failing_index,non_finite_values,status,notes\n";
    const double lr = 0.05;
    seq::RNG r(42);
    seq::Params P(8, r);                                     // same initial parameters as the reference
    auto syn = seq::synthetic(64, 43);                       // same generator/seed as --synthetic training

    // Controlled pooling tie: an all-zero image with zero conv bias makes every 2x2 window a
    // four-way tie (A = 0 everywhere) -> the winner must be the top-left element.
    seq::Params Ptie = P;
    std::fill(Ptie.b.begin(), Ptie.b.end(), 0.0);
    vector<seq::Sample> tie = { syn[0], seq::Sample{ vector<double>(seq::H * seq::W, 0.0), 3 }, syn[1] };
    long neg = 0;                                             // check that negative preactivations occur
    { seq::WS w(8); seq::forward(syn[0].x, P, w); for (double v : w.S) neg += v < 0; }
    std::printf("negative preactivations in S of image 0: %ld of %d\n\n", neg, 8 * 26 * 26);

    std::puts("== FORWARD (exact equality) ==");
    for (int T : { 1, 2, 4 }) check_forward({ syn.begin(), syn.begin() + 1 }, P, T, "1 image");
    check_forward({ syn.begin(), syn.begin() + 5 }, P, 3, "B=5 uneven");
    check_forward({ syn.begin(), syn.begin() + 3 }, P, 8, "T>B");
    check_forward(tie, Ptie, 2, "pooling tie");
    check_forward({ syn.begin(), syn.begin() + 37 }, P, 5, "B=37");

    std::puts("\n== ONE TRAINING STEP (gradients before division, then one update) ==");
    for (int T : { 1, 2 }) check_step({ syn.begin(), syn.begin() + 1 }, P, T, lr, "1 image");
    check_step({ syn.begin(), syn.begin() + 5 }, P, 3, lr, "B=5 uneven");
    check_step({ syn.begin(), syn.begin() + 3 }, P, 8, lr, "T>B");
    check_step({ syn.begin(), syn.begin() + 32 }, P, 4, lr, "B=32");
    check_step({ syn.begin(), syn.begin() + 32 }, P, 7, lr, "B=32 uneven");

    std::puts("\n== SEVERAL STEPS (whole epochs, 2000 images, batch 32, last batch 16) ==");
    auto big = seq::synthetic(2000, 43);
    for (int T : { 1, 3, 16 }) check_epochs(big, P, 32, lr, 3, T);

    if (argc > 1) {
        std::printf("\n== MNIST (%s) ==\n", argv[1]);
        auto m = seq::load_mnist(argv[1], true, 64);
        check_forward({ m.begin(), m.begin() + 5 }, P, 3, "MNIST B=5");
        check_step({ m.begin(), m.begin() + 32 }, P, 5, lr, "MNIST B=32");
    }
    std::printf("\n%s  (%d failing comparisons)  -> validation.csv\n", n_fail ? "VALIDATION FAIL" : "VALIDATION PASS", n_fail);
    return n_fail ? 1 : 0;
}
