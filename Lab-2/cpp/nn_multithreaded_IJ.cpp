// PC40 Lab 2 - Part I (correctness validation) + Part J (scalability)
// Based on nn_sequential.cpp. Two parallel structures in one binary:
//   --mode create : create P threads per mini-batch, join()   (Part G design)
//   --mode pool   : P persistent workers + std::barrier       (Part K design)
// Both: thread-local accumulators + single-threaded reduction + one update.
//
// Small fully-connected neural network for MNIST:
//     784 inputs -> 64 hidden neurons (ReLU) -> 10 outputs (softmax)
// trained by mini-batch stochastic gradient descent (SGD).
//
// The file is organised in the regions students must identify in Part A:
//   [1] Data loading          load_idx(), synthetic()
//   [2] Model initialisation  NN::NN()
//   [3] Forward pass          first half of NN::sample_grad()
//   [4] Backward pass         second half of NN::sample_grad()
//   [5] Mini-batch loop       inner loop of NN::train()   <-- parallel target
//   [6] Weight update         end of the batch in NN::train()
//   [7] Evaluation            NN::accuracy()
//
// Build:  g++ -O2 -std=c++20 -pthread nn_multithreaded_IJ.cpp -o nn_IJ
// Run:    ./nn_IJ ../data --threads 4 --mode create [--bs 64]
//         ./nn_IJ --synthetic --threads 4 --mode pool

#include <algorithm>
#include <barrier>
#include <iomanip>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using std::vector;

// ---------------------------------------------------------------------------
// [1] Data loading
// ---------------------------------------------------------------------------

// Read a 32-bit big-endian integer (IDX header format).
static uint32_t be32(std::ifstream& f) {
    unsigned char b[4];
    f.read((char*)b, 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | b[3];
}

struct Data {
    vector<float>   x;        // n * d pixels, row-major, normalised to [0,1]
    vector<uint8_t> y;        // n labels in 0..9
    int n = 0;                // number of samples
    int d = 784;              // pixels per sample
};

// Load at most `limit` samples from an IDX image file and its label file.
Data load_idx(const std::string& ip, const std::string& lp, int limit) {
    std::ifstream fi(ip, std::ios::binary), fl(lp, std::ios::binary);
    if (!fi || !fl) throw std::runtime_error("MNIST files not found");
    if (be32(fi) != 2051 || be32(fl) != 2049) throw std::runtime_error("Bad IDX magic");

    int ni = be32(fi), nl = be32(fl), rows = be32(fi), cols = be32(fi);
    int n = std::min({ni, nl, limit});

    Data d;
    d.n = n;
    d.d = rows * cols;
    d.x.resize((size_t)n * d.d);
    d.y.resize(n);
    vector<unsigned char> buf(d.d);
    for (int i = 0; i < n; i++) {
        fi.read((char*)buf.data(), d.d);
        for (int j = 0; j < d.d; j++) d.x[(size_t)i * d.d + j] = buf[j] / 255.0f;
        fl.read((char*)&d.y[i], 1);
    }
    return d;
}

// Synthetic dataset for smoke tests: class y lights up 20 fixed pixels.
Data synthetic(int n) {
    Data d;
    d.n = n;
    d.x.assign((size_t)n * 784, 0);
    d.y.resize(n);
    std::mt19937 g(1);
    std::normal_distribution<float> noise(0, 0.03f);
    for (int i = 0; i < n; i++) {
        int y = i % 10;
        d.y[i] = y;
        for (int j = 0; j < 784; j++) d.x[(size_t)i * 784 + j] = noise(g);
        for (int k = 0; k < 20; k++) d.x[(size_t)i * 784 + (y * 73 + k) % 784] += 1.0f;
    }
    return d;
}

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------
struct NN {
    int D = 784, H = 64, C = 10;
    // Parameters. Layout: W1[j*H+k] connects input j to hidden k,
    //                     W2[k*C+c] connects hidden k to output c.
    vector<float> W1, b1, W2, b2;

    // [2] Model initialisation: deterministic pseudo-random weights (seed 42).
    NN() {
        std::mt19937 g(42);
        std::normal_distribution<float> nd(0, 0.05f);
        W1.resize(D * H);
        b1.assign(H, 0);
        W2.resize(H * C);
        b2.assign(C, 0);
        for (auto& v : W1) v = nd(g);
        for (auto& v : W2) v = nd(g);
    }

    // Forward + backward pass for ONE sample (x, label).
    // Reads:  W1, b1, W2, b2 (the model)              -> shared, read-only
    // Adds to: g1, gb1, g2, gb2 (gradient accumulators),
    //          loss and correct (statistics)          -> whoever owns them
    // The method is `const`: it never modifies the model.
    void sample_grad(const float* x, int label,
                     vector<float>& g1, vector<float>& gb1,
                     vector<float>& g2, vector<float>& gb2,
                     float& loss, int& correct) const {
        // ---- [3] Forward pass -------------------------------------------
        vector<float> h(H), logit(C), prob(C);

        // Hidden layer: h = ReLU(W1^T x + b1)
        for (int k = 0; k < H; k++) {
            float s = b1[k];
            for (int j = 0; j < D; j++) s += x[j] * W1[j * H + k];
            h[k] = std::max(0.0f, s);
        }

        // Output layer: logits = W2^T h + b2
        float m = -1e30f;
        for (int c = 0; c < C; c++) {
            float s = b2[c];
            for (int k = 0; k < H; k++) s += h[k] * W2[k * C + c];
            logit[c] = s;
            m = std::max(m, s);
        }

        // Softmax (shifted by the max for numerical stability)
        float z = 0;
        for (int c = 0; c < C; c++) {
            prob[c] = std::exp(logit[c] - m);
            z += prob[c];
        }
        for (auto& v : prob) v /= z;

        // Statistics: cross-entropy loss and correct prediction count
        loss -= std::log(std::max(prob[label], 1e-8f));
        correct += int(std::max_element(prob.begin(), prob.end()) - prob.begin()) == label;

        // ---- [4] Backward pass ------------------------------------------
        // dL/dlogit = prob - onehot(label)
        vector<float> dl = prob;
        dl[label] -= 1;

        // Output layer gradients
        for (int k = 0; k < H; k++)
            for (int c = 0; c < C; c++) g2[k * C + c] += h[k] * dl[c];
        for (int c = 0; c < C; c++) gb2[c] += dl[c];

        // Back-propagate to the hidden layer through ReLU
        vector<float> dh(H, 0);
        for (int k = 0; k < H; k++) {
            for (int c = 0; c < C; c++) dh[k] += W2[k * C + c] * dl[c];
            if (h[k] <= 0) dh[k] = 0;
        }

        // Hidden layer gradients (the most expensive loop: 784 x 64)
        for (int j = 0; j < D; j++)
            for (int k = 0; k < H; k++) g1[j * H + k] += x[j] * dh[k];
        for (int k = 0; k < H; k++) gb1[k] += dh[k];
    }

    // Split [s, e2) into P contiguous chunks (first `rem` chunks get +1).
    static void split(int s, int e2, int P, vector<int>& a, vector<int>& b) {
        int n = e2 - s, base = n / P, rem = n % P, cur = s;
        for (int p = 0; p < P; p++) {
            int len = base + (p < rem ? 1 : 0);
            a[p] = cur; b[p] = cur + len; cur += len;
        }
    }

    // [6] Reduction (fixed order p = 0..P-1) + exactly ONE weight update.
    void reduce_and_update(int P, int n, float lr,
                           vector<vector<float>>& lg1, vector<vector<float>>& lgb1,
                           vector<vector<float>>& lg2, vector<vector<float>>& lgb2,
                           vector<float>& lloss, vector<int>& lok,
                           vector<float>& g1, vector<float>& gb1,
                           vector<float>& g2, vector<float>& gb2,
                           float& L, int& ok) {
        std::fill(g1.begin(), g1.end(), 0.0f);
        std::fill(gb1.begin(), gb1.end(), 0.0f);
        std::fill(g2.begin(), g2.end(), 0.0f);
        std::fill(gb2.begin(), gb2.end(), 0.0f);
        for (int p = 0; p < P; p++) {
            for (size_t i = 0; i < g1.size(); i++) g1[i] += lg1[p][i];
            for (int i = 0; i < H; i++) gb1[i] += lgb1[p][i];
            for (size_t i = 0; i < g2.size(); i++) g2[i] += lg2[p][i];
            for (int i = 0; i < C; i++) gb2[i] += lgb2[p][i];
            L += lloss[p];
            ok += lok[p];
        }
        float a = lr / n;
        for (size_t i = 0; i < W1.size(); i++) W1[i] -= a * g1[i];
        for (int i = 0; i < H; i++) b1[i] -= a * gb1[i];
        for (size_t i = 0; i < W2.size(); i++) W2[i] -= a * g2[i];
        for (int i = 0; i < C; i++) b2[i] -= a * gb2[i];
    }

    // One worker's job: zero its private buffers, accumulate its chunk.
    void work(const Data& d, int from, int to,
              vector<float>& g1, vector<float>& gb1,
              vector<float>& g2, vector<float>& gb2,
              float& loss, int& correct) const {
        std::fill(g1.begin(), g1.end(), 0.0f);
        std::fill(gb1.begin(), gb1.end(), 0.0f);
        std::fill(g2.begin(), g2.end(), 0.0f);
        std::fill(gb2.begin(), gb2.end(), 0.0f);
        loss = 0; correct = 0;
        for (int i = from; i < to; i++)
            sample_grad(&d.x[(size_t)i * d.d], d.y[i], g1, gb1, g2, gb2, loss, correct);
    }

    // mode = false -> create/join per batch (Part G), true -> persistent pool (Part K)
    void train_parallel(const Data& d, int epochs, int bs, float lr, int P, bool pool) {
        vector<vector<float>> lg1(P, vector<float>(W1.size())), lgb1(P, vector<float>(H));
        vector<vector<float>> lg2(P, vector<float>(W2.size())), lgb2(P, vector<float>(C));
        vector<float> lloss(P);
        vector<int> lok(P), a(P), b(P);
        vector<float> g1(W1.size()), gb1(H), g2(W2.size()), gb2(C);

        // ---- pool infrastructure (only used when pool == true) ----
        bool stop = false;
        std::barrier sync_start(P + 1), sync_done(P + 1);
        vector<std::thread> workers;
        if (pool) {
            for (int p = 0; p < P; p++)
                workers.emplace_back([&, p]() {
                    while (true) {
                        sync_start.arrive_and_wait();      // wait for a batch
                        if (stop) return;
                        work(d, a[p], b[p], lg1[p], lgb1[p], lg2[p], lgb2[p], lloss[p], lok[p]);
                        sync_done.arrive_and_wait();       // batch done
                    }
                });
        }

        for (int e = 0; e < epochs; e++) {
            float L = 0;
            int ok = 0;
            for (int s = 0; s < d.n; s += bs) {
                int e2 = std::min(s + bs, d.n);
                split(s, e2, P, a, b);

                if (pool) {
                    sync_start.arrive_and_wait();          // release workers
                    sync_done.arrive_and_wait();           // wait for all of them
                } else {
                    vector<std::thread> ts;
                    for (int p = 0; p < P; p++)
                        ts.emplace_back([&, p]() {
                            work(d, a[p], b[p], lg1[p], lgb1[p], lg2[p], lgb2[p], lloss[p], lok[p]);
                        });
                    for (auto& t : ts) t.join();           // synchronization point
                }
                // Weights are only touched here, while no worker is running.
                reduce_and_update(P, e2 - s, lr, lg1, lgb1, lg2, lgb2, lloss, lok,
                                  g1, gb1, g2, gb2, L, ok);
            }
            std::cout << "Epoch " << e + 1 << " loss=" << L / d.n
                      << " train_acc=" << 100.0 * ok / d.n << "%\n";
        }

        if (pool) {
            stop = true;
            sync_start.arrive_and_wait();                  // wake workers so they exit
            for (auto& t : workers) t.join();
        }
    }

    // Part I checksum: sum of all parameters (and sum of |w|), in double.
    void checksum(double& sum, double& abs_sum) const {
        sum = abs_sum = 0;
        for (const auto* v : {&W1, &b1, &W2, &b2})
            for (float w : *v) { sum += w; abs_sum += std::fabs(w); }
    }

    // ---- [7] Evaluation ---------------------------------------------------
    // Forward pass only (no softmax needed: argmax of logits == argmax of probs).
    double accuracy(const Data& d) const {
        int ok = 0;
        for (int i = 0; i < d.n; i++) {
            vector<float> h(H), o(C);
            const float* x = &d.x[(size_t)i * d.d];
            for (int k = 0; k < H; k++) {
                float s = b1[k];
                for (int j = 0; j < D; j++) s += x[j] * W1[j * H + k];
                h[k] = std::max(0.0f, s);
            }
            for (int c = 0; c < C; c++) {
                float s = b2[c];
                for (int k = 0; k < H; k++) s += h[k] * W2[k * C + c];
                o[c] = s;
            }
            ok += int(std::max_element(o.begin(), o.end()) - o.begin()) == d.y[i];
        }
        return 100.0 * ok / d.n;
    }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    try {
        bool syn = false, pool = false;
        int P = 1, bs = 64;
        std::string root = "../data";
        for (int i = 1; i < argc; i++) {
            std::string s = argv[i];
            if (s == "--synthetic") syn = true;
            else if (s == "--threads" && i + 1 < argc) P = std::stoi(argv[++i]);
            else if (s == "--bs" && i + 1 < argc) bs = std::stoi(argv[++i]);
            else if (s == "--mode" && i + 1 < argc) {
                std::string m = argv[++i];
                if (m == "pool") pool = true;
                else if (m != "create") throw std::runtime_error("--mode must be create or pool");
            } else root = s;
        }
        if (P < 1) throw std::runtime_error("--threads must be >= 1");

        Data tr = syn ? synthetic(2000)
                      : load_idx(root + "/train-images-idx3-ubyte",
                                 root + "/train-labels-idx1-ubyte", 12000);
        Data te = syn ? synthetic(500)
                      : load_idx(root + "/t10k-images-idx3-ubyte",
                                 root + "/t10k-labels-idx1-ubyte", 2000);

        std::cout << std::setprecision(9);
        std::cout << "Threads: " << P << "  mode: " << (pool ? "pool" : "create")
                  << "  batch: " << bs << "\n";

        NN nn;
        auto t = std::chrono::steady_clock::now();
        nn.train_parallel(tr, /*epochs=*/3, bs, /*learning rate=*/0.08f, P, pool);
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();

        double cs, cabs;
        nn.checksum(cs, cabs);
        std::cout << "Test accuracy: " << nn.accuracy(te) << "%\n"
                  << std::setprecision(15)
                  << "Checksum: sum=" << cs << " abs_sum=" << cabs << "\n"
                  << std::setprecision(9)
                  << "Training time: " << sec << " s\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\nRun data/download_mnist.py or use --synthetic\n";
        return 1;
    }
}
