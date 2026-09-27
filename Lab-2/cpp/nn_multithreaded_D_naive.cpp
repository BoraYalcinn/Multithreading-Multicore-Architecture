// PC40: Parallel Computing, Fall 2026 (A26), J.Gaber, gaber@utbm.fr
// Part D/E - naive multithreaded version.
// Shared accumulators (g1, gb1, g2, gb2, loss, correct) are written by
// multiple threads with NO synchronization. This is intentional: it exists
// to demonstrate the race condition described in Part E.
//
// Build: g++ -O2 -std=c++17 -pthread nn_multithreaded_D_naive.cpp -o nn_D
// TSan:  g++ -std=c++17 -O1 -g -fsanitize=thread -pthread nn_multithreaded_D_naive.cpp -o nn_tsan

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using std::vector;

// ---------------------------------------------------------------------------
// [1] Data loading
// ---------------------------------------------------------------------------
static uint32_t be32(std::ifstream& f) {
    unsigned char b[4];
    f.read((char*)b, 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | b[3];
}

struct Data {
    vector<float>   x;
    vector<uint8_t> y;
    int n = 0;
    int d = 784;
};

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
    vector<float> W1, b1, W2, b2;

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

    void sample_grad(const float* x, int label,
                     vector<float>& g1, vector<float>& gb1,
                     vector<float>& g2, vector<float>& gb2,
                     float& loss, int& correct) const {
        vector<float> h(H), logit(C), prob(C);

        for (int k = 0; k < H; k++) {
            float s = b1[k];
            for (int j = 0; j < D; j++) s += x[j] * W1[j * H + k];
            h[k] = std::max(0.0f, s);
        }

        float m = -1e30f;
        for (int c = 0; c < C; c++) {
            float s = b2[c];
            for (int k = 0; k < H; k++) s += h[k] * W2[k * C + c];
            logit[c] = s;
            m = std::max(m, s);
        }

        float z = 0;
        for (int c = 0; c < C; c++) {
            prob[c] = std::exp(logit[c] - m);
            z += prob[c];
        }
        for (auto& v : prob) v /= z;

        loss -= std::log(std::max(prob[label], 1e-8f));
        correct += int(std::max_element(prob.begin(), prob.end()) - prob.begin()) == label;

        vector<float> dl = prob;
        dl[label] -= 1;

        for (int k = 0; k < H; k++)
            for (int c = 0; c < C; c++) g2[k * C + c] += h[k] * dl[c];
        for (int c = 0; c < C; c++) gb2[c] += dl[c];

        vector<float> dh(H, 0);
        for (int k = 0; k < H; k++) {
            for (int c = 0; c < C; c++) dh[k] += W2[k * C + c] * dl[c];
            if (h[k] <= 0) dh[k] = 0;
        }

        for (int j = 0; j < D; j++)
            for (int k = 0; k < H; k++) g1[j * H + k] += x[j] * dh[k];
        for (int k = 0; k < H; k++) gb1[k] += dh[k];
    }

    // ---- Part D/E: naive multithreaded training ---------------------------
    // Each mini-batch is split across P worker threads. All P workers write
    // directly into the SAME shared accumulators (g1, gb1, g2, gb2, L, ok)
    // with no mutex. This is a deliberate, unsafe design used to expose the
    // race condition in Part E.
    void train_parallel(const Data& d, int epochs, int bs, float lr, int P) {
        for (int e = 0; e < epochs; e++) {
            float L = 0;
            int ok = 0;

            for (int s = 0; s < d.n; s += bs) {
                int e2 = std::min(s + bs, d.n);
                int n = e2 - s;

                vector<float> g1(W1.size(), 0.0f), gb1(H, 0.0f);
                vector<float> g2(W2.size(), 0.0f), gb2(C, 0.0f);

                int base = n / P;
                int remainder = n % P;

                vector<std::thread> workers;
                int start = s;
                for (int p = 0; p < P; p++) {
                    int count = base + (p < remainder ? 1 : 0);
                    int end = start + count;

                    // UNSAFE: g1, gb1, g2, gb2, L, ok are shared and written
                    // by every worker with no synchronization whatsoever.
                    workers.emplace_back([this, &d, start, end, &g1, &gb1, &g2, &gb2, &L, &ok]() {
                        for (int i = start; i < end; i++) {
                            sample_grad(&d.x[(size_t)i * d.d], d.y[i], g1, gb1, g2, gb2, L, ok);
                        }
                    });

                    start = end;
                }

                for (auto& t : workers) t.join();

                float a = lr / n;
                for (size_t i = 0; i < W1.size(); i++) W1[i] -= a * g1[i];
                for (int i = 0; i < H; i++) b1[i] -= a * gb1[i];
                for (size_t i = 0; i < W2.size(); i++) W2[i] -= a * g2[i];
                for (int i = 0; i < C; i++) b2[i] -= a * gb2[i];
            }

            std::cout << "Epoch " << e + 1 << " loss=" << L / d.n
                      << " train_acc=" << 100.0 * ok / d.n << "%\n";
        }
    }

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
        bool syn = false;
        std::string root = "../data";
        int P = 1;

        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--synthetic") syn = true;
            else if (arg == "--threads" && i + 1 < argc) P = std::stoi(argv[++i]);
            else if (arg[0] != '-') root = arg;
        }

        Data tr = syn ? synthetic(2000)
                      : load_idx(root + "/train-images-idx3-ubyte",
                                 root + "/train-labels-idx1-ubyte", 12000);
        Data te = syn ? synthetic(500)
                      : load_idx(root + "/t10k-images-idx3-ubyte",
                                 root + "/t10k-labels-idx1-ubyte", 2000);

        NN nn;
        std::cout << "Workers P = " << P << "\n";

        auto t = std::chrono::steady_clock::now();
        nn.train_parallel(tr, /*epochs=*/3, /*batch size=*/64, /*learning rate=*/0.08f, P);
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();

        std::cout << "Test accuracy: " << nn.accuracy(te) << "%\n"
                  << "Training time: " << sec << " s\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\nRun data/download_mnist.py or use --synthetic\n";
        return 1;
    }
}