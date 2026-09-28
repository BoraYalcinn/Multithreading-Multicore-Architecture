// PC40 Lab 2 - Optional HPC challenge: false sharing
// P threads, each increments ONLY its own counter (logically independent).
//   compact : counters[p] next to each other -> several share one 64-byte cache line
//   padded  : alignas(64) -> each counter on its own cache line
// Build: g++ -O2 -std=c++20 -pthread false_sharing.cpp -o false_sharing
// Run:   ./false_sharing [P]
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

struct Compact { long v = 0; };                 // 8 bytes -> 8 counters per cache line
struct alignas(64) Padded { long v = 0; };      // 64 bytes -> 1 counter per cache line

constexpr long N = 100'000'000;                 // increments per thread

template <class T>
double run(int P) {
    std::vector<T> c(P);
    auto t = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int p = 0; p < P; p++)
        ts.emplace_back([&c, p]() {
            volatile long* x = &c[p].v;         // volatile: force a real memory write each time
            for (long i = 0; i < N; i++) *x = *x + 1;
        });
    for (auto& th : ts) th.join();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    for (auto& e : c) if (e.v != N) std::cerr << "wrong count!\n";
    return s;
}

int main(int argc, char** argv) {
    int P = argc > 1 ? std::stoi(argv[1]) : 4;
    std::cout << "sizeof(Compact)=" << sizeof(Compact)
              << "  sizeof(Padded)=" << sizeof(Padded) << "  P=" << P << "\n";
    for (int r = 1; r <= 3; r++)
        std::cout << "run " << r << "  compact: " << run<Compact>(P)
                  << " s   padded: " << run<Padded>(P) << " s\n";
}
