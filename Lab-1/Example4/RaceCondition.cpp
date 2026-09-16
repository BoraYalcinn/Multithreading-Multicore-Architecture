#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <vector>

std::mutex m;
static long counter = 0;

void incrementNoSync(long n) {
    for (long i = 0; i < n; i++) counter++;
}
void decrementNoSync(long n) {
    for (long i = 0; i < n; i++) counter--;
}
void incrementSync(long n) {
    for (long i = 0; i < n; i++) {
        std::lock_guard<std::mutex> guard(m);
        counter++;
    }
}
void decrementSync(long n) {
    for (long i = 0; i < n; i++) {
        std::lock_guard<std::mutex> guard(m);
        counter--;
    }
}

void runTest(long n, bool sync) {
    counter = 0;
    auto start = std::chrono::steady_clock::now();

    std::thread t1, t2;
    if (sync) {
        t1 = std::thread(incrementSync, n);
        t2 = std::thread(decrementSync, n);
    } else {
        t1 = std::thread(incrementNoSync, n);
        t2 = std::thread(decrementNoSync, n);
    }
    t1.join();
    t2.join();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "[" << (sync ? "Sync" : "NoSync") << "] n=" << n
              << " counter=" << counter << " time=" << elapsed.count() << "s\n";
}

int main() {
    std::vector<long> sizes = {10000, 100000, 1000000};
    for (long n : sizes) {
        runTest(n, false);
        runTest(n, true);
    }
    return 0;
}

// TO COMPILE g++ -std=c++20 -O2 -pthread RaceCondition2.cpp -o RaceCondition2
