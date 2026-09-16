#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

// to compile g++ -std=c++20 -O2 -pthread MultithreadingExampleWorkers.cpp -o MultithreadingExampleWorkers

const int TOTAL_TASKS = 50;
const long WAIT_MS = 200;

void performTask() {
    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_MS));
}

void runWithWorkers(int p) {
    std::vector<std::thread> workers;

    int tasksPerWorker = TOTAL_TASKS / p;
    int remainder = TOTAL_TASKS % p;

    for (int i = 0; i < p; ++i) {
        int tasksForThisWorker = tasksPerWorker + (i < remainder ? 1 : 0);

        workers.emplace_back([tasksForThisWorker]() {
            for (int j = 0; j < tasksForThisWorker; ++j) {
                performTask();
            }
        });
    }

    for (auto& t : workers) t.join();
}

int main() {
    std::vector<int> workerCounts = {1, 2, 3, 4, 5};
    int repetitions = 2;

    for (int p : workerCounts) {
        double totalTime = 0.0;

        for (int r = 0; r < repetitions; ++r) {
            auto start = std::chrono::steady_clock::now();
            runWithWorkers(p);
            auto end = std::chrono::steady_clock::now();

            std::chrono::duration<double> elapsed = end - start;
            totalTime += elapsed.count();
        }

        double avgSeconds = totalTime / repetitions;
        std::cout << "p=" << p << " -> avg time: " << avgSeconds << " s" << std::endl;
    }

    return 0;

}
