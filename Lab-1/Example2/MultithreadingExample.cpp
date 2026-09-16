#include <iostream>
#include <thread>
#include <chrono>

void performTask(long milliseconds){

    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

int main(){
    // Sequential execution
    auto start = std::chrono::steady_clock::now();
    performTask(2000);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> seqElapsed = end - start;
    std::cout << "Sequential time: " << seqElapsed.count() << " s" << std::endl;

    // Execution using one additional thread
    start = std::chrono::steady_clock::now();
    std::thread t(performTask, 2000);
    t.join();
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double> threadElapsed = end - start;
    std::cout << "One-thread time: " << threadElapsed.count() << " s" << std::endl;


    return 0;
}