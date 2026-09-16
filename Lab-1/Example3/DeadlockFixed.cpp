#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

std::mutex L1, L2;

void thread1() {
    // Her iki thread de AYNI SIRADA kilit alıyor: önce L1, sonra L2
    std::lock_guard<std::mutex> lock1(L1);
    std::cout << "Thread 1 acquired L1" << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "Thread 1 waiting for L2..." << std::endl;
    std::lock_guard<std::mutex> lock2(L2);
    std::cout << "Thread 1 acquired L2" << std::endl;
}

void thread2() {
    // DÜZELTME: Thread 2 de önce L1, sonra L2 sırasını izliyor (ters değil)
    std::lock_guard<std::mutex> lock1(L1);
    std::cout << "Thread 2 acquired L1" << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "Thread 2 waiting for L2..." << std::endl;
    std::lock_guard<std::mutex> lock2(L2);
    std::cout << "Thread 2 acquired L2" << std::endl;
}

int main() {
    std::thread t1(thread1);
    std::thread t2(thread2);

    t1.join();
    t2.join();

    std::cout << "Both threads finished successfully." << std::endl;
    return 0;
}
