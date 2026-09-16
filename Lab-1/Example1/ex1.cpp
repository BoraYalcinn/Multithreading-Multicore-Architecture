#include <iostream>
#include <thread>

void printRange(int start, int end, int threadNum) {
    for(int i = start; i<=end; ++i){
        std::cout << "Thread " << threadNum << ':' << i << std::endl;
    }


}

int main(){

    std::thread t1(printRange,1,5,1);
    std::thread t2(printRange,6,10,2);

    t1.join();
    t2.join();

    std::cout <<"Both threads are finished !" << std::endl;

    return 0;
    // to compile this file run g++ -std=c++20 -O2 -pthread ex1.cpp -o ex1
}