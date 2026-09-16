#include <iostream>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <chrono>
#include <random>

std::vector<int> map_phase(const std::vector<int>& numberList){
    std::vector<int> result;
    for(int n : numberList){
        result.push_back(n*n);
    }
    return result;
}

int reduce_phase(const std::vector<int>& squared){
    int sum = 0;
    for(int n : squared){
        sum += n;
    }
    return sum;
}


int main(){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(0, 1000);

    std::vector<int> numberList;
    for(int i = 0; i < 1000;i++){
        numberList.push_back(distrib(gen));
    }
    
    
    //------SEQUENTIAL-------
    auto timerStart = std::chrono::steady_clock::now();
    // MAP PHASE
    std::vector<int> squaredNumberList = map_phase(std::cref(numberList));

    // REDUCE PHASE
    int sum = reduce_phase(std::cref(squaredNumberList));
    auto timerEnd = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeElapsed = timerEnd - timerStart;

    std::cout << "SEQUENTIAL : Sum of squares = " << sum << std::endl;
    std::cout << "             Time Elapsed   = " << timeElapsed << "s" << std::endl;
    std::cout << "--------------------------------\n";

    //------PARALLEL-------
    int numberOfProcesses = 4;

    // SHARED MEMORY
    long long* partialSums = static_cast<long long*>(mmap(nullptr,sizeof(long long) * numberOfProcesses, PROT_READ | PROT_WRITE,MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    int itemsPerProcess = numberList.size() / numberOfProcesses;
    int remainder = numberList.size() % numberOfProcesses;

    std::vector<pid_t> children;
    int currentStart = 0;
    std::vector<int> squared;
    
    auto parallelStart = std::chrono::steady_clock::now();

    for(int i = 0; i < numberOfProcesses;i++){
        int itemsForThisProcess = itemsPerProcess;
        if(remainder != 0){
            itemsForThisProcess++;
            remainder--;
        }
        int start = currentStart;
        int end = currentStart + itemsForThisProcess;
        currentStart = end;

        pid_t pid = fork();

        if(pid == 0){
            std::vector<int> subList(numberList.begin() + start, numberList.begin() + end);
            std::vector<int> squaredSubList = map_phase(subList);
            long long partialSum = reduce_phase(squaredSubList);

            partialSums[i] = partialSum;
            _exit(0);
        }else{
            children.push_back(pid);
        }
    }
    for (const auto& child : children) {
        waitpid(child, nullptr, 0);
    }

    long long totalSum = 0;
    for (int i = 0; i < numberOfProcesses; i++) {
        totalSum += partialSums[i];
    }
    
    auto parallelEnd = std::chrono::steady_clock::now();
    std::chrono::duration<double> parallelTimeElapsed = parallelEnd - parallelStart;


    std::cout << "PARALLEL   : Sum of squares = " << totalSum << std::endl;
    std::cout << "             Time Elapsed   = " << parallelTimeElapsed.count() << "s" << std::endl;

    munmap(partialSums, sizeof(long long) * numberOfProcesses);

    return 0;
}