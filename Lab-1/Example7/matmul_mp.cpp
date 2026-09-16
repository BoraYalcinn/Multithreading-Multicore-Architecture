#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>

void mult(int matrixSize, int startRow, int endRow,
          const std::vector<std::vector<int>> &A,
          const std::vector<std::vector<int>> &B,
          int* result) {

    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < matrixSize; j++) {
            int sum = 0;
            for (int k = 0; k < matrixSize; k++) {
                sum += A[i][k] * B[k][j];
            }
            result[i * matrixSize + j] = sum;
        }
    }
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(-10, 10);

    int matrixSize = 800;
    std::vector<int> numberOfProcesses = {1, 2, 4, 5, 10};

    std::vector<std::vector<int>> A(matrixSize, std::vector<int>(matrixSize));
    std::vector<std::vector<int>> B(matrixSize, std::vector<int>(matrixSize));

    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            A[i][j] = distrib(gen);
            B[i][j] = distrib(gen);
        }
    }

    std::cout << " ------- USING PROCESSES --------\n";
    for (int n : numberOfProcesses) {

        // allocate space in shared memory for result matrix
        int* result = static_cast<int*>(mmap(nullptr, sizeof(int) * matrixSize * matrixSize,PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));

        int rowsPerProcess = matrixSize / n;
        int remainder = matrixSize % n;

        auto start = std::chrono::steady_clock::now();

        std::vector<pid_t> children;
        int currentRow = 0;
        for (int i = 0; i < n; i++) {
            int rowsForThisProcess = 0;
            if (remainder != 0) {
                rowsForThisProcess = rowsPerProcess + 1;
                remainder--;
            } else {
                rowsForThisProcess = rowsPerProcess;
            }

            pid_t pid = fork();

            if (pid == 0) {
                // CHILD process
                mult(matrixSize, currentRow, currentRow + rowsForThisProcess, A, B, result);
                _exit(0);
            } else if (pid > 0) {
                // PARENT process
                children.push_back(pid);
            } else {
                std::cerr << "Fork Failed" << std::endl;
                return 1;
            }

            currentRow += rowsForThisProcess;
        }

        for (const auto& child : children) {
            waitpid(child, nullptr, 0);
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> timeElapsed = end - start;
        printf("Number of processes = %d\nTime elapsed in seconds = %fs\n", n, timeElapsed.count());

        munmap(result, sizeof(int) * matrixSize * matrixSize);
    }
    std::cout << " -------------------------------\n";

    return 0;
}