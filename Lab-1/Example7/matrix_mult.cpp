#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>

void mult(int matrixSize,int startRow,int endRow,const std::vector<std::vector<int>> &A,const std::vector<std::vector<int>> &B,std::vector<std::vector<int>> &C){

    for(int i = startRow ; i < endRow ; i++){
        for(int j = 0 ; j < matrixSize;j++){
            int sum = 0;
            for(int k = 0; k < matrixSize;k++){
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
}


int main(){
    // initialize random device to generate random number for the matrix
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(-10, 10);

    int matrixSize = 800;
    std::vector<int> matrixSizes = {100, 200, 400, 800};
    std::vector<int> numberOfThreads = {1, 2, 4, 5, 10};

    // initialize A and B matrices 
    std::vector<std::vector<int>> A(matrixSize, std::vector<int>(matrixSize));
    std::vector<std::vector<int>> B(matrixSize, std::vector<int>(matrixSize));

    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            A[i][j] = distrib(gen);
            B[i][j] = distrib(gen);
        }
    }


    std::cout << " ----------- USING THREADS -----------\n";
    for(int n : numberOfThreads){
        // initialize matrix C
        std::vector<std::vector<int>> C(matrixSize, std::vector<int>(matrixSize, 0));
        std::vector<std::thread> threads;

        int rowsPerThread = matrixSize / n;
        int remainder = matrixSize % n;

        //start the timer
        auto start = std::chrono::steady_clock::now();

        int currentRow = 0;
        for(int i = 0; i < n ; i++){
            int rowsForThisThread = 0;
            if(remainder != 0 ){
                rowsForThisThread = rowsPerThread + 1;
                remainder--;
            }else{
                rowsForThisThread = rowsPerThread;
            }
            threads.emplace_back(mult,matrixSize,currentRow,currentRow + rowsForThisThread,std::cref(A),std::cref(B),std::ref(C));
            currentRow += rowsForThisThread;
            
        }

        for (auto& t : threads) {
            t.join();
        }
        // end the timer
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> timeElapsed = end - start;
        printf("Number of threads = %d\nTime elapsed in seconds = %fs\n", n,timeElapsed.count());
    }
    std::cout << " ------------------------------------------------\n";
    std::cout << " ----------- SEQUENTIAL NO THREADS --------------\n";

    std::vector<double> T1;

    for (int n : matrixSizes) {
        std::vector<std::vector<int>> seqA(n, std::vector<int>(n));
        std::vector<std::vector<int>> seqB(n, std::vector<int>(n));
        std::vector<std::vector<int>> seqC(n, std::vector<int>(n, 0));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                seqA[i][j] = distrib(gen);
                seqB[i][j] = distrib(gen);
            }
        }

        auto start = std::chrono::steady_clock::now();
        mult(n, 0, n, seqA, seqB, seqC);
        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> timeElapsed = end - start;
        T1.push_back(timeElapsed.count());

        printf("n=%d -> T1=%fs\n", n, timeElapsed.count());
    }

    std::cout << "\n --------- THEORETICAL MODEL (Tmodel = C * n^3) ---------\n";
    double n0 = matrixSizes[0];
    double C_const = T1[0] / (n0 * n0 * n0);

    for (size_t i = 0; i < matrixSizes.size(); i++) {
        double Tmodel = C_const * matrixSizes[i] * matrixSizes[i] * matrixSizes[i];
        printf("n=%d -> T1=%f, Tmodel=%f\n", matrixSizes[i], T1[i], Tmodel);
    }

    std::cout << "\n --------- DOUBLING TEST R(n) = T1(2n)/T1(n) ---------\n";
    for (size_t i = 0; i + 1 < matrixSizes.size(); i++) {
        if (matrixSizes[i+1] == 2 * matrixSizes[i]) {
            double R = T1[i+1] / T1[i];
            printf("R(%d->%d) = %f (ideal: 8.0)\n", matrixSizes[i], matrixSizes[i+1], R);
        }
    }




    return 0;
}