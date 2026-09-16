#include <unistd.h>
#include <iostream>
#include <vector>
#include <sys/wait.h>



int factorial(int number){
    if (number < 0) {
        std::cout << "the number must be larger or equal to 0 !" << std::endl;
        return -1;
    }
    if (number == 0) return 1;
    int result = 1;
    for(int i = 1; i <= number;i++){
        result *= i;
    }
    return result;
}

int main(){


    std::vector<int> numbers = {3,4,2,0,5}; // list for numbers awaiting to be taken factorial of

    std::vector<int> parentReadings; // every reading parent does
    std::vector<pid_t> pids;       // saving every child's pid

    for(int n : numbers){
        int processPipe[2]; // read end [0] -> write end [1]

        if (pipe(processPipe) == -1){
            perror("pipe");
            return 1;
        }

        pid_t pid = fork();

        if(pid == 0){
            close(processPipe[0]);

            int result = factorial(n);
            write(processPipe[1], &result, sizeof(int));

            close(processPipe[1]);
            _exit(0);

        }else{
            close(processPipe[1]);
            parentReadings.push_back(processPipe[0]);
            pids.push_back(pid);
        }

    }

    std::vector<int> results(numbers.size());

    for(int i = 0 ; i < numbers.size();i++){

        read(parentReadings[i], &results[i], sizeof(int));
        close(parentReadings[i]);
        waitpid(pids[i], nullptr, 0);

        std::cout << "Factorial of " << numbers[i] << " = " << results[i] << std::endl;
    }

    return 0;

}