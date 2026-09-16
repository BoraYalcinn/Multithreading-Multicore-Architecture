#include <iostream>
#include <thread>
#include <barrier>
#include <vector>
#include <chrono>

// Game's scene parameters
const int NUMBER_OF_OBJETCS = 4;
std::vector<double> positions(NUMBER_OF_OBJETCS);
std::vector<double> velocities = {1.,2.5,-1.5,.5};
std::mutex coutMutex;


void update(int objectID,std::barrier<>& sync){
    // |----------- PHASE-1 -------------|
    // | Update the position of objects  |
    // |---------------------------------|
    
    positions[objectID] += velocities[objectID];
    {
        std::lock_guard<std::mutex> lock(coutMutex);
        std::cout << "Object " << objectID << " updated position -> " << positions[objectID] << std::endl;
    }

    // makes the results more visible if we didn't have this the difference would only be milliseconds
    std::this_thread::sleep_for(std::chrono::milliseconds(50 * (objectID + 1)));

    sync.arrive_and_wait(); // BARRIER

    // |------------------------ PHASE-2 ------------------------------|
    // | Now the new positions are updated we can check for collisions |
    // |---------------------------------------------------------------|
    for (int other = 0; other < NUMBER_OF_OBJETCS; other++) {

        if (other == objectID){
            continue;
        }

        double distance = std::abs(positions[objectID] - positions[other]);

        if (distance < 1.0) {
            std::lock_guard<std::mutex> lock(coutMutex);
            std::cout << "Object " << objectID << " updated position -> " << positions[objectID] << std::endl;
        }
    }
}


int main(){

    // initialize objects first positions
    for(int i = 0; i < NUMBER_OF_OBJETCS; i++){
        positions[i] = i * 2.0;
    }

    std::barrier sync(NUMBER_OF_OBJETCS);
    
    
    std::vector<std::thread> threads;
    for(int i = 0;i < NUMBER_OF_OBJETCS;i++){

        threads.emplace_back(update,i, std::ref(sync));
    }
    for(auto& t : threads){
        t.join();
    }
    
    std::cout << "Frame update complete." << std::endl;
    return 0;


}