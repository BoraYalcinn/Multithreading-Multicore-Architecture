#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <string>
#include <chrono>

struct Slot {
    std::string itemType = "";   // empty slot
    int quantity = 0;
    int slotLimit = 20;
};

const int NUM_SLOTS = 5;   // there 5 unique item slots in the inventory and each has 20 item limit
std::vector<Slot> inventory(NUM_SLOTS);

std::mutex m;
std::condition_variable slotAvailable; // producers want to know if there are available slots
std::condition_variable itemAvailable; // consumers want to knmow if there are available items to consume

// active producers and active lock
int activeProducers;
std::mutex activeMutex;

// to check if the requested item type already exist or not 
int findSlotFor(const std::string& itemType) {
    int emptySlot = -1;
    for (int i = 0; i < NUM_SLOTS; ++i) {
        if (inventory[i].itemType == itemType && inventory[i].quantity < inventory[i].slotLimit)
            return i; // if the item type exist first fill that 
        if (inventory[i].itemType == "" && emptySlot == -1)
            emptySlot = i; // if all slots are full
    }
    return emptySlot; // if there are same types and 1 slot is already full move to the next empty slot (-1 means there are no spaces left)
}

void producer(int producerID, const std::string& itemType, int itemCount) {
    for (int i = 0; i < itemCount; ++i) {
        std::unique_lock<std::mutex> lock(m);

        slotAvailable.wait(lock, [&] { return findSlotFor(itemType) != -1; });

        int idx = findSlotFor(itemType);
        inventory[idx].itemType = itemType; 
        inventory[idx].quantity++;

        std::cout << "[Producer " << producerID << "] added " << itemType
                  << " -> slot " << idx << " (id=" << idx << "-" << inventory[idx].quantity
                  << "), quantity now=" << inventory[idx].quantity << "/" << inventory[idx].slotLimit
                  << std::endl;

        lock.unlock();
        itemAvailable.notify_one();

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::lock_guard<std::mutex> guard(activeMutex);
    activeProducers--;
    if (activeProducers == 0) itemAvailable.notify_all();
}

void consumer(int consumerID) {
    while (true) {
        std::unique_lock<std::mutex> lock(m);

        itemAvailable.wait(lock, [] {
            for (auto& s : inventory) {
                if (s.quantity > 0){
                    return true;
                } 
            return activeProducers == 0;
            }
        });

        int idx = -1;
        for (int i = 0; i < NUM_SLOTS; ++i) {
            if (inventory[i].quantity > 0) {
                 idx = i;
                  break;
                }
        }

        if (idx == -1) {
            break;
        } 

        inventory[idx].quantity--;
        std::string consumedType = inventory[idx].itemType;
        int remaining = inventory[idx].quantity;

        if (remaining == 0) inventory[idx].itemType = ""; 

        std::cout << "    [Consumer " << consumerID << "] took " << consumedType  << " from slot " << idx << ", quantity now=" << remaining << std::endl;

        lock.unlock();
        slotAvailable.notify_one();

        std::this_thread::sleep_for(std::chrono::milliseconds(70));
    }
}

int main() {
    // 3 producers
    activeProducers = 3;

    std::vector<std::thread> producers;
    producers.emplace_back(producer, 0, "apple", 8);
    producers.emplace_back(producer, 1, "pear", 8);
    producers.emplace_back(producer, 2, "banana", 8);

    std::vector<std::thread> consumers;
    for (int i = 0; i < 2; ++i)
        consumers.emplace_back(consumer, i);

    for (auto& p : producers) p.join();
    for (auto& c : consumers) c.join();

    std::cout << "All producers and consumers finished." << std::endl;
    return 0;
}