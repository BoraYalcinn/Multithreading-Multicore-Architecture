import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;


class ProducerConsumer{

    final static int NUM_SLOTS = 5;
    
    static ReentrantLock lock = new ReentrantLock();
    static Condition slotAvailable = lock.newCondition();
    static Condition itemAvailable = lock.newCondition();
    
    static int activeProducers = 0;
    static ReentrantLock activeLock = new ReentrantLock();

    static class Slot{

        String itemType = "";
        int quantity = 0;
        static int slotLimit = 20;
    }

    static ArrayList<Slot> inventory = new ArrayList(){};

    static int findSlotFor(String itemType) {
        int emptySlot = -1;
        for (int i = 0; i < inventory.size(); i++) {
            Slot s = (Slot) inventory.get(i);
            if (s.itemType.equals(itemType) && s.quantity < s.slotLimit) {
                return i;
            }
            if (s.itemType.equals("") && emptySlot == -1) {
                emptySlot = i;
            }
        }
        return emptySlot;
    }

    static void producer(int producerId, String itemType, int itemCount) throws InterruptedException {
        for (int i = 0; i < itemCount; i++) {
            lock.lock();
            try {
                while (findSlotFor(itemType) == -1) {
                    slotAvailable.await();
                }
                int index = findSlotFor(itemType);
                inventory.get(index).itemType = itemType;
                inventory.get(index).quantity++;

                System.out.println("[Producer " + producerId + "] added " + itemType
                    + " -> slot " + index + " (id=" + index + "-" + inventory.get(index).quantity
                    + "), quantity now=" + inventory.get(index).quantity + "/" + inventory.get(index).slotLimit);

                itemAvailable.signal();
            } finally {
                lock.unlock();
            }
            Thread.sleep(50);
        }

        lock.lock();
        try {
            activeProducers--;
            if (activeProducers == 0) {
                itemAvailable.signalAll();
            }
        } finally {
            lock.unlock();
        }
    }

    static void consumer(int consumerId) throws InterruptedException {
        while (true) {
            int idx = -1;
            String consumedType = null;
            int remaining = 0;

            lock.lock();
            try {
                while (allEmpty() && activeProducers > 0) {
                    itemAvailable.await();
                }

                // which slot has items
                for (int i = 0; i < inventory.size(); i++) {
                    if (inventory.get(i).quantity > 0) {
                        idx = i;
                        break;
                    }
                }

                if (idx == -1) {
                    // no item left
                    break;
                }

                inventory.get(idx).quantity--;
                consumedType = inventory.get(idx).itemType;
                remaining = inventory.get(idx).quantity;

                if (remaining == 0) {
                    inventory.get(idx).itemType = "";
                }

                System.out.println("    [Consumer " + consumerId + "] took " + consumedType
                    + " from slot " + idx + ", quantity now=" + remaining);

                slotAvailable.signal();
            } finally {
                lock.unlock();
            }

            Thread.sleep(70);
        }
    }

    static boolean allEmpty() {
        for (Slot s : inventory) {
            if (s.quantity > 0) return false;
        }
        return true;
    }

    public static void main(String[] args)throws InterruptedException {

        activeProducers = 3;

        for (int i = 0; i < NUM_SLOTS; i++) {
            inventory.add(new Slot());
        }

        List<Thread> producers = new ArrayList<>();
        producers.add(new Thread(() -> {
            try { producer(0, "apple", 8); } catch (InterruptedException e) {}
        }));
        producers.add(new Thread(() -> {
            try { producer(1, "pear", 8); } catch (InterruptedException e) {}
        }));
        producers.add(new Thread(() -> {
            try { producer(2, "banana", 8); } catch (InterruptedException e) {}
        }));

        List<Thread> consumers = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            int consumerId = i; 
            consumers.add(new Thread(() -> {
                try { consumer(consumerId); } catch (InterruptedException e) {}
            }));
        }

        for (Thread t : producers) t.start();
        for (Thread t : consumers) t.start();

        for (Thread t : producers) t.join();
        for (Thread t : consumers) t.join();

        System.out.println("All producers and consumers finished.");


    }

}   