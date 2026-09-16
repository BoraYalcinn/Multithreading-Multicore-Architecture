public class Deadlock {

    static final Object L1 = new Object();
    static final Object L2 = new Object();

    static void thread1() {
        synchronized (L1) {
            System.out.println("Thread 1 acquired L1");
            try { Thread.sleep(100); } catch (InterruptedException e) {}

            System.out.println("Thread 1 waiting for L2...");
            synchronized (L2) {
                System.out.println("Thread 1 acquired L2");
            }
        }
    }

    static void thread2() {
        synchronized (L2) {
            System.out.println("Thread 2 acquired L2");
            try { Thread.sleep(100); } catch (InterruptedException e) {}

            System.out.println("Thread 2 waiting for L1...");
            synchronized (L1) {
                System.out.println("Thread 2 acquired L1");
            }
        }
    }

    public static void main(String[] args) throws InterruptedException {
        Thread t1 = new Thread(Deadlock::thread1);
        Thread t2 = new Thread(Deadlock::thread2);

        t1.start();
        t2.start();

        t1.join();
        t2.join();

        System.out.println("Both threads finished (should never print!)");
    }
}
