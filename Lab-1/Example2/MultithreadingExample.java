

public class MultithreadingExample{


    public static void main(String[] args) throws InterruptedException{

        long start = System.nanoTime(); // initial time
        performTask(2000);
        long end = System.nanoTime(); // end time

        System.out.printf("Sequential time: %.3f s%n",(end-start)/1_000_000_000.0);

        start = System.nanoTime();

        Thread thread = new Thread( () -> performTask(2000));
        thread.start();
        thread.join();

        end = System.nanoTime();
        System.out.printf("One-thread time: %.3f s%n",(end-start)/1_000_000_000.0);

        
    }

    public static void performTask(long milliseconds) {
        try {
            Thread.sleep(milliseconds);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }


}