import java.util.ArrayList;
import java.util.List;

public class MultithreadingExampleWorkers{


    static final int TOTAL_TASKS = 50;
    static final long WAIT_MS = 200;

    public static void main(String[] args) throws InterruptedException{

        int[] workerCounts = {1,2,3,4,5};

        for (int p : workerCounts) {
            
            long totalTime = 0;
            int repetitions = 2;

            for (int r = 0; r < repetitions; r++) {
                long start = System.nanoTime();
                runWithWorkers(p);
                long end = System.nanoTime();


                totalTime += (end - start);
            }

            double avgSeconds = (totalTime / (double) repetitions) / 1_000_000_000.0;

            System.out.printf("p=%d -> avg time: %.3f s%n", p, avgSeconds);
        }
    }


        static void runWithWorkers(int p) throws InterruptedException {

            List<Thread> workers = new ArrayList<>();
            int tasksPerWorker = TOTAL_TASKS / p;
            int remainder = TOTAL_TASKS % p;

            int assigned = 0;

            for (int i = 0; i < p; i++) {

                int tasksForThisWorker = tasksPerWorker + (i < remainder ? 1 : 0);
                int finalTasksForThisWorker = tasksForThisWorker;
                assigned += tasksForThisWorker;
 
                Thread t = new Thread(() -> {
                    for (int j = 0; j < finalTasksForThisWorker; j++) {
                        performTask();
                    }
                });

                workers.add(t);
            }

            for (Thread t : workers) t.start();
            for (Thread t : workers) t.join();


        }

        static void performTask() {

            try {

                Thread.sleep(WAIT_MS);

            } catch (InterruptedException e) {

                Thread.currentThread().interrupt();
            }
        }
}

    


