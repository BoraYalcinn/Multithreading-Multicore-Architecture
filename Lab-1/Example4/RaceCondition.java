



class RaceCondition {

    static long counter = 0;
    static final Object lock = new Object();

    static void incrementWithoutSync(long number){
        
        for(int i = 0; i < number ; i++){
            counter++;
        }

    }

    static void decrementWithoutSync(long number){

        for(int i = 0; i< number ; i++){
            counter--;
        }

    }

    static void incrementWithSync(long number){

        for (int i = 0; i < number;i++ ){
            synchronized (lock) {
                counter++;
            }
        }
    }

    static void decrementWithSync(long number){

        for (int i = 0; i < number;i++ ){
            synchronized (lock) {
                counter--;
            }
        }
    }

    static double run(long number,boolean isSync) throws InterruptedException{

        // initiliaze counter to 0 in the start of the test for each time we test
        counter = 0;
        long start = System.nanoTime();

        Thread t1;
        Thread t2;

        if(isSync){
            t1 = new Thread(() -> incrementWithSync(number));
            t2 = new Thread( () -> decrementWithSync(number));
        }else{
            t1 = new Thread( () -> incrementWithoutSync(number));
            t2 = new Thread( () -> decrementWithoutSync(number));
        }
        t1.start();
        t2.start();

        t1.join();
        t2.join();

        long end = System.nanoTime();
        double timeElapsedinSeconds = ( end - start )/ 1_000_000_000.0;

        if(isSync){
            System.out.println("Number of Sync Iterations = "+ number + ", Final Counter = " + counter + ", Time Elapsed (seconds) = " + timeElapsedinSeconds);
        }else{
            System.out.println("Number of NonSync Iterations = "+ number + ", Final Counter = " + counter + ", Time Elapsed (seconds) = " + timeElapsedinSeconds);
        }


        return timeElapsedinSeconds;
    }

    public static void main(String[] args) throws InterruptedException{

        // number of iteration = 10_000
        run(10_000,false);
        run(10_000,true);

        System.out.println("-----------------------");

        // number of iteration = 100_000
        run(100_000,false);
        run(100_000,true);

        System.out.println("-----------------------");

        // number of iteration = 1_000_000
        run(1_000_000,false);
        run(1_000_000,true);



    }




}