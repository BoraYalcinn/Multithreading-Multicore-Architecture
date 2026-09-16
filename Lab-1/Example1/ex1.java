 

public class ex1 {


    static class printRange extends Thread{

        int start;
        int end;
        int threadNumber;

        public printRange(int start,int end,int threadNumber){
            this.start = start;
            this.end = end;
            this.threadNumber = threadNumber;
        }

        public void run(){
            for(int i = start ; i <= end;i++ ){
                System.out.println("Thread "+threadNumber + ": " + i);
            }
        }
    }

    public static void main(String[] args) throws InterruptedException{

        printRange thread1 = new printRange(1,5,1);
        printRange thread2 = new printRange(6,10,2);
        
        thread1.start();
        thread2.start();
        
        thread1.join();
        thread2.join();

        System.out.println("Both threads finished.");

    }


}