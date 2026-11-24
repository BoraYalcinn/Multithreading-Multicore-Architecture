PC40 Lab 05 -- Model Validation and Parallel Cross-Validation
=============================================================

Files
-----
PC40_Lab05_NestedCV.ipynb
    The main deliverable. Walks the whole lab: the three validation schemes,
    the sequential nested CV, the three explicit decompositions with their
    correctness check, the benchmarks, the final model, and the two optional
    extensions. All four synthesis checkpoints and the eight "find the
    parallelism" questions are answered in it. Executed on the machine
    described in its section 0.

nested_cv.py
    The implementation. Outer LOOCV over 25 observations, inner 5-fold over
    the 24, a 36-point decision-tree grid, every loop written out by hand:
    the lab forbids GridSearchCV and n_jobs=-1 for the required version.
    Three parallel decompositions share the same primitives --
      A  one task per outer fold          (25 tasks)
      B  one task per hyperparameter      (25 x 36 = 900 tasks)
      C  one task per inner fold          (25 x 36 x 5 = 4500 tasks)
    so the only thing that changes between them is where tasks are created.
    The notebook imports this module and prints the source of each function,
    rather than keeping a second copy that could drift.

bench_lab5.py
    The performance experiments: correctness, decomposition A with a process
    pool and with a thread pool, a positive control, the three decompositions
    against each other, and the extensions. Configurations are measured in
    interleaved trials; median and best trial are both reported.

plot_lab5.py
    Builds the two report figures from bench_lab5.json.

bench_lab5.json          raw results of the run used in the report
L5_time_speedup.png      execution time and speedup, processes vs threads
L5_gil_granularity.png   the cores-busy measurement and the task-count effect

How to run
----------
scikit-learn is not in the system Python. Create an environment once:

    python3 -m venv venv
    ./venv/bin/pip install scikit-learn matplotlib jupyter
    ./venv/bin/pip install xgboost          # optional, for extension 2

Then:

    ./venv/bin/jupyter nbconvert --to notebook --execute --inplace \
        PC40_Lab05_NestedCV.ipynb
    ./venv/bin/python bench_lab5.py --trials 5
    ./venv/bin/python plot_lab5.py

Two things worth knowing before changing anything
--------------------------------------------------
1. The native thread limits (OMP_NUM_THREADS and friends) are set at the TOP
   of nested_cv.py, before numpy is imported. Those variables are only read
   when the library initialises, so setting them later silently does nothing:
   an earlier version of this code set them afterwards, BLAS kept one thread
   per core inside every task, and the positive control reported 13 cores busy
   at p=1, which made it worthless as a control.

2. The process-pool path needs the usual `if __name__ == "__main__":` guard in
   whatever script calls it. Python 3.14 starts workers with forkserver, so a
   module that spawns a pool at import time will re-import itself in the child
   and fail.

Dataset
-------
The brief states the problem for 25 observations but supplies no data, so
make_dataset() generates it from a written-down function with a fixed seed:

    x1, x2, x3 ~ U(0,1)
    y = 8*x1 + 5*sin(3*x2) - 4*x3^2 + e,    e ~ N(0, 0.5^2)

The response is smooth rather than piecewise constant, so a deeper tree fits
the training data better while generalising worse. That is what makes the
validation protocol meaningful rather than decorative.

Measurement conditions of the recorded run
------------------------------------------
Intel Core Ultra 9 285H, 16 logical CPUs, no SMT, hybrid 6 P-cores + 8 E-cores
+ 2 LP-E cores, kernel 7.0.0-43, Python 3.14.4, scikit-learn 1.9.1,
numpy 2.5.5, xgboost 3.4.1, float64, OMP_NUM_THREADS=1.

The host was an ordinary interactive desktop, which is why every configuration
is measured in interleaved trials and both the median and the best trial are
reported.
