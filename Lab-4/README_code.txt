PC40 Lab 4 -- Parallelizing Transformers for GenAI
==================================================

Files
-----
PC40_Lab04_Transformer_HPC_v5.ipynb
    The lab notebook with both mandatory functions implemented
    (TODO 2.1 attention_impl, TODO 3.1 parallel_heads), every automatic
    test executed, the three figures, and the answers to Q1-Q14 and the
    four synthesis checkpoints.  Executed on the local CPU node described
    in its section 0, not on Colab; the protocol is unchanged.

bench_lab4.py
    Extended strong-scaling study on the same node.  The notebook caps the
    thread count at 4 (the conservative choice for a shared Colab VM); this
    script sweeps p = 1, 2, 4, 8, 16 and adds the measurements the notebook
    does not make: a workload with H = 16 head tasks, the cost of pool.map
    on its own, a granularity sweep, a sequence-length sweep, prefill versus
    cached decoding, and a control experiment that separates the candidate
    causes of the poor per-head scaling.
    Every configuration is measured once per trial and the trials are
    interleaved, so a slow period of the host hits all configurations
    equally.  Reported values are the median over trials and the best trial.

ceiling_lab4.py
    Measures the limit instead of inferring it from the timings above:
      1. the ceiling of the machine itself, from one large compute-bound GEMM
         at every thread count, together with the ratio of process CPU time to
         wall time, which shows how many cores were really busy;
      2. single-thread throughput of one P-core, one E-core and one LP-E core,
         since a join waits for the slowest worker on a heterogeneous CPU;
      3. the same GEMM and the same per-head pool pinned to homogeneous core
         subsets, which separates the hardware ceiling from the decomposition;
      4. a hypothesis that did NOT survive -- that the dense attention kernel is
         memory-bandwidth-bound -- kept in the output as a rejected hypothesis
         rather than dropped.

plot_simple.py / plot_lab4.py / plot_ceiling.py
    Build the report figures.  plot_simple.py makes the one the synthesis uses
    (execution time and speedup versus p); the other two make the supporting
    figures kept for the oral evaluation.

bench_lab4.csv / bench_lab4.json / ceiling_lab4.json
    Raw results of the runs used in the report (2026-11-16).

L4_time_speedup.png execution time and speedup versus p  (used in the report)
L4_speedup.png      per-head pool and vectorized intra-op, both estimators
L4_control.png      same pool, three kinds of task (first control; superseded
                    by L4_ceiling.png, kept because it is what ruled out the GIL)
L4_length.png       latency versus sequence length against an L^2 reference
L4_generation.png   prefill, one decoding step, and the KV cache
L4_dispatch.png     cost of pool.map with no work in the tasks
L4_ceiling.png      what this machine gives on the friendliest workload
L4_homog.png        the same work on homogeneous core subsets

How to run
----------
PyTorch is not in the system Python.  Create an environment once:

    python3 -m venv venv
    ./venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    ./venv/bin/pip install matplotlib jupyter

Then:

    ./venv/bin/jupyter nbconvert --to notebook --execute --inplace \
        PC40_Lab04_Transformer_HPC_v5.ipynb
    ./venv/bin/python bench_lab4.py --trials 7
    ./venv/bin/python ceiling_lab4.py
    ./venv/bin/python plot_simple.py
    ./venv/bin/python plot_lab4.py
    ./venv/bin/python plot_ceiling.py

Measurement conditions of the recorded run
------------------------------------------
Intel Core Ultra 9 285H, 16 cores, no SMT, hybrid 6 P-cores (5.4 GHz) +
8 E-cores + 2 LP-E cores, 24 MiB L3, 30 GiB RAM, Kubuntu, kernel 7.0.0-43,
Python 3.14.4, PyTorch 2.15.0+cpu, float32, no cgroup CPU quota exposed,
no CFS throttling observed.

The host was an interactive desktop with its usual applications running:
the load average was 1.80 at the start of the benchmark and 6.22 at the end.
This is recorded rather than hidden, and it is why both the median and the
best trial are reported.  It is also the reason the p = 16 points of the
vectorized series are unstable.

The CPU is heterogeneous and this dominates the results: a single P-core
delivers 80.2 GFLOP/s, an E-core 60.3 and an LP-E core 30.3, so a join that
waits for the slowest worker pays a factor of up to 2.64.  On a homogeneous
subset the same code scales at E_p = 0.94.  Anyone repeating these numbers on
a different machine should expect different curves and should record the core
inventory with ceiling_lab4.py before comparing.
