# Lab 1 — Thread Fundamentals, Synchronisation and Process-Based Parallelism

The first block of lab work. Each example is self-contained and can be built and
run on its own. The C++ version is the primary implementation; where a Java
version exists it solves the same problem and is kept for comparison.

The written report is in [`../docs/lab-1/`](../docs/lab-1/).

## Building

    make              # build everything into bin/
    make run-race     # build and run a single example
    make help         # list all targets

## Examples

**Example 1** (`ex1`) — Two threads print disjoint integer ranges. Introduces
`std::thread` construction and `join()`, and shows that output interleaving is
not deterministic.

**Example 2** (`ex2`) — The same fixed-duration task executed sequentially, on
one additional thread, and on two threads, with wall-clock timings for each.

**Example 2, Parts B and C** (`ex2_workers`) — Fifty tasks of 200 ms each,
distributed over `p` worker threads. Produces the execution times from which
speedup `Sp = T1/Tp` and efficiency `Ep = Sp/p` are derived and plotted.

**Example 3** (`deadlock`, `deadlock_fixed`) — Two threads acquiring two mutexes
in opposite order, producing a circular wait. The corrected version imposes a
consistent global lock ordering.

**Example 4** (`race`) — A shared counter incremented and decremented
concurrently, first without synchronisation and then under a `std::mutex`,
showing that the unsynchronised result is neither correct nor reproducible.

**Example 5** (`producer`) — A bounded-capacity inventory with multiple
producers and consumers, coordinated by two condition variables so that
producers block when the inventory is full and consumers block when it is empty.

**Example 6** (`factorial`) — Factorials computed in separate child processes
created with `fork()`, with the parent collecting them via `waitpid()`.
Contrasts process isolation with the shared address space of threads.

**Example 7** (`matmul_mt`, `matmul_mp`) — Matrix multiplication parallelised by
row blocks, implemented once with threads sharing the result matrix directly and
once with processes writing into a shared anonymous `mmap()` region.

**Example 8** (`barrier`) — A two-phase scene update in which all threads must
finish updating positions before any may read them, enforced with
`std::barrier`.

**Example 9** (`mapreduce`) — A map-reduce decomposition of a sum of squares:
child processes compute partial results into shared memory and the parent
reduces them.

**Example 10** (`io_seq`, `io_threads`, `io_procs`) — An I/O-bound workload of
twelve HTTP downloads, executed sequentially, with a thread pool, and with a
process pool. Demonstrates that concurrency gains on blocking I/O are not
limited by the core count.

## Notes on individual examples

### Example 2, Parts B and C

Run the measurement, then generate the figures:

    make run-ex2_workers
    python3 Example2-Part-BandC/plot_results.py

The `Tp` list inside `plot_results.py` holds the measured times; update it if
you re-run the measurement on different hardware. The script writes
`Example2-Part-BandC/ex2_partC_curves.png`.

### Example 3

`make run-deadlock` deliberately hangs. Observe the interleaved output, then
terminate it with `Ctrl+C`. `make run-deadlock_fixed` shows the corrected
version running to completion.

### Example 10

The downloads target a local web server. Start it first, from the directory
containing the sample files:

    cd Example10/webroot && python3 -m http.server 8001

Then, in another terminal, run `make run-io_seq`, `make run-io_threads` and
`make run-io_procs`. The libcurl development package must be installed.

## Java versions

    make java
    java -cp bin/java RaceCondition

Java implementations exist for Examples 1, 2, 2B/C, 3, 4 and 5.
