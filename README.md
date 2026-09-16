# Multithreading and Multicore Architecture

Lab work and written reports from a course on multithreaded and parallel
programming taken at UTBM (Université de Technologie de Belfort-Montbéliard),
together with the notes I made while working through the material.

For most of the history of computing, a program became faster simply because the
next generation of processors ran at a higher clock frequency. That is no longer
the case. Power density and heat dissipation put an end to frequency scaling in
the mid-2000s, and the industry answered by putting more cores on a die rather
than making a single core faster. The consequence for anyone writing software is
direct: performance is no longer something the hardware hands you for free. A
single-threaded program on a sixteen-core machine leaves fifteen cores idle, and
no compiler will fix that on its own. Getting the remaining performance out of a
modern machine means reasoning explicitly about how work is decomposed, how it is
distributed across cores, and how the pieces communicate once they are running
concurrently.

That reasoning is what this repository is about. The examples here are small on
purpose, because the difficulties in parallel programming are rarely difficulties
of scale. Two threads incrementing the same counter without synchronisation are
enough to produce a result that is wrong and, worse, not reproducible. Two
threads acquiring two mutexes in opposite order are enough to deadlock a program
that is otherwise correct. A speedup curve measured over five worker threads
already shows where the theoretical `Sp = p` line stops holding and why. Working
through these cases one at a time, and measuring rather than assuming, is how
the underlying model becomes intuitive.

The same ideas extend well beyond a handful of CPU threads. The GPU is the
clearest example of where this line of thinking leads: thousands of lightweight
threads executing in lockstep, a memory hierarchy whose behaviour dominates
performance far more than the arithmetic does, and a programming model in which
occupancy, divergence and memory coalescing decide whether a kernel runs at a
fraction of its potential or close to the hardware's limit. Computer graphics
and GPU architecture, and the optimisation work that both demand, are the parts
of this field I care most about, and they are what led me to want a solid footing
in the fundamentals first. Concurrency on the CPU is where those fundamentals
are easiest to see clearly, and this repository is my record of learning them
properly rather than approximately.

I am publishing it because the examples turned out to be reasonably didactic on
their own, and because material of this kind was useful to me when I was looking
for it. If you are working through similar topics, I hope you find something here
worth reading.

## Repository layout

`Lab-N/` holds the source for one lab together with a `Makefile` and a README
describing the individual examples. `docs/` holds the written report for each
lab and a template used when starting a new one. Compiled output is written to
`Lab-N/bin/` and is not tracked by git.

## Topics covered

Thread creation and joining; measurement of speedup and efficiency against the
ideal linear case; race conditions and mutual exclusion; deadlock and its
resolution through consistent lock ordering; the producer–consumer problem with
condition variables; process-based parallelism with `fork()` and shared memory
via `mmap()`; phase synchronisation with barriers; a map-reduce decomposition;
and a comparison of CPU-bound against I/O-bound workloads, where the useful
degree of concurrency is not bounded by the core count.

Examples are written primarily in C++20. Several have a Java counterpart kept
alongside the C++ version so that the two threading models can be compared
directly.

## Requirements

A C++20 compiler is required; g++ 11 or newer is recommended, as Example 8 uses
`std::barrier`. Example 10 links against libcurl. The Java versions require a
JDK 11 or newer, and the plotting script in Example 2 requires Python 3 with
matplotlib. Both are optional.

On Debian or Ubuntu:

    sudo apt install build-essential libcurl4-openssl-dev default-jdk python3-matplotlib

## Building and running

Each lab is built from its own directory:

    cd Lab-1
    make              # build all C++ examples into bin/
    make java         # compile the Java examples into bin/java/
    make run-ex1      # build and run a single example
    make help         # list the available targets
    make clean        # remove all build output

Individual examples can also be compiled by hand; the flags used throughout are
`-std=c++20 -O2 -pthread`, with `-lcurl` added for Example 10.

## Labs

**Lab 1** — thread fundamentals, synchronisation and process-based parallelism.
Ten examples covering the topics listed above. Source in [`Lab-1/`](Lab-1/),
report in [`docs/lab-1/`](docs/lab-1/).

## Reports

The report for each lab is kept under [`docs/`](docs/) in Markdown, together
with the measurement tables and figures it refers to. Each report states the
machine and compiler it was measured on, since all timing results are specific
to the hardware that produced them. Reading a report alongside the corresponding
source directory should give a complete picture of what was done and what was
observed.

## Elsewhere

I write about this and related work, including computer graphics and GPU
optimisation, on my site at [borayalcinn.github.io](https://borayalcinn.github.io),
where you can also find my other projects.

Questions, corrections and suggestions are welcome — open an issue or get in
touch. If something here is unclear or wrong, I would rather hear about it than
leave it standing.

## Licence

Released under the [MIT Licence](LICENSE).
