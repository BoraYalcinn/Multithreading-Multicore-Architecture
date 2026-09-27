# PC40 Lab session 2 - Source and environment guide
PC40 Fall 2026 (A26),
J.Gaber,
gaber@utbm.fr

## Same algorithm, intentionally different practical defaults
The three programs implement the same network and SGD algorithm: 784 -> 64 ReLU -> 10 softmax. C++/Java can process larger pure-language workloads quickly; pure Python is slower. Therefore the handout recommends smaller Python limits for convenience. Use `--limit N` to choose the workload explicitly.

For a fair speedup study within a language, keep **all conditions identical** between T1 and TP: same executable/interpreter build, same `--limit`, same epochs, same batch size, same machine.

## Python free-threaded requirement
Use a free-threaded CPython build for the Python multithreading path. Example with uv:
```bash
uv python install 3.14t
uv run --python 3.14t python -c "import sysconfig; print(sysconfig.get_config_var('Py_GIL_DISABLED'))"
```
The expected value is `1`. Depending on installation, the executable may be named `python3.14t`.

The Python baseline stores numeric arrays in `array.array('d')` rather than large shared lists of Python float objects. This reduces reference-count contention that can otherwise obscure the parallel-programming experiment in free-threaded CPython.

## C++ ThreadSanitizer
A C++ data race is **undefined behavior**, not merely a possible lost update. To detect the deliberately introduced race in Part E:
```bash
g++ -std=c++17 -O1 -g -fsanitize=thread -pthread nn_multithreaded.cpp -o nn_tsan
./nn_tsan --synthetic --limit 500
```
After correcting the race, run ThreadSanitizer again.

## Java timing
The JVM uses JIT compilation. For timing, perform a warm-up run before recording measurements, or explicitly state that the first run is excluded. Use the same procedure for every thread count.

## Persistent-worker primitives (advanced part)
- C++20: `std::barrier`; C++17 alternative: `std::condition_variable`.
- Java: `ExecutorService`, and when an explicit reusable barrier is needed, `CyclicBarrier`.
- Python: `threading.Barrier`; a persistent pool may use `concurrent.futures.ThreadPoolExecutor`.
Do not implement a home-made barrier with unsynchronized boolean flags.

## False-sharing challenge
Per-thread counters stored contiguously may occupy the same cache line. In C++, compare a compact array with padded/aligned records (for example `alignas(64)`). The variables are logically private but may still interfere through cache coherence.
