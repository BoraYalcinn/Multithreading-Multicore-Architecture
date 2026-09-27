# PC40 Lab session 2 - From Sequential Neural-Network Training to Explicit Multithreading,
PC40 Fall 2026 (A26),
J.Gaber,
gaber@utbm.fr


## Goal
Start from a working sequential neural-network training program and build a correct and efficient multithreaded version in **one** language: C++, Java, or free-threaded Python.

This is a parallel-programming lab, not a machine-learning lab. The sequential neural-network code is provided, documented, and ready to run.

## Package
- `docs/PC40_Lab_Multithreaded_NN_v2.tex` - complete lab handout (LaTeX)
- `docs/PC40_Lab_Multithreaded_NN_v2.pdf` - compiled handout
- `cpp/nn_sequential.cpp` - formatted C++ reference implementation
- `java/NeuralNetworkSequential.java` - formatted Java reference implementation
- `python/nn_sequential.py` - formatted free-threading-friendly Python reference implementation
- `data/download_mnist.py` - downloads and decompresses MNIST IDX files
- `SOURCE_GUIDE.md` - source-code and environment guide

## 1. Get MNIST
From the package root:
```bash
python3 data/download_mnist.py
```
All programs also support `--synthetic` for smoke tests. Use MNIST for report measurements unless your instructor says otherwise.

## 2. Common size option
All three baselines accept `--limit N` to control the number of training samples. This is useful for making experiments comparable and for adapting runtime to the machine.

## 3. Compile / run
### C++
```bash
cd cpp
g++ -O2 -Wall -Wextra -std=c++17 nn_sequential.cpp -o nn_sequential
./nn_sequential ../data --limit 12000
./nn_sequential --synthetic --limit 2000
```
### Java
```bash
cd java
javac NeuralNetworkSequential.java
java NeuralNetworkSequential ../data --limit 12000
java NeuralNetworkSequential --synthetic --limit 2000
```
### Python free-threaded
Install a free-threaded CPython, for example with `uv python install 3.14t`, or select the free-threaded option in the python.org installer. Then verify it as described in `SOURCE_GUIDE.md`.
```bash
cd python
python3.14t nn_sequential.py ../data --limit 3000
python3.14t nn_sequential.py --synthetic --limit 500
```
**Important:** compute speedup using the sequential baseline and parallel version with the **same interpreter/build**, compiler options, dataset size, and machine.

## 4. Keep the baseline unchanged
Copy the reference source to a new file before parallelizing it, e.g. `nn_multithreaded.cpp`, `NeuralNetworkMultithreaded.java`, or `nn_multithreaded.py`.
