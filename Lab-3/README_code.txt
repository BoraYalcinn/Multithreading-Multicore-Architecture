PC40 Lab 03 - Sequential CNN reference implementations (v7) - J.Gaber (gaber@utbm.fr)

Choose ONE language for the lab and keep it for both the sequential baseline and your multithreaded version.

C++17
  g++ -O2 -std=c++17 -Wall -Wextra -pedantic cnn_sequential.cpp -o cnn_seq
  ./cnn_seq --self-test
  ./cnn_seq --gradcheck

Java
  javac CNNSequential.java
  java CNNSequential --self-test
  java CNNSequential --gradcheck

Python
  python3 cnn_sequential.py --self-test
  python3 cnn_sequential.py --gradcheck
  The Python reference uses pure Python loops (no NumPy): it is about 100x slower than C++/Java.
  Use the reduced workloads below. For a real multithreaded speedup, use a free-threaded build
  (python3.13t / python3.14t) and record whether the GIL is enabled at run time.

Common main options
  --synthetic
  --data DIR
  --mode train|infer
  --train N --test N --epochs N --batch N --filters N
  --seed N --lr X --repeats N --output-csv FILE

Checks
  --self-test  1) hand-computed 2x2 convolution on a 4x4 input
               2) forward() against a naive direct implementation (S, P, z, sum of probabilities)
               3) pooling tie rule: zero image -> every 2x2 window is a tie -> winner = top-left element
  --gradcheck  central finite differences (eps = 1e-6, threshold 1e-5) on one synthetic image, 2 filters,
               for K[0], K[7], b[0], b[1], W[0], W[169], W[2000], c[3]

Common smoke test
  --synthetic --mode train --train 2000 --test 500 --epochs 3 --batch 32 --filters 8 --seed 42 --lr 0.05
  expected: epoch 1 train_loss=1.191873, epoch 3 train_loss=0.006749

Common inference benchmark
  --synthetic --mode infer --test 500 --filters 8 --seed 42 --repeats 10
  expected: checksum=500.023634460

Reduced workloads for the Python track
  --synthetic --mode infer --test 100 --filters 8 --seed 42 --repeats 2          (checksum=19.995653523)
  --synthetic --mode train --train 320 --test 100 --epochs 1 --batch 32 --seed 42 (train_loss=2.279992)

Implementation notes
  - One-image workspace (S, A, P, winners, z, p): no batch dimension. si(f,r,q)=(26f+r)*26+q, pi(f,r,q)=(13f+r)*13+q.
  - Samples are processed in stored order (no shuffling).
  - Pooling tie rule: first maximum in row-major order (strict '>'). ReLU'(0)=0.
  - backward_accumulate() allocates its temporaries (dz, dP, dA) at every call.

The three programs use the same 32-bit LCG, architecture, formulas and data layout.

Small floating-point differences across languages/platforms are possible.

Performance comparisons required by the lab are within one chosen language, not across languages.



