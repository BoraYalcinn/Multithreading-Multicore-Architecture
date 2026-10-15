# Part E – KPI tables (median of 5 independent runs)

- sequential reference, infer images=500 batch=500: median 0.9153 s [0.8805–0.9484]
- sequential reference, infer images=1 batch=1: median 0.5364 s [0.5066–0.5755]
- sequential reference, train images=10000 batch=32: median 0.4292 s [0.3953–0.4708]
- sequential reference, train images=10000 batch=256: median 0.4272 s [0.4207–0.4718]

## Inference B=500 (synthetic, 100 repeats) – v1 image, workers create

| T | conv median (s) | total median (s) | min–max (s) | S_T = t1/tT | E_T | S vs sequential | throughput (img/s) | extra memory (MB) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6947 | 1.1433 | 1.0881–1.1672 | 1.00 | 1.00 | 0.80 | 43,733 | 0.11 |
| 2 | 0.3834 | 0.6432 | 0.6085–0.6848 | 1.78 | 0.89 | 1.42 | 77,736 | 0.21 |
| 4 | 0.2317 | 0.4106 | 0.3956–0.4507 | 2.78 | 0.70 | 2.23 | 121,762 | 0.42 |
| 8 | 0.2625 | 0.4459 | 0.3701–0.6187 | 2.56 | 0.32 | 2.05 | 112,142 | 0.83 |
| 16 | 0.2261 | 0.4869 | 0.3819–0.6511 | 2.35 | 0.15 | 1.88 | 102,696 | 1.65 |

## Inference B=500 (synthetic, 100 repeats) – v1 image, workers pool

| T | conv median (s) | total median (s) | min–max (s) | S_T = t1/tT | E_T | S vs sequential | throughput (img/s) | extra memory (MB) |
|---|---|---|---|---|---|---|---|---|
| 1 | – | 0.9101 | 0.8813–0.9825 | 1.00 | 1.00 | 1.01 | 54,940 | 0.11 |
| 2 | – | 0.6436 | 0.5352–0.7049 | 1.41 | 0.71 | 1.42 | 77,690 | 0.21 |
| 4 | – | 0.4246 | 0.3570–0.4551 | 2.14 | 0.54 | 2.16 | 117,772 | 0.42 |
| 8 | – | 0.4190 | 0.3596–0.4641 | 2.17 | 0.27 | 2.18 | 119,318 | 0.83 |
| 16 | – | 0.3205 | 0.3101–0.3695 | 2.84 | 0.18 | 2.86 | 156,023 | 1.65 |

## Inference B=500 (synthetic, 100 repeats) – v2 rows, workers create

| T | conv median (s) | total median (s) | min–max (s) | S_T = t1/tT | E_T | S vs sequential | throughput (img/s) | extra memory (MB) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.7288 | 1.2568 | 1.2103–1.3003 | 1.00 | 1.00 | 0.73 | 39,784 | 51.46 |
| 2 | 0.3986 | 0.7207 | 0.7111–0.7937 | 1.74 | 0.87 | 1.27 | 69,377 | 51.46 |
| 4 | 0.2352 | 0.4609 | 0.4510–0.4780 | 2.73 | 0.68 | 1.99 | 108,479 | 51.46 |
| 8 | 0.2580 | 0.4863 | 0.4637–0.4996 | 2.58 | 0.32 | 1.88 | 102,817 | 51.46 |
| 16 | 0.3239 | 0.6011 | 0.5431–0.6950 | 2.09 | 0.13 | 1.52 | 83,183 | 51.46 |

## Inference B=500 (synthetic, 100 repeats) – v2 rows, workers pool

| T | conv median (s) | total median (s) | min–max (s) | S_T = t1/tT | E_T | S vs sequential | throughput (img/s) | extra memory (MB) |
|---|---|---|---|---|---|---|---|---|
| 1 | – | 1.0884 | 1.0710–1.2519 | 1.00 | 1.00 | 0.84 | 45,940 | 51.46 |
| 2 | – | 0.7813 | 0.7068–0.8170 | 1.39 | 0.70 | 1.17 | 63,997 | 51.46 |
| 4 | – | 0.6655 | 0.5839–0.7027 | 1.64 | 0.41 | 1.38 | 75,130 | 51.46 |
| 8 | – | 0.5920 | 0.4352–0.6274 | 1.84 | 0.23 | 1.55 | 84,459 | 51.46 |
| 16 | – | 0.4850 | 0.4530–0.5098 | 2.24 | 0.14 | 1.89 | 103,103 | 51.46 |

## Inference B=1 < T (one image, 30000 repeats) – v1 image, workers create

| T | conv median (s) | total median (s) | min–max (s) | S_T = t1/tT | E_T | S vs sequential | throughput (img/s) | extra memory (MB) |
|---|---|---|---|---|---|---|---|---|
| 1 | – | 2.9444 | 2.8998–3.1903 | 1.00 | 1.00 | 0.18 | 10,189 | 0.10 |
| 2 | – | 3.2806 | 3.1562–3.6054 | 0.90 | 0.45 | 0.16 | 9,145 | 0.21 |
| 4 | – | 4.5231 | 4.2746–4.8297 | 0.65 | 0.16 | 0.12 | 6,633 | 0.41 |
| 8 | – | 6.5190 | 6.3473–6.9265 | 0.45 | 0.06 | 0.08 | 4,602 | 0.82 |
| 16 | – | 11.5993 | 11.3346–11.9453 | 0.25 | 0.02 | 0.05 | 2,586 | 1.65 |

## Inference B=1 < T (one image, 30000 repeats) – v1 image, workers pool

| T | conv median (s) | total median (s) | min–max (s) | S_T = t1/tT | E_T | S vs sequential | throughput (img/s) | extra memory (MB) |
|---|---|---|---|---|---|---|---|---|
| 1 | – | 0.5296 | 0.5135–0.5388 | 1.00 | 1.00 | 1.01 | 56,642 | 0.10 |
| 2 | – | 1.2575 | 1.2058–1.3355 | 0.42 | 0.21 | 0.43 | 23,857 | 0.21 |
| 4 | – | 1.9110 | 1.8203–2.0647 | 0.28 | 0.07 | 0.28 | 15,699 | 0.41 |
| 8 | – | 2.8316 | 2.6451–2.8917 | 0.19 | 0.02 | 0.19 | 10,595 | 0.82 |
| 16 | – | 5.1105 | 5.0199–5.2830 | 0.10 | 0.01 | 0.10 | 5,870 | 1.65 |

## Inference B=1 < T (one image, 30000 repeats) – v2 rows, workers create

| T | conv median (s) | total median (s) | min–max (s) | S_T = t1/tT | E_T | S vs sequential | throughput (img/s) | extra memory (MB) |
|---|---|---|---|---|---|---|---|---|
| 1 | – | 2.9999 | 2.9289–3.0821 | 1.00 | 1.00 | 0.18 | 10,000 | 0.10 |
| 2 | – | 4.5905 | 4.3631–4.6692 | 0.65 | 0.33 | 0.12 | 6,535 | 0.10 |
| 4 | – | 7.6009 | 7.2130–7.7821 | 0.39 | 0.10 | 0.07 | 3,947 | 0.10 |
| 8 | – | 12.3413 | 12.1825–12.4585 | 0.24 | 0.03 | 0.04 | 2,431 | 0.10 |
| 16 | – | 19.4411 | 19.2531–19.5182 | 0.15 | 0.01 | 0.03 | 1,543 | 0.10 |

## Inference B=1 < T (one image, 30000 repeats) – v2 rows, workers pool

| T | conv median (s) | total median (s) | min–max (s) | S_T = t1/tT | E_T | S vs sequential | throughput (img/s) | extra memory (MB) |
|---|---|---|---|---|---|---|---|---|
| 1 | – | 0.5546 | 0.5509–0.5930 | 1.00 | 1.00 | 0.97 | 54,096 | 0.10 |
| 2 | – | 1.1715 | 1.0246–1.4393 | 0.47 | 0.24 | 0.46 | 25,608 | 0.10 |
| 4 | – | 2.0542 | 1.9671–2.2962 | 0.27 | 0.07 | 0.26 | 14,604 | 0.10 |
| 8 | – | 3.9681 | 3.8737–4.2090 | 0.14 | 0.02 | 0.14 | 7,560 | 0.10 |
| 16 | – | 7.6919 | 7.6219–7.7991 | 0.07 | 0.00 | 0.07 | 3,900 | 0.10 |

## Training, MNIST 10000 images, batch 32, image ownership (create per batch)

| T | conv median (s) | total median (s) | min–max (s) | S_T = t1/tT | E_T | S vs sequential | throughput (img/s) | extra memory (MB) |
|---|---|---|---|---|---|---|---|---|
| 1 | – | 0.6175 | 0.5941–0.6313 | 1.00 | 1.00 | 0.70 | 16,195 | 0.21 |
| 2 | – | 0.3895 | 0.3808–0.4090 | 1.59 | 0.79 | 1.10 | 25,676 | 0.42 |
| 4 | – | 0.3074 | 0.2894–0.3170 | 2.01 | 0.50 | 1.40 | 32,527 | 0.85 |
| 8 | – | 0.3751 | 0.3468–0.3836 | 1.65 | 0.21 | 1.14 | 26,661 | 1.69 |
| 16 | – | 0.4435 | 0.4359–0.4730 | 1.39 | 0.09 | 0.97 | 22,548 | 3.39 |

## Training, MNIST 10000 images, batch 256, image ownership (create per batch)

| T | conv median (s) | total median (s) | min–max (s) | S_T = t1/tT | E_T | S vs sequential | throughput (img/s) | extra memory (MB) |
|---|---|---|---|---|---|---|---|---|
| 1 | – | 0.5271 | 0.5242–0.5412 | 1.00 | 1.00 | 0.81 | 18,970 | 0.21 |
| 2 | – | 0.3015 | 0.2866–0.3290 | 1.75 | 0.87 | 1.42 | 33,162 | 0.42 |
| 4 | – | 0.1927 | 0.1800–0.1986 | 2.74 | 0.68 | 2.22 | 51,901 | 0.85 |
| 8 | – | 0.1810 | 0.1711–0.1982 | 2.91 | 0.36 | 2.36 | 55,259 | 1.69 |
| 16 | – | 0.1677 | 0.1608–0.1734 | 3.14 | 0.20 | 2.55 | 59,625 | 3.39 |

## Part F – Amdahl fraction measured from the T=1 training phases

- batch 32: T=1 total 0.6175 s = parallel region 0.6013 + reduction 0.0068 + update 0.0081 + other 0.0014 → **φ = 0.974**
  - T=1: measured S=1.00, Amdahl model S=1.00; parallel region 0.6013 s, reduction 0.0068 s
  - T=2: measured S=1.59, Amdahl model S=1.95; parallel region 0.3749 s, reduction 0.0092 s
  - T=4: measured S=2.01, Amdahl model S=3.71; parallel region 0.2809 s, reduction 0.0193 s
  - T=8: measured S=1.65, Amdahl model S=6.76; parallel region 0.3214 s, reduction 0.0427 s
  - T=16: measured S=1.39, Amdahl model S=11.48; parallel region 0.3396 s, reduction 0.0867 s
- batch 256: T=1 total 0.5271 s = parallel region 0.5245 + reduction 0.0009 + update 0.0008 + other 0.0009 → **φ = 0.995**
  - T=1: measured S=1.00, Amdahl model S=1.00; parallel region 0.5245 s, reduction 0.0009 s
  - T=2: measured S=1.75, Amdahl model S=1.99; parallel region 0.2986 s, reduction 0.0013 s
  - T=4: measured S=2.74, Amdahl model S=3.94; parallel region 0.1879 s, reduction 0.0028 s
  - T=8: measured S=2.91, Amdahl model S=7.73; parallel region 0.1742 s, reduction 0.0051 s
  - T=16: measured S=3.14, Amdahl model S=14.88; parallel region 0.1506 s, reduction 0.0096 s
