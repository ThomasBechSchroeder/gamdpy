## PRELIMINARY benchmarks.

### The Lennard-Jones benchmark (single precision), comparing to Lammps. 

TPS = Timesteps Per Second. Integers next to Lammps results give the number of CPU threads found to be optimal. Gamdpy uses only a single CPU thread, since the whole simulation is done on the GPU. Gamdpy results are optimized by the builtin autotuner (see below).


![Fig](./Data/benchmark_LJ_compare_tps.png)

### The Lennard-Jones benchmark (single precision).

Here followed by parameters chosen by the gamdpy autotuner.

![Fig](./Data/benchmark_LJ_tps.png)

RTX_2060_Super_AT:
|        N  |   TPS   |  MATS |  pb | tp | skin | gridsync |  nblist      |  NIII  |
| --------: | ------: | ----: | --: | --:| ---: | :------: | :----------: | :----: |
|       512 | 63894.5 |  32.7 |   8 | 16 | 0.50 |   True   | N squared    | False  |
|      1024 | 45073.0 |  46.2 |  32 | 11 | 0.70 |   True   | N squared    | True   |
|      2048 | 29870.5 |  61.2 |  32 |  6 | 0.70 |   True   | N squared    | True   |
|      4096 | 15226.4 |  62.4 | 128 |  3 | 0.70 |   True   | N squared    | True   |
|      8192 |  7679.6 |  62.9 | 256 |  1 | 0.50 |   True   | linked lists | True   |
|     16384 |  2399.1 |  39.3 | 128 |  5 | 1.10 |   False  | N squared    | True   |
|     32768 |  1926.1 |  63.1 |  32 |  7 | 0.70 |   False  | linked lists | True   |
|     65536 |  1302.1 |  85.3 |  64 |  5 | 0.50 |   False  | linked lists | True   |
|    131072 |   741.0 |  97.1 | 128 |  3 | 0.50 |   False  | linked lists | True   |
|    262144 |   416.9 | 109.3 | 128 |  3 | 0.50 |   False  | linked lists | True   |
|    524288 |   224.5 | 117.7 | 128 |  4 | 0.50 |   False  | linked lists | True   |
|   1048576 |   118.5 | 124.2 | 128 |  4 | 0.50 |   False  | linked lists | True   |

RTX_2080_Ti_AT:
|        N  |   TPS   |  MATS |  pb | tp | skin | gridsync |  nblist      |  NIII  |
| --------: | ------: | ----: | --: | --:| ---: | :------: | :----------: | :----: |
|       512 | 70729.7 |  36.2 |   8 | 18 | 0.50 |   True   | N squared    | False  |
|      1024 | 56465.4 |  57.8 |  16 | 13 | 0.90 |   True   | N squared    | True   |
|      2048 | 37493.6 |  76.8 |  16 |  9 | 1.10 |   True   | N squared    | True   |
|      4096 | 22429.2 |  91.9 |  32 |  6 | 0.90 |   True   | N squared    | True   |
|      8192 | 12030.0 |  98.5 |  64 |  3 | 0.70 |   True   | linked lists | True   |
|     16384 |  6505.3 | 106.6 | 128 |  1 | 0.30 |   True   | linked lists | True   |
|     32768 |  2302.5 |  75.4 | 128 |  4 | 1.10 |   False  | linked lists | True   |
|     65536 |  1812.7 | 118.8 | 128 |  4 | 0.50 |   False  | linked lists | True   |
|    131072 |  1143.0 | 149.8 | 128 |  4 | 0.50 |   False  | linked lists | True   |
|    262144 |   658.4 | 172.6 | 128 |  4 | 0.50 |   False  | linked lists | True   |
|    524288 |   379.4 | 198.9 | 128 |  4 | 0.50 |   False  | linked lists | True   |
|   1048576 |   195.0 | 204.4 | 128 |  4 | 0.50 |   False  | linked lists | True   |

RTX_4070_AT:
|        N  |   TPS   |  MATS |  pb | tp | skin | gridsync |  nblist      |  NIII  |
| --------: | ------: | ----: | --: | --:| ---: | :------: | :----------: | :----: |
|       512 | 85560.4 |  43.8 |   8 | 18 | 0.30 |   True   | N squared    | False  |
|      1024 | 63033.0 |  64.5 |  16 | 15 | 0.50 |   True   | N squared    | False  |
|      2048 | 42450.2 |  86.9 |  32 |  8 | 0.70 |   True   | N squared    | False  |
|      4096 | 25580.1 | 104.8 |  32 |  5 | 0.90 |   True   | N squared    | False  |
|      8192 | 13267.3 | 108.7 | 128 |  2 | 0.90 |   True   | linked lists | True   |
|     16384 |  7237.8 | 118.6 | 256 |  1 | 0.50 |   True   | linked lists | True   |
|     32768 |  3536.3 | 115.9 | 256 |  4 | 0.90 |   False  | linked lists | True   |
|     65536 |  2523.5 | 165.4 | 256 |  4 | 0.50 |   False  | linked lists | False  |
|    131072 |  1471.4 | 192.9 | 512 |  2 | 0.30 |   False  | linked lists | False  |
|    262144 |   795.4 | 208.5 | 512 |  2 | 0.30 |   False  | linked lists | False  |
|    524288 |   444.4 | 233.0 | 512 |  2 | 0.30 |   False  | linked lists | False  |
|   1048576 |   228.4 | 239.4 | 512 |  2 | 0.30 |   False  | linked lists | False  |

RTX_4090_AT:
|        N  |   TPS   |  MATS |  pb | tp | skin | gridsync |  nblist      |  NIII  |
| --------: | ------: | ----: | --: | --:| ---: | :------: | :----------: | :----: |
|       512 | 94405.6 |  48.3 |  16 | 19 | 1.10 |   True   | N squared    | True   |
|      1024 | 77428.4 |  79.3 |  16 | 17 | 0.50 |   True   | N squared    | False  |
|      2048 | 57104.9 | 117.0 |  16 | 16 | 0.90 |   True   | N squared    | False  |
|      4096 | 37137.6 | 152.1 |  32 | 11 | 0.90 |   True   | N squared    | False  |
|      8192 | 25993.3 | 212.9 |  64 |  7 | 1.10 |   True   | N squared    | True   |
|     16384 | 14937.9 | 244.7 | 128 |  4 | 1.10 |   True   | linked lists | True   |
|     32768 |  8982.8 | 294.3 | 256 |  2 | 0.70 |   True   | linked lists | True   |
|     65536 |  5460.3 | 357.8 | 512 |  1 | 0.50 |   True   | linked lists | True   |
|    131072 |  2754.4 | 361.0 | 256 |  3 | 0.50 |   False  | linked lists | False  |
|    262144 |  1482.0 | 388.5 | 256 |  4 | 0.50 |   False  | linked lists | False  |
|    524288 |   843.3 | 442.1 | 512 |  2 | 0.30 |   False  | linked lists | False  |
|   1048576 |   460.2 | 482.6 | 512 |  2 | 0.30 |   False  | linked lists | False  |
