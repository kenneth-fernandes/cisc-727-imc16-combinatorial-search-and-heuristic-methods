# IMC16 — GPU-Accelerated N-Queens (Combinatorial Search)

**Author:** Kenneth Peter Fernandes  
**Course:** CISC 719 — Contemporary Computing Systems Modeling Algorithms (CCSM), Harrisburg University, Spring 2026  
**Topic:** Combinatorial Search & Heuristic Methods (Quinn Ch. 16)  
**Execution:** Google Colab (GPU runtime — T4)  

---

## Overview

Implements and benchmarks the **N-Queens problem** across four backends — from a serial recursive backtrack to a bitmask-optimized Numba CUDA kernel — following the backtrack search and parallel backtrack methodology described in Quinn Chapter 16 (§16.3, §16.4).

The N-Queens problem (place N non-attacking queens on an N×N chessboard) is the canonical illustration of **backtrack search over an exponential state-space tree**. It exposes the load-imbalance challenges Quinn discusses in §16.4: subtrees prune at vastly different rates, making naive subtree-per-process partitioning suboptimal.

All implementations are in a single Jupyter notebook.

---

## Repository Structure

```
cisc-727-imc16-combinatorial-search-and-heuristic-methods/
├── README.md
├── notebooks/
│   └── imc16_nqueens_main.ipynb     # All implementations + benchmarks
├── results/
│   ├── benchmarks.csv               # Timing data (impl × N)
│   ├── benchmarks.png               # Wall-clock vs N (log-y)
│   └── speedup.png                  # Speedup vs bitmask CPU
└── docs/
    ├── src/
    │   ├── report.tex               # LaTeX source
    │   └── references.bib           # Bibliography
    └── pdf/
        └── report.pdf               # Compiled report (17 pages)
```

---

## How to Run

1. Open `notebooks/imc16_nqueens_main.ipynb` in Google Colab
2. Set runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`)
3. Run all cells top-to-bottom (`Runtime → Run all`)
4. Results are written to `results/` (benchmarks.csv, benchmarks.png, speedup.png)

---

## Implementations

| # | Implementation | Backend | Strategy |
|---|----------------|---------|----------|
| 1 | Naive recursive backtrack | Pure Python | Serial baseline — board as list, conflict check by scan |
| 2 | Bitmask serial backtrack | Numba JIT (CPU) | Single-threaded with `cols`, `diag1`, `diag2` bitmasks |
| 3 | Parallel CPU backtrack | Numba `prange` / multiprocessing | Split state-space tree at row 0; one task per top-level subtree |
| 4 | GPU backtrack (baseline) | Numba CUDA | One thread per 2-row prefix, iterative bitmask DFS |
| 5 | GPU backtrack (optimized) | Numba CUDA | 3-row prefix + block-level shared-memory reduction + int32 masks |

All implementations **count the total number of distinct solutions** for board size N — the standard N-Queens benchmark.

---

## Benchmark Configuration

- **Board sizes:** N = 10, 12, 13, 14, 15, 16
- **Protocol:** 3 warmup runs + 5 timed runs, median reported
- **Timing:** `time.perf_counter` wall-clock; `cuda.synchronize()` before stopping the GPU timer
- **Correctness oracle:** OEIS A000170 (e.g., N=15 → 2,279,184; N=16 → 14,772,512)

---

## Key Results (T4 GPU)

| Implementation | N=14 (s) | N=15 (s) | N=16 (s) | Speedup vs Bitmask CPU @ N=16 |
|----------------|---------:|---------:|---------:|------------------------------:|
| Naive Python   | —        | —        | —        | (only run up to N=12)          |
| Bitmask CPU    | 0.435    | 2.107    | 14.540   | 1× (baseline)                  |
| Parallel CPU   | 0.236    | 1.513    | 11.502   | 1.26×                          |
| GPU baseline   | 0.177    | 0.946    | 5.555    | 2.62×                          |
| GPU optimized  | 0.015    | 0.069    | **0.363**| **40.05×**                     |

**Headline:** GPU optimized solves N=16 in 363 ms — a 40× speedup over the fastest single-threaded native implementation, and a 15.3× improvement over its own GPU baseline. See `docs/pdf/report.pdf` for the full analysis.

---

## References

- Quinn, M. J. (2004). *Parallel Programming in C with MPI and OpenMP*, Ch. 16.
- Somers, J. (2002). The N-Queens problem (bitmask backtrack technique).
- Sloane, N. J. A. OEIS A000170: Number of ways of placing n non-attacking queens on an n×n board.
- NVIDIA Corporation. (2024). CUDA C++ Programming Guide.
- Lam, S. K., Pitrou, A., & Seibert, S. (2015). Numba: A LLVM-based Python JIT compiler. *LLVM-HPC '15*.
