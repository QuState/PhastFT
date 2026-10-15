# Bit-Reversal Kernel Benchmarks

A micro-benchmark of 5 [bit-reversal permutation](https://en.wikipedia.org/wiki/Bit-reversal_permutation) algorithms.

<p align="center">
  <img src="https://raw.githubusercontent.com/smu160/PhastFT/main/assets/criterion_overlay_kernel_bit_reversal_f32_10_16.svg" width="480" title="Bit Reversal (f32), small-N" alt="Bit Reversal (f32), 2^10–2^16: COBRAVO, BRAVO, COBRA, Elaan, Naive BR runtime relative to COBRA">
  <img src="https://raw.githubusercontent.com/smu160/PhastFT/main/assets/criterion_overlay_kernel_bit_reversal_f32_17_24.svg" width="480" title="Bit Reversal (f32), large-N" alt="Bit Reversal (f32), 2^17–2^24: COBRAVO, BRAVO, COBRA, Elaan, Naive BR runtime relative to COBRA">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/smu160/PhastFT/main/assets/criterion_overlay_kernel_bit_reversal_f64_10_16.svg" width="480" title="Bit Reversal (f64), small-N" alt="Bit Reversal (f64), 2^10–2^16: COBRAVO, BRAVO, COBRA, Elaan, Naive BR runtime relative to COBRA">
  <img src="https://raw.githubusercontent.com/smu160/PhastFT/main/assets/criterion_overlay_kernel_bit_reversal_f64_17_24.svg" width="480" title="Bit Reversal (f64), large-N" alt="Bit Reversal (f64), 2^17–2^24: COBRAVO, BRAVO, COBRA, Elaan, Naive BR runtime relative to COBRA">
</p>

Benchmarking platform:

| Component | Specification |
| :-- | :-- |
| **OS** | macOS Tahoe 26.4.1 |
| **Kernel** | Darwin 25.4.0|
| **Machine** | MacBook Air (M2, 2022) |
| **CPU** | Apple M2; 8 cores (4P + 4E) |
| **Memory** | 24 GB unified |
| **L1 cache (I/D, per core)** | P-cores 192/128 KiB; E-cores 128/64 KiB |
| **L2 cache (per 4-core cluster)** | P 16 MiB  E 4 MiB |
| **Cache line size** | 128 B |

### Steps to Reproduce

From the repo root, run the following:
```sh
cargo bench --bench bit_reversal --features bench-internals
```

### Render the plots
```sh
uv run benches/plot_criterion_overlay.py --groups kernel_bit_reversal_f32,kernel_bit_reversal_f64
```
