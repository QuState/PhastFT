[![Build](https://github.com/QuState/PhastFT/actions/workflows/rust.yml/badge.svg)](https://github.com/QuState/PhastFT/actions/workflows/rust.yml)
[![codecov](https://codecov.io/gh/QuState/PhastFT/graph/badge.svg?token=IM86XMURHN)](https://codecov.io/gh/QuState/PhastFT)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)
[![](https://img.shields.io/crates/v/phastft)](https://crates.io/crates/phastft)
[![](https://docs.rs/phastft/badge.svg)](https://docs.rs/phastft/)

# PhastFT

PhastFT is a high-performance, "quantum-inspired" Fast Fourier
Transform (FFT) library written in safe Rust.

Designed for large FFTs common in scientific workloads, e.g. in quantum computer simulators.

## Features

- Simple implementation using the Cooley-Tukey FFT algorithm and [CO-BRAVO](https://dl.acm.org/doi/abs/10.1145/1248377.1248411) bit reversal
- Optional multi-threading, speeding up FFTs on large arrays
- Performance competitive with other Rust FFT implementations, outperforming them if multi-threading is enabled
- Zero `unsafe` code
- SIMD acceleration on SSE4.2, AVX2, NEON and WASM thanks to [`fearless_simd`](https://crates.io/crates/fearless_simd)
- Selects the fastest SIMD implementation at runtime. No need for `-C target-cpu=native`!
- Up to 5x lower memory usage than [RustFFT](https://crates.io/crates/rustfft/)
- Python bindings (via [PyO3](https://github.com/PyO3/pyo3))

## Limitations

- Only supports input with a length of `2^n` (i.e., a power of 2) -- input should be padded with zeros to the next power of 2

## Planned features

- Real-to-complex FFT
- Even more work on performance

## How is it so fast?

PhastFT is designed around the capabilities and limitations of modern hardware
(that is, anything made in the last 10 years or so).

The two major bottlenecks in FFT are the **CPU cycles** and **memory accesses**.

Most literature on FFT focuses on reducing the amount of math operations,
but today's CPUs are heavily memory-bottlenecked for any amount of data that doesn't fit into the cache.
It doesn't matter how much or how little CPU instructions you need to execute
if the CPU spends most of the time just waiting on memory anyway!

[Notes on FFTs for implementers](https://fgiesen.wordpress.com/2023/03/19/notes-on-ffts-for-implementers/) is a good read
if you want to understand the trade-offs on modern hardware. Its author is not affiliated with PhastFT.

The trade-offs we chose are:

- **In-place** FFT with a separate bit-reversal step reduces memory traffic and peak memory usage compared to out-of-place and auto-sorter FFTs
- **Radix-2** Cooley-Turkey FFT: radix-4 and split-radix do less math, but require complex and slow bit reversals.
  - We still need to experiment with fusing multiple radix-2 passes to reduce memory traffic in single-threaded scenarios
- [**CO-BRAVO**](https://dl.acm.org/doi/abs/10.1145/1248377.1248411) cache-optimal, SIMD-accelerated bit reversal trounces other algorithms.
- **Decimation in time** maps better to SIMD fused multiply-adds than decimation-in-frequency, and CO-BRAVO makes skipping bit reversal less appealing.
- **Recursive formulation** enables cache-oblivious FFT and easy parallelism. We switch over to a loop when reaching L1 cache size.

All of this combined results in a fast and efficient FFT implementation competitive with
the performance of existing Rust FFT crates on medium to large sizes, while using significantly less memory.

## Quickstart

### Rust

```rust
use phastft::{planner::Direction, fft_64};

// Using the default DIF algorithm
let big_n = 1 << 10;
let mut reals: Vec<f64> = (1..=big_n).map(|i| i as f64).collect();
let mut imags: Vec<f64> = (1..=big_n).map(|i| i as f64).collect();
fft_64(&mut reals, &mut imags, Direction::Forward);
```

### Using DIT Algorithm

```rust
use phastft::{fft_64_dit, fft_64_dit_with_planner, planner::{Direction, PlannerDit64}};

// Using DIT algorithm - may have better cache performance for some sizes
let big_n = 1 << 20;
let mut reals: Vec<f64> = (1..=big_n).map(|i| i as f64).collect();
let mut imags: Vec<f64> = (1..=big_n).map(|i| i as f64).collect();

// Simple API
fft_64_dit(&mut reals, &mut imags, Direction::Forward);

// Or with a reusable planner for better performance with multiple FFTs
let planner = PlannerDit64::new(big_n, Direction::Forward);
fft_64_dit_with_planner(&mut reals, &mut imags, &planner);
```

#### Complex Number Support (Interleaved Format)

When the `complex-nums` feature is enabled, you can also use the interleaved
format with the `num_complex::Complex` type:

```rust
use phastft::{
    planner::Direction,
    fft_64_interleaved
};
use num_complex::Complex;

let big_n = 1 << 10;
let mut signal: Vec<Complex<f64>> = (1..=big_n)
    .map(|i| Complex::new(i as f64, i as f64))
    .collect();
fft_64_interleaved(&mut signal, Direction::Forward);
```

Both `fft_32_interleaved` and `fft_64_interleaved` are available for `f32` and
`f64` precision respectively.

### Python

Follow the instructions at <https://rustup.rs/> to install Rust.

Then you can install PhastFT itself:
```bash
pip install numpy
pip install git+https://github.com/QuState/PhastFT#subdirectory=pyphastft
```

```python
import numpy as np
from pyphastft import fft

sig_re = np.asarray(sig_re, dtype=np.float64)
sig_im = np.asarray(sig_im, dtype=np.float64)

fft(a_re, a_im)
```

### Normalization

`phastft` only scales the output of the inverse FFT. Namely, running IFFT(x)
will scale each element by `1/N`, where `N` is the number of data points, and
`IFFT(FFT(x)) == x`. If your use case(s) require(s) something different, please
don't hesitate to create an issue.

### Bit Reversal and Output Order

PhastFT provides two FFT algorithms with different bit reversal behaviors:

#### DIF (Decimation-in-Frequency) - Default Algorithm

- Input: Normal order
- Output: Bit-reversed order (by default)
- Bit Reversal: Can be disabled using `Options::dif_perform_bit_reversal = false`

The ability to skip bit reversal in the DIF FFT is useful in cases such as:

- Bit-reversed output is not required, potentially leading to significantly better run-times.
- You need the output in decimated order for specific algorithms
- Simulating the QFT

```rust
use phastft::{fft_64_with_opts_and_plan, options::Options, planner::{Direction, Planner64}};

let size = 1024;
let mut reals = vec![0.0f64; size];
let mut imags = vec![0.0f64; size];

// Skip bit reversal for DIF FFT
let mut opts = Options::default();
opts.dif_perform_bit_reversal = false;  // Output stays in decimated order
let planner = Planner64::new(size, Direction::Forward);
fft_64_with_opts_and_plan(&mut reals, &mut imags, &opts, &planner);
```

#### DIT (Decimation-in-Time)  

- Input: Normal order (bit-reversed internally)
- Output: Normal order
- Bit Reversal: Always performed on input (required for correctness)

## Performance Notes

**Use DIT for speed.** The DIT algorithm is faster than DIF in most cases due to better memory access patterns. The performance gap is largest for FFTs that fit in L1/L2 cache (up to ~2^17 elements). For huge transforms that blow past cache, both algorithms perform similarly.

**Reuse planners.** If you're doing multiple FFTs of the same size, create the planner once and reuse it.

**Threading.** The library automatically uses 2 threads for bit reversal on inputs ≥ 2^22 elements. This threshold was determined empirically.

**Complex numbers.** The separate real/imaginary array API is faster than the interleaved complex API, which currently allocates temporary buffers.

## Benchmarks

PhastFT is benchmarked against several other FFT libraries. Scripts and
instructions to reproduce benchmark results and
plots are available [here](https://github.com/QuState/PhastFT/tree/main/benches#readme).

<p align="center">
  <img src="https://raw.githubusercontent.com/QuState/PhastFT/main/assets/benchmarks_bar_plot_4_12.png" width="400" title="PhastFT vs. RustFFT vs. FFTW3" alt="PhastFT vs. RustFFT vs. FFTW3">
  <img src="https://raw.githubusercontent.com/QuState/PhastFT/main/assets/benchmarks_bar_plot_13_29.png" width="400" title="PhastFT vs. RustFFT vs. FFTW3" alt="PhastFT vs. RustFFT vs. FFTW3">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/QuState/PhastFT/main/assets/py_benchmarks_bar_plot_0_8.png" width="400" title="PhastFT vs. NumPy FFT vs. pyFFTW" alt="PhastFT vs. NumPy FFT vs. pyFFTW">
  <img src="https://raw.githubusercontent.com/QuState/PhastFT/main/assets/py_benchmarks_bar_plot_9_28.png" width="400" title="PhastFT vs. NumPy FFT vs. pyFFTW" alt="PhastFT vs. NumPy FFT vs. pyFFTW">
</p>

## Contributing

Contributions to PhastFT are welcome! If you find any issues or have
improvements to suggest, please open an issue or submit a pull request. Follow
the contribution guidelines outlined in the CONTRIBUTING.md file.

## License

PhastFT is licensed under MIT or Apache 2.0 license, at your option.

## PhastFT vs. RustFFT

[RustFFT](https://crates.io/crates/rustfft/) is another excellent FFT
implementation in pure Rust. RustFFT and PhastFT make different trade-offs.

### PhastFT advantages

 - Up to 5x lower memory usage, letting you use laptops or much cheaper cloud instances for large FFTs
 - Multi-threading support, much higher performance on large sizes when using multi-threading
 - No `unsafe` code

### RustFFT advantages

 - Higher performance for very small sizes thanks to dedicated handwritten kernels for each size
 - Supports FFT sizes that aren't powers of 2 (with a large performance penalty)

## What's with the name?

The name, **PhastFT**, is derived from the implementation of the
[Quantum Fourier Transform](https://en.wikipedia.org/wiki/Quantum_Fourier_transform) (QFT). Namely, the
[quantum circuit implementation of QFT](https://en.wikipedia.org/wiki/Quantum_Fourier_transform#Circuit_implementation)
consists of the **P**hase gates and **H**adamard gates. Hence, **Ph**astFT.
