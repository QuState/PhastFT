[![Build](https://github.com/QuState/PhastFT/actions/workflows/rust.yml/badge.svg)](https://github.com/QuState/PhastFT/actions/workflows/rust.yml)
[![codecov](https://codecov.io/gh/QuState/PhastFT/graph/badge.svg?token=IM86XMURHN)](https://codecov.io/gh/QuState/PhastFT)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)
[![](https://img.shields.io/crates/v/phastft)](https://crates.io/crates/phastft)
[![](https://docs.rs/phastft/badge.svg)](https://docs.rs/phastft/)

# PhastFT

PhastFT is a high-performance, "quantum-inspired" Fast Fourier
Transform (FFT) library written in safe Rust.

Designed for large FFTs (gigabytes of data) common in scientific workloads, e.g. in quantum computer simulators.

## Features

- Performance competitive with other Rust implementations in a single thread, and much faster when multi-threaded
- In-place algorithm for low memory usage, allows running large FFTs on cheaper hardware
- Simple implementation using the Cooley-Tukey FFT algorithm and [CO-BRAVO](https://dl.acm.org/doi/abs/10.1145/1248377.1248411) bit reversal
- No `unsafe` code
- Optional multi-threading
- SIMD acceleration on SSE4.2, AVX2, NEON and WASM thanks to [`fearless_simd`](https://crates.io/crates/fearless_simd)
- Selects the fastest SIMD implementation at runtime. No need for `-C target-cpu=native`!
- Coming soon: Python bindings (via [PyO3](https://github.com/PyO3/pyo3))

## Limitations

- Only supports input with a length of `2^n` (i.e., a power of 2) -- input should be padded with zeros to the next power of 2

## Planned features

- Additional algorithms for non-power-of-2 FFTs
- Even more work on performance

## Quickstart

### Rust

```rust
use phastft::{fft_64_dit, fft_64_dit_with_planner, planner::{Direction, PlannerDit64}};

let big_n = 1 << 20;
let mut reals: Vec<f64> = (1..=big_n).map(|i| i as f64).collect();
let mut imags: Vec<f64> = (1..=big_n).map(|i| i as f64).collect();

// Simple API
fft_64_dit(&mut reals, &mut imags, Direction::Forward);

// Or with a reusable planner for better performance with multiple FFTs
let planner = PlannerDit64::new(big_n);
fft_64_dit_with_planner(&mut reals, &mut imags, Direction::Forward, &planner);
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

#### Real-Valued FFT (R2C)

For purely real-valued input signals, the R2C transform is approximately
2x faster than running a full complex FFT with zeroed imaginary components:

```rust
use phastft::{r2c_fft_f64, c2r_ifft_f64};

let n = 1 << 16;
let signal: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
let mut spec_re = vec![0.0; n];
let mut spec_im = vec![0.0; n];

r2c_fft_f64(&signal, &mut spec_re, &mut spec_im);

// Recover original signal
let mut recovered = vec![0.0; n];
c2r_ifft_f64(&spec_re, &spec_im, &mut recovered);
```

For repeated FFTs of the same size, use a planner to avoid
re-computing twiddle factors:

```rust
use phastft::planner::{Direction, PlannerR2c64};
use phastft::r2c_fft_f64_with_planner;

let n = 1 << 16;
let signal: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
let mut spec_re = vec![0.0; n];
let mut spec_im = vec![0.0; n];

let planner = PlannerR2c64::new(n, Direction::Forward);
// reuse `planner` across calls
r2c_fft_f64_with_planner(&signal, &mut spec_re, &mut spec_im, &planner);
```

### Python (coming soon)

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

- Input: Normal order (bit-reversed internally)
- Output: Normal order
- Bit Reversal: Always performed on input (required for correctness)

## Performance Notes

**Reuse planners.** If you're doing multiple FFTs of the same size, create the planner once and reuse it.

**Threading.** If the `parallel` feature is enabled, the library uses 2 threads for bit reversal and all available threads for FFT calculation when it seems profitable.

**Complex numbers.** The separate real/imaginary array API is faster and uses less memory than the interleaved complex API which allocates temporary buffers.

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

 - Up to 2x lower memory usage, letting you use laptops or cheaper cloud instances for large FFTs
 - Multi-threading support, much higher performance on large sizes when using multi-threading
 - No `unsafe` code

### RustFFT advantages

 - Higher performance for small sizes thanks to dedicated handwritten kernels for each size
 - Supports FFT sizes that aren't powers of 2 (with a large performance penalty)

## What's with the name?

The name, **PhastFT**, is derived from the implementation of the
[Quantum Fourier Transform](https://en.wikipedia.org/wiki/Quantum_Fourier_transform) (QFT). Namely, the
[quantum circuit implementation of QFT](https://en.wikipedia.org/wiki/Quantum_Fourier_transform#Circuit_implementation)
consists of the **P**hase gates and **H**adamard gates. Hence, **Ph**astFT.
