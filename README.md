[![Build](https://github.com/QuState/PhastFT/actions/workflows/rust.yml/badge.svg)](https://github.com/QuState/PhastFT/actions/workflows/rust.yml)
[![codecov](https://codecov.io/gh/QuState/PhastFT/graph/badge.svg?token=IM86XMURHN)](https://codecov.io/gh/QuState/PhastFT)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)
[![](https://img.shields.io/crates/v/phastft)](https://crates.io/crates/phastft)
[![](https://docs.rs/phastft/badge.svg)](https://docs.rs/phastft/)

# PhastFT

PhastFT is a high-performance, "quantum-inspired" Fast Fourier
Transform (FFT) library written in pure Rust.

## Features

- Simple implementation using the Cooley-Tukey FFT algorithm
- Performance on par with other Rust FFT implementations
- Zero `unsafe` code
- Takes advantage of latest CPU features up to and including `AVX-512`, but performs well even without them
- Selects the fastest implementation at runtime. No need for `-C target-cpu=native`!
- Optional parallelization of some steps to 2 threads (with even more planned)
- 2x lower memory usage than [RustFFT](https://crates.io/crates/rustfft/)
- Python bindings (via [PyO3](https://github.com/PyO3/pyo3))

## Limitations

- Only supports input with a length of `2^n` (i.e., a power of 2) -- input should be padded with zeros to the next power
  of 2
- Requires nightly Rust compiler due to use of portable SIMD

## Planned features

- Bluestein's algorithm (to handle arbitrary sized FFTs)
- More multi-threading
- More work on cache-optimal FFT

## How is it so fast?

PhastFT is designed around the capabilities and limitations of modern hardware (that is, anything made in the last 10
years or so).

The two major bottlenecks in FFT are the **CPU cycles** and **memory accesses**.

We picked an efficient, general-purpose FFT algorithm. Our implementation can make use of latest CPU features such as
`AVX-512`, but performs well even without them.

Our key insight for speeding up memory accesses is that FFT is equivalent to applying gates to all qubits in `[0, n)`.
This creates the opportunity to leverage the same memory access patterns as
a [high-performance quantum state simulator](https://github.com/QuState/spinoza).

We also use the Cache-Optimal Bit Reversal
Algorithm ([COBRA](https://csaws.cs.technion.ac.il/~itai/Courses/Cache/bit.pdf))
on large datasets and optionally run it on 2 parallel threads, accelerating it even further.

All of this combined results in a fast and efficient FFT implementation competitive with
the performance of existing Rust FFT crates,
including [RustFFT](https://crates.io/crates/rustfft/), while using significantly less memory.

## Quickstart

### Rust

```rust
use phastft::{
    planner::Direction,
    fft_64
};

let big_n = 1 << 10;
let mut reals: Vec<f64> = (1..=big_n).map(|i| i as f64).collect();
let mut imags: Vec<f64> = (1..=big_n).map(|i| i as f64).collect();
fft_64(&mut reals, &mut imags, Direction::Forward);
```

### Python

Follow the instructions at <https://rustup.rs/> to install Rust, then switch to the nightly channel with

```bash
rustup default nightly
```

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

### Output Order

`phastft` always finishes processing input data by running
a [bit-reversal permutation](https://en.wikipedia.org/wiki/Bit-reversal_permutation) on the processed data.

## Benchmarks

PhastFT is benchmarked against several other FFT libraries. Scripts and instructions to reproduce benchmark results and
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

Contributions to PhastFT are welcome! If you find any issues or have improvements to suggest, please open an issue or
submit a pull request. Follow the contribution guidelines outlined in the CONTRIBUTING.md file.

## License

PhastFT is licensed under MIT or Apache 2.0 license, at your option.

## PhastFT vs. RustFFT

[RustFFT](https://crates.io/crates/rustfft/) is another excellent FFT implementation in pure Rust.
RustFFT and PhastFT make different trade-offs.

RustFFT made the choice to work on stable Rust compiler at the cost of `unsafe` code,
while PhastFT contains no `unsafe` blocks but requires a nightly build of Rust compiler
to access the Portable SIMD API.

RustFFT implements multiple FFT algorithms and tries to pick the best one depending on the workload,
while PhastFT has a single FFT implementation and still achieves competitive performance.

PhastFT uses 2x less memory than RustFFT, which is important for processing large datasets.

## What's with the name?

The name, **PhastFT**, is derived from the implementation of the
[Quantum Fourier Transform](https://en.wikipedia.org/wiki/Quantum_Fourier_transform) (QFT). Namely, the
[quantum circuit implementation of QFT](https://en.wikipedia.org/wiki/Quantum_Fourier_transform#Circuit_implementation)
consists of the **P**hase gates and **H**adamard gates. Hence, **Ph**astFT.
