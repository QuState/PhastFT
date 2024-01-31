# PhastFT

PhastFT is a high-performance, "quantum-inspired" Fast Fourier
Transform (FFT) library written in pure and safe Rust. It is the fastest
pure-Rust FFT library according to our benchmarks.

## Features

- Takes advantage of latest CPU features up to and including AVX-512, but performs well even without them.
- Zero `unsafe` code
- Python bindings (via [PyO3](https://github.com/PyO3/pyo3)).
- Simple implementation using a single, general-purpose FFT algorithm and no costly "planning" step
- Optional parallelization of some steps to 2 threads (with even more planned).

## Limitations

 - No runtime CPU feature detection (yet). Right now achieving the highest performance requires compiling with `-C target-cpu=native` or [`cargo multivers`](https://github.com/ronnychevalier/cargo-multivers).
 - Requires nightly Rust compiler due to use of portable SIMD

## How is it so fast?

PhastFT is designed around the capabilities and limitations of modern hardware (that is, anything made in the last 10 years or so).

The two major bottlenecks in FFT are the **CPU cycles** and **memory accesses.**

We picked an FFT algorithm that maps well to modern CPUs. The implementation can make use of latest CPU features such as AVX-512, but performs well even without them.

Our key insight for speeding up memory accesses is that FFT is equivalent to applying gates to all qubits in `[0, n)`.
This creates to oppurtunity to leverage the same memory access patterns as a [high-performance quantum state simulator](https://github.com/QuState/spinoza).

We also use the Cache-Optimal Bit Reveral Algorithm ([COBRA](https://csaws.cs.technion.ac.il/~itai/Courses/Cache/bit.pdf))
on large datasets and optionally run it on 2 parallel threads, accelerating it even further.

All of this combined results in a fast and efficient FFT implementation that surpasses the performance of existing Rust FFT crates,
including [RustFFT](https://crates.io/crates/rustfft/), on both large and small inputs and while using significantly less memory.

## Getting Started

To integrate PhastFT into your Rust project:

...

To use PhastFT with Python:

```bash
pip install ...
```

...

## Examples

### Rust

PhastFT provides a straightforward API for performing FFT computations. Here's an example of using PhastFT for a basic FFT
operation:

...

### Python

...

## Benchmarks

PhastFT is benchmarked against other FFT libraries. Detailed benchmarking results and instructions are available in the
benchmarks directory.

## Contributing

Contributions to PhastFT are welcome! If you find any issues or have improvements to suggest, please open an issue or
submit a pull request. Follow the contribution guidelines outlined in the CONTRIBUTING.md file.

## License

...

## Profiling

Navigate to the cloned repo:

```bash
cd PhastFt
```

On linux, open access to performance monitoring, and observability operations for processes:

```bash
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

Add debug to `Cargo.toml` under `profile.release`:

```bash
[profile.release]
debug = true
```

Finally, run:

```bash
./profile.sh
```

## What's with the name?

The name, **PhastFT**, is derived from the implementation of the
[Quantum Fourier Transform](https://en.wikipedia.org/wiki/Quantum_Fourier_transform) (QFT). Namely, the
[quantum circuit implementation of QFT](https://en.wikipedia.org/wiki/Quantum_Fourier_transform#Circuit_implementation)
consists of the **P**hase gates and **H**adamard gates. Hence, **Ph**astFT.