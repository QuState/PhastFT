# PHFT

**PH**ast**FT** (PHFT) is a high-performance, "quantum-inspired" Fast Fourier Transform (FFT) library written in pure
and
safe Rust.

What's with the name? Great question!

The name, **PHFT**, is derived from the implementation of the
[Quantum Fourier Transform](https://en.wikipedia.org/wiki/Quantum_Fourier_transform) (QFT). Namely, the
[quantum circuit implementation of QFT](https://en.wikipedia.org/wiki/Quantum_Fourier_transform#Circuit_implementation)
consists of the **P**hase gates and **H**adamard gates. Hence, **PH**ast**FT**.

In general, the FFT is equivalent to applying gates to all qubits in `[0, n)`. This approach creates to oppurtunity to
leverage the same memory access patterns as high-performance quantum state simulator. This results in a fast and
efficient FFT implementation that surpasses the performance of existing Rust FFT crates, including RustFFT.

## Features

- Performance ...
- Python bindings (via PyO3) ...
- Safety ...

## Getting Started

To integrate PHFT into your Rust project:

...

To use PHFT with Python:

```bash
pip install ...
```

...

## Examples

### Rust

PHFT provides a straightforward API for performing FFT computations. Here's an example of using PHFT for a basic FFT
operation:

...

### Python

...

## Benchmarks

PHFT is benchmarked against other FFT libraries. Detailed benchmarking results and instructions are available in the
benchmarks directory.

## Contributing

Contributions to PHFT are welcome! If you find any issues or have improvements to suggest, please open an issue or
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
