# Benchmarks and Profiling

## Run benchmarks

### Setup Environment

1. Install [FFTW3](http://www.fftw.org/download.html)[^1]

   It may be possible to install `fftw3` using a package manager.

   ##### debian
   ```bash
   sudo apt install libfftw3-dev
   ```

2. Clone the `PhastFT` git repository [^2].

3. Create virtual env

```bash
cd ~/PhastFT/benches && python -m venv .env && source .env/bin/activate
```

4. Install python dependencies[^1]

```bash
pip install -r requirements.txt
cd ~/PhastFT/pyphastft
RUSTFLAGS='-Ctarget-cpu=native' pip install .
```

5. Run the `FFTW3` vs. `RustFFT` vs. `PhastFT` benchmark for all inputs of size `2^n`, where `n \in [4, 30].`

```bash
./benchmark.sh 4 30
```

6. Plot the results

```bash
python benchmark_plots.py
```

The generated images will be saved in your working directory.

7. Run the python benchmarks and plot the results

```bash
python py_benchmarks.py
```

The generated images will be saved in your working directory.

## Benchmark Configuration

### Libraries and Packages

| Library/Package | Version        | Language  | Benchmark Compilation Flags                                                                          |
|-----------------|----------------|-----------|------------------------------------------------------------------------------------------------------|
| `FFTW3`         | 3.3.10-1 amd64 | C, OCaml  | `-O3 -march=native`                                                                                  |
| `RustFFT`       | 6.2.0          | Rust      | `-C target-cpu=native -C opt-level=3 --edition=2021; codegen-units = 1; lto = true; panic = "abort"` |
| `PhastFT`       | 0.1.0          | Rust      | `-C target-cpu=native -C opt-level=3 --edition=2021; codegen-units = 1; lto = true; panic = "abort"` |
| `NumPy`         | 1.26.4         | Python, C | `N/A`                                                                                                |
| `pyFFTW`        | 0.13.1         | Python, C | `N/A`                                                                                                |

### Benchmark System Configuration

|                      |                                                                                                 |
|----------------------|-------------------------------------------------------------------------------------------------|
| **CPU**              | AMD Ryzen 9 7950X (SMT disabled)                                                                |
| L1d Cache            | 512 KiB (16 instances)                                                                          |
| L1i Cache            | 512 KiB (16 instances)                                                                          |
| L2 Cache             | 16 MiB (16 instances)                                                                           |
| L3 Cache             | 64 MiB (2 instances)                                                                            |
|                      |                                                                                                 |
| **Memory**           |                                                                                                 |
| /0/f/0               | 64GiB System Memory                                                                             |
| /0/f/1               | 32GiB DIMM Synchronous Unbuffered (Unregistered) 6000 MHz (0.2 ns)                              |
| /0/f/3               | 32GiB DIMM Synchronous Unbuffered (Unregistered) 6000 MHz (0.2 ns)                              |
|                      |                                                                                                 |
| **OS**               | Linux 7950x 6.1.0-17-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.1.69-1 (2023-12-30) x86_64 GNU/Linux |
|                      |
| **Rust**             |                                                                                                 |
| Installed Toolchains | stable-x86_64-unknown-linux-gnu                                                                 |
|                      | nightly-x86_64-unknown-linux-gnu (default)                                                      |
| Active Toolchain     | nightly-x86_64-unknown-linux-gnu (default)                                                      |
| Rustc Version        | 1.77.0-nightly (5bd5d214e 2024-01-25)                                                           |

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

[^1]: Those with macOS on Apple Silicon should consult [pyFFTW Issue #352](https://github.com/pyFFTW/pyFFTW/issues/352#issuecomment-1945444558)

[^2]: This tutorial assumes you will clone `PhastFT` to `$HOME`
