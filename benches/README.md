# Benchmarks and Profiling

## Run benchmarks

### Setup Environment

1. Install [FFTW3](http://www.fftw.org/download.html)

   It may be possible to install `fftw3` using a package manager.

   ##### debian
   ```bash
   sudo apt install libfftw3-dev
   ```

2. Clone the repo. *Note: This tutorial assumes you will clone `PhastFT` to `$HOME`*

3. Create virtual env

```bash
cd ~/PhastFT/benches && python -m venv .env && source .env/bin/activate
```

4. Install python dependencies

```bash
RUSTFLAGS='-Ctarget-cpu=native' pip install -r requirements.txt
```

5. Run the `FFTW3` vs. `RustFFT` vs. `PhastFT` benchmark for all inputs of size `2^n`, where `n \in [4, 30].`

```bash
./benchmark.sh 4 30
```

6. Plot the results

```bash
python benchmark_plots.py
```

The generated image will show up as `benchmarks_bar_plot.png`.

7. Run the python benchmarks and plot the results

```bash
python py_benchmarks.py
```

The generated image will show up as `py_benchmarks_bar_plot.png`.

## Benchmark Configuration

All benchmark results assume
| |

All benchmark results were produced on the following system configuration:

|                      |                                                                                                 |
|----------------------|-------------------------------------------------------------------------------------------------|
| **CPU**              | AMD Ryzen 9 7950X (SMT enabled)                                                                 |
| L1d Cache            | 512 KiB (16 instances)                                                                          |
| L1i Cache            | 512 KiB (16 instances)                                                                          |
| L2 Cache             | 16 MiB (16 instances)                                                                           |
| L3 Cache             | 64 MiB (2 instances)                                                                            |
|                      |                                                                                                 |
| **Memory**           |                                                                                                 |
| /0/f/0               | memory          64GiB System Memory                                                             |
| /0/f/1               | memory          32GiB DIMM Synchronous Unbuffered (Unregistered) 6000 MHz (0.2 ns)              |
| /0/f/3               | memory          32GiB DIMM Synchronous Unbuffered (Unregistered) 6000 MHz (0.2 ns)              |
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