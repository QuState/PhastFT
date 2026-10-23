# Benchmarks and Profiling

This directory hosts two distinct benchmarking workflows. They serve
different purposes; don't conflate them.

1. **Criterion bench targets** — declared as `[[bench]]` entries in the
   root `Cargo.toml` and run with `cargo bench`. Cover small-to-mid sizes
   for PhastFT internals and cross-library comparisons against
   [RustFFT](https://crates.io/crates/rustfft),
   [FFTW3](https://www.fftw.org/), and the
   [`realfft`](https://crates.io/crates/realfft) crate.
2. **Shell-driven cross-library pipeline** — `benchmark.sh` plus
   `benchmark_plots.py`, drives the example binaries in `../examples/`
   for big-N comparisons against RustFFT and FFTW3.

The whole `benches/` directory is excluded from the published crate
(`exclude = ["benches"]` in the root `Cargo.toml`), so nothing here ships
on crates.io.

## 1. Criterion benches

Always run from the **repo root**, not from inside this directory:

```bash
cargo bench --bench <name>                 # one bench target
cargo bench --all-features                 # every target
```

| Bench target    | Required features  | Coverage                                                                                                                                                                                                  |
|-----------------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `bench`         | (none)             | PhastFT complex FFT (`fft_{32,64}_dit_with_planner_and_opts`) — forward + inverse, f32 + f64.                                                                                                             |
| `rustfft`       | (none)             | Same sizes routed through RustFFT.                                                                                                                                                                        |
| `small`         | (none)             | PhastFT vs. RustFFT at powers of two from 1 through 1024; both inverse transforms include normalization. |
| `fftw_estimate` | (none)             | Same sizes through FFTW with `FFTW_ESTIMATE`.                                                                                                                                                             |
| `fftw_measure`  | (none)             | Same sizes through FFTW with `FFTW_MEASURE`.                                                                                                                                                              |
| `fftw_conserve` | (none)             | Same sizes through FFTW with `FFTW_MEASURE \| FFTW_CONSERVE_MEMORY` — the apples-to-apples comparison for PhastFT's low-memory design.                                                                    |
| `realfft`       | (none)             | PhastFT R2C/C2R (`r2c_fft_*`, `c2r_fft_*`) vs. the `realfft` crate — forward + inverse, f32 + f64.                                                                                                        |
| `planner`       | (none)             | Planner construction cost (`PlannerDit{32,64}::new` vs. RustFFT's `FftPlanner::plan_fft_forward`).                                                                                                        |
| `twiddle_modes` | (none)             | Explicit `Precomputed` vs. `LowMemory` plans: complex and real transforms, both directions and precisions, serial and default threading. |
| `interleave`    | `complex-nums`     | Internal SIMD interleave / deinterleave kernels.                                                                                                                                                          |
| `bit_reversal`  | `bench-internals`  | Five bit-reversal kernels head-to-head — CO-BRAVO, BRAVO, COBRA, Elaan, Naive ([`BIT_REVERSAL.md`](BIT_REVERSAL.md)).                                                                                     |

### Why the complex comparison is split across five binaries

PhastFT vs. RustFFT vs. FFTW (three planning modes) is split across
**five** separate `[[bench]]` binaries (`bench`, `rustfft`,
`fftw_estimate`, `fftw_measure`, `fftw_conserve`) so FFTW's per-process
wisdom cache cannot leak between planning modes — every run starts with
a fresh process and empty wisdom. All five write into the same shared
criterion group folders: `c2c_forward_f32`, `c2c_inverse_f32`,
`c2c_forward_f64`, `c2c_inverse_f64`.

The three FFTW binaries each call `fftw_lib::run_all(c, id, flags)` to
emit all four C2C groups — the shared body lives once in
`benches/fftw_lib/mod.rs` so only the per-mode `Flag` set and series ID
differ between the three.

The R2C/C2R cross-library comparison lives in **one** binary
(`realfft`) because the `realfft` crate has no per-process planner cache
to isolate. Its groups (`r2c_f{32,64}`, `c2r_f{32,64}`) are distinct
from the C2C groups, so no cross-binary aggregation is needed. R2C is
forward by definition and C2R is inverse by definition — direction is
implicit in the prefix.

### Transform timing and repeatability

Benchmarks allocate fresh inputs for every transform.
The shared generators use a fixed seed: values repeat for a given precision
and length on the same target with the locked `rand` version, and
split/interleaved layouts contain the same values. `SmallRng` does not promise
identical streams across platforms or dependency upgrades. Fresh allocation
does **not** imply cold-cache execution.

Transform groups collect 20 samples using Criterion's `Flat` sampling mode.
Every sample performs the same number of iterations. Batch size is independent
of that count: `clamp(1 MiB / input_bytes, 1, 256)`. The budget counts both split
arrays or the corresponding interleaved array; real inputs and half-spectra use
their actual lengths. It excludes plans, scratch, output buffers, and temporary
setup allocations. Inputs larger than 1 MiB use a batch of one. Criterion may
finish a sample with a partial batch, identical across samples in flat mode.

`iter_batched_ref` keeps input generation, allocation, and destruction outside
timing. Planning and reusable scratch/output allocation remain outside timing;
allocations performed internally by an FFT API remain part of its execution.
The timed region measures the FFT and output black-boxing, plus Criterion's
batch traversal overhead. Planner and internal-kernel timing policies are
unchanged (their shared input generators are also seeded).

### Twiddle storage modes

`twiddle_modes` compares the two explicit modes with identical seeded inputs.
Planning, input generation, allocation, and destruction are outside the timed
region. C2R scratch is allocated once and reused. The `serial` groups disable
both FFT and bit-reversal parallelism; `parallel` groups use the default options
and Rayon pool and are omitted when the `parallel` feature is disabled.
`Auto` is resolved during planning and runs exactly one of these two paths.

A short sweep around the crossover:

```bash
cargo bench --bench twiddle_modes -- \
  'twiddles_f(32|64)_forward_(serial|parallel)/.*/(1048576|2097152|4194304|16777216)$' \
  --warm-up-time 0.1 --measurement-time 0.3 --noplot
```

Use `twiddles_real_.*` to select R2C/C2R, or `inverse` in place of `forward` for
complex inverse transforms. The full sweep covers `2^6` through `2^24`. Large
transforms automatically extend measurement time to collect the 20 samples;
repeat runs and use longer warm-up/measurement windows for tuning a specific
machine. This benchmark uses the standard fresh-allocation policy, which does
not guarantee cold caches.

The initial `Auto` cutoff is a **32 MiB complex input footprint**: `2^22` points
for f32 or `2^21` for f64. Real plans use the half-length complex FFT's footprint
and apply the resolved mode to the inner FFT and the real twiddles together.
The cutoff is a fixed heuristic, not runtime cache detection or autotuning.

Representative forward medians from a short sweep on an AMD Ryzen 9 9950X,
16-core Rayon pool, Rust 1.99.1, release profile (milliseconds):

| Precision / complex points | Serial stored | Serial low memory | Default threading stored | Default threading low memory |
| --- | ---: | ---: | ---: | ---: |
| f32 / `2^22` | 9.70 | 9.22 | 4.40 | 3.21 |
| f32 / `2^24` | 48.66 | 47.09 | 29.63 | 24.70 |
| f64 / `2^21` | 9.35 | 8.58 | 3.70 | 3.28 |
| f64 / `2^24` | 99.86 | 99.16 | 69.11 | 59.49 |

Below the cutoff, compact generation can cost more than it saves in memory
traffic. Above it, benefits vary with threading, precision, and cache behavior;
an explicit mode remains useful for workload-specific tuning. Retune the named
`AUTO_LOW_MEMORY_BYTES` constant in `src/planner.rs` when broader hardware
measurements justify a change.

### Small fixed kernels

`small` compares both libraries in one binary at powers of two from 1 to 1024.
It reuses plans and scratch, with the same seeded inputs and batch sizing as
the other transform benchmarks. Unlike the separate `rustfft` target, it
includes `1/N` scaling in RustFFT's inverse timing to match PhastFT. It uses
Criterion's default sampling mode; pass `--sample-size` to control samples.
On Linux, pin both comparison runs to the same available CPU (CPU 8 below).

```bash
taskset -c 8 cargo bench --bench small -- --warm-up-time 0.1 --measurement-time 0.3 --sample-size 30 --noplot

# Limit fearless_simd's runtime dispatch to AVX2 on an AVX-512 machine.
CARGO_TARGET_DIR=target/avx2 RUSTFLAGS='--cfg disable_dispatch_avx512' \
  taskset -c 8 cargo bench --bench small -- --warm-up-time 0.1 --measurement-time 0.3 --sample-size 30 --noplot
```

The new path covers 1–512 points. The 1024-point case exercises the existing
DIT path as a control. Kernels operate on split real/imaginary vectors, without
converting to interleaved complex numbers or running a bit-reversal pass.
The decompositions are 4×4, 4×8, 8×8, 2×64, 4×64, and 8×64 for sizes 16
through 512. The planner stores connecting twiddles in contiguous rows shared
by the AVX2 and AVX-512 kernels. Sizes 32–512 now do this extra work during
planning; execution timings exclude it.

AVX-512 uses sixteen f32 lanes in the outer stages and processes pairs of
64-point f32 leaves in parallel. Each half of a register holds one leaf;
transposes preserve that separation until the outer stage interleaves the
results into frequency order. The f32 transposes use lane-local unpack/shuffle
instructions before combining lanes. Intermediate stack buffers are aligned
to 64 bytes to avoid split cache-line accesses. The four/eight-lane f64 paths
also benefit from that alignment and the revised 32- and 128-point factorizations.

The explicit vector loads are intentional: LLVM outlined some array-building
iterators into baseline-ISA functions, creating temporary stack copies and
SIMD helper calls. Fixed-size twiddle views and explicit loads keep that work
in the caller's SIMD context. Performance changes were evaluated through the
public API on hardware, including input, twiddle, scratch, and output memory
traffic.

Local measurements on an AMD Ryzen 9 9950X with Rust 1.99.1 / LLVM 23.1.1,
default Cargo features, reusable plans, and execution pinned to CPU 8:

| Points | AVX2 f32 speedup | AVX2 f64 speedup | AVX-512 f32 speedup | AVX-512 f64 speedup |
|--------|-----------------|-----------------|--------------------|--------------------|
| 16 | 1.00× | 0.99× | 1.01× | 1.00× |
| 32 | 1.35× | 1.35× | 1.07× | 1.30× |
| 64 | 1.21× | 1.04× | 1.11× | 1.04× |
| 128 | 1.40× | 0.98× | 2.76× | 1.43× |
| 256 | 1.13× | 1.04× | 1.71× | 1.12× |
| 512 | 1.13× | 1.08× | 1.55× | 1.03× |

These are forward-transform ratios against commit `475dafb`. Both versions
used 30 samples, 0.1 s warmup, and 0.3 s measurement. The baseline uses the
faster mean of two separate runs at each size: the old f64 kernels varied
between processes, and choosing the faster run avoids overstating gains.
The AVX2 build used `--cfg disable_dispatch_avx512`. These are local timings,
not portable speed guarantees; differences of a few percent need caution.

Forward execution times in nanoseconds with PhastFT's AVX-512 dispatch and
RustFFT 6.4.1's AVX2 dispatch, measured in the same benchmark binary:

| Points | PhastFT f32 | RustFFT f32 | PhastFT f64 | RustFFT f64 |
|--------|-------------|-------------|-------------|-------------|
| 16 | 6.6 | 4.8 | 7.6 | 6.3 |
| 32 | 11.3 | 8.2 | 12.0 | 16.1 |
| 64 | 16.0 | 20.1 | 20.7 | 36.4 |
| 128 | 25.1 | 43.5 | 67.2 | 75.4 |
| 256 | 71.2 | 97.4 | 149.7 | 175.3 |
| 512 | 164.5 | 213.9 | 314.5 | 421.9 |

The largest normalized inverse gains are also in AVX-512 f32: 128/256/512 points
measure 26.6/72.7/167.2 ns, versus 68.3/118.3/254.3 ns before. The 1024-point
control stays within roughly 2%. RustFFT retains its forward advantage at
16 points for both precisions and at 32 points for f32; this tuning does not
make PhastFT faster at every small size. Input setup and allocation remain
outside timing; actual transforms include all memory accesses. Fresh batched
inputs do not guarantee cold caches.

### Cross-binary overlay plots

Criterion does **not** auto-aggregate across `[[bench]]` binaries — the
per-group `report/{lines,violin}.svg` is regenerated by each
`criterion_main!` using only the IDs registered in that process, so the
last bench to run clobbers the overlay. Use the overlay aggregator
after running the five complex benches:

```bash
uv run benches/plot_criterion_overlay.py                              # all groups, per-group default baselines
uv run benches/plot_criterion_overlay.py --groups c2c_forward_f32     # filter (snake_case, no quoting)
uv run benches/plot_criterion_overlay.py --baseline "PhastFT DIT"     # global baseline override
uv run benches/plot_criterion_overlay.py --out-dir target/overlays    # write elsewhere
```

The script walks `target/criterion/<group>/<id>/<size>/new/{sample,benchmark}.json`
and emits `criterion_overlay_<group>_<log2_lo>_<log2_hi>.svg` —
two files per group (small-N and large-N halves split at the size
list's midpoint). The `GROUPS` registry in the script holds both the
humanized title and per-group default baseline (`RustFFT` for the
`c2c_*` groups, `realfft` for `r2c_*` / `c2r_*`, `Heuristic` for
`planner_mode_*`, `COBRA` for `kernel_bit_reversal_*`). Groups on disk
that aren't in the registry get a loud warning and are skipped — better
to surface the gap than to silently normalize against a guessed
baseline. PEP 723 inline metadata means `uv run` fetches matplotlib /
numpy / pandas on demand — no venv required.

### Single-threaded overlay results

These are the single-threaded cross-library overlays: PhastFT runs on one
thread (without the `parallel` feature) against RustFFT and FFTW3. The forward
transforms' multi-threaded counterparts headline the [root
README](../README.md#benchmarks).

<p align="center">
  <img src="https://raw.githubusercontent.com/smu160/PhastFT/main/assets/criterion_overlay_c2c_forward_f32_6_14.svg" width="400" title="C2C Forward (f32), small-N — single-threaded" alt="C2C Forward (f32), small-N: PhastFT (single-threaded) vs. RustFFT vs. FFTW3">
  <img src="https://raw.githubusercontent.com/smu160/PhastFT/main/assets/criterion_overlay_c2c_forward_f32_15_24.svg" width="400" title="C2C Forward (f32), large-N — single-threaded" alt="C2C Forward (f32), large-N: PhastFT (single-threaded) vs. RustFFT vs. FFTW3">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/smu160/PhastFT/main/assets/criterion_overlay_c2c_forward_f64_6_14.svg" width="400" title="C2C Forward (f64), small-N — single-threaded" alt="C2C Forward (f64), small-N: PhastFT (single-threaded) vs. RustFFT vs. FFTW3">
  <img src="https://raw.githubusercontent.com/smu160/PhastFT/main/assets/criterion_overlay_c2c_forward_f64_15_24.svg" width="400" title="C2C Forward (f64), large-N — single-threaded" alt="C2C Forward (f64), large-N: PhastFT (single-threaded) vs. RustFFT vs. FFTW3">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/smu160/PhastFT/main/assets/criterion_overlay_c2c_inverse_f32_6_14.svg" width="400" title="C2C Inverse (f32), small-N — single-threaded" alt="C2C Inverse (f32), small-N: PhastFT (single-threaded) vs. RustFFT vs. FFTW3">
  <img src="https://raw.githubusercontent.com/smu160/PhastFT/main/assets/criterion_overlay_c2c_inverse_f32_15_24.svg" width="400" title="C2C Inverse (f32), large-N — single-threaded" alt="C2C Inverse (f32), large-N: PhastFT (single-threaded) vs. RustFFT vs. FFTW3">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/smu160/PhastFT/main/assets/criterion_overlay_c2c_inverse_f64_6_14.svg" width="400" title="C2C Inverse (f64), small-N — single-threaded" alt="C2C Inverse (f64), small-N: PhastFT (single-threaded) vs. RustFFT vs. FFTW3">
  <img src="https://raw.githubusercontent.com/smu160/PhastFT/main/assets/criterion_overlay_c2c_inverse_f64_15_24.svg" width="400" title="C2C Inverse (f64), large-N — single-threaded" alt="C2C Inverse (f64), large-N: PhastFT (single-threaded) vs. RustFFT vs. FFTW3">
</p>

### Group naming convention

Group names are **snake_case** with category prefixes:

- `c2c_*` — complex-to-complex FFT (`c2c_forward_f32`, `c2c_inverse_f64`).
- `r2c_*` / `c2r_*` — real-input / real-output FFT. Direction is
  implicit (R2C forward, C2R inverse), so no `forward_` / `inverse_`
  prefix.
- `planner_*` — planner construction.
- `kernel_*` — internal SIMD kernels (`kernel_bit_reversal_f64`,
  `kernel_combine_re_im_f32`), distinct from full-pipeline FFT benches.

snake_case lets names round-trip cleanly through criterion's filename
sanitizer (which rewrites `?"/\*<>:|^` to `_`), survive shell argument
parsing without quoting, and tab-complete. Series IDs (`PhastFT DIT`,
`RustFFT`, `FFTW Estimate`, …) stay PascalCase since they only appear
as `--baseline` arguments. The overlay script humanizes group names at
plot time — chart titles read "C2C Forward (f32)".

Size sweeps (`LENGTHS`, `BIT_REVERSAL_LENGTHS`, `PLANNER_MODE_LENGTHS`),
sample-distribution helpers (`split_complex` / `interleaved_complex` /
`real_signal` / `spectrum_*`), throughput helpers (`throughput_complex` /
`throughput_real`), the per-bench loop runner (`bench_at_sizes`), and
group / ID constants all live in `benches/common/mod.rs` so renames
touch one file. `bench_at_sizes` takes the throughput function as an
argument, so each bench file picks its data layout (split vs.
interleaved) independently of the throughput accounting (paired vs.
single scalar). The shared FFTW C2C bench body lives in
`benches/fftw_lib/mod.rs` and is consumed by all three FFTW bench
binaries.

## 2. Shell-driven cross-library pipeline

`benchmark.sh` builds the example binaries in `../examples/`
(`benchmark`, `rustfft`, `fftwrb`) and drives them through a power-of-2
size sweep, randomizing per-size invocation order so no single library
always runs first. Output is a timestamped `benchmark-data.<ts>/`
directory with one newline-separated ns-per-iter file per
(library, size).

### Run

```bash
./benchmark.sh <n-lower-bound> <n-upper-bound>
```

Each size's iteration count is derived from an N·log2(N) cost model
targeting `BUDGET_NS` (default 2 s) of wall clock. Override with
environment variables:

| Variable      | Default      | Controls                                  |
|---------------|--------------|-------------------------------------------|
| `PRECISION`   | `32`         | `32` or `64` — single or double precision. |
| `BUDGET_NS`   | `2000000000` | Target wall-clock ns per size.            |
| `OVERHEAD_NS` | `200`        | Modeled fixed cost per iteration.         |
| `MIN_ITERS`   | `100`        | Floor on per-size iteration count.        |
| `MAX_ITERS`   | `10000000`   | Cap on per-size iteration count.          |

Output layout:

```
benchmark-data.YYYY.MM.DD.HH-MM-SS/
├── phastft/size_{n}    # newline-separated ns-per-iter floats
├── rustfft/size_{n}
└── fftwrb/size_{n}
```

### Plot

```bash
uv run benches/benchmark_plots.py
```

`benchmark_plots.py` uses PEP 723 inline script metadata, so `uv run`
fetches matplotlib / numpy / pandas on demand — no venv needed. It
auto-discovers the **latest** `benchmark-data.*` directory; to plot an
older run, move newer directories out of the way (there is no CLI
flag).

The plot is a grouped bar chart per size: bars are each library's
median, normalized against RustFFT's median; whiskers are the IQR in
that normalized space; the dashed `y = 1.0` line is RustFFT by
construction. PNGs land in the current working directory.

## Profiling

From the repo root:

```bash
# Linux: open access to performance monitoring
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid

./profile.sh
```
