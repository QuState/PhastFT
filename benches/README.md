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
| `fftw_estimate` | (none)             | Same sizes through FFTW with `FFTW_ESTIMATE`.                                                                                                                                                             |
| `fftw_measure`  | (none)             | Same sizes through FFTW with `FFTW_MEASURE`.                                                                                                                                                              |
| `fftw_conserve` | (none)             | Same sizes through FFTW with `FFTW_MEASURE \| FFTW_CONSERVE_MEMORY` — the apples-to-apples comparison for PhastFT's low-memory design.                                                                    |
| `realfft`       | (none)             | PhastFT R2C/C2R (`r2c_fft_*`, `c2r_fft_*`) vs. the `realfft` crate — forward + inverse, f32 + f64.                                                                                                        |
| `planner`       | (none)             | Planner construction cost (`PlannerDit{32,64}::new` vs. RustFFT's `FftPlanner::plan_fft_forward`).                                                                                                        |
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

## Four-pass prototype at 2 GiB

`examples/benchmark_four_pass.rs` compares the experimental implementation in
`examples/support/four_pass.rs` with the existing DIT FFT. Run from the repo root:

```bash
cargo build --release --example benchmark_four_pass
RAYON_NUM_THREADS=16 target/release/examples/benchmark_four_pass 64 27 10
RAYON_NUM_THREADS=16 target/release/examples/benchmark_four_pass 32 28 10
```

Each input contains exactly 2,147,483,648 bytes across the real and imaginary
arrays: 2^27 complex f64 values or 2^28 complex f32 values. The optional last
argument is `both` (default), `dit`, or `four-pass`. CSV on stdout contains
`algorithm,iteration,nanoseconds`; setup information and sampled correctness
checks and the selected matrix dimensions go to stderr. Set `FOUR_PASS_PROFILE=1`
to also print the four individual pass times in milliseconds (including warm-up).
Leave it unset for final timing runs. `RAYON_NUM_THREADS` controls both algorithms; building with
`--no-default-features` also supports serial execution.

The benchmark excludes planning, allocation, page-touching, and input generation.
It warms up each implementation once, alternates their order, and regenerates
the same seeded input before each transform. The `both` mode compares 4096
distributed output bins during warm-up. For full forward/inverse comparisons
against RustFFT at sizes 2^0 through 2^22, run:

```bash
cargo test --release --example benchmark_four_pass
cargo test --release --no-default-features --example benchmark_four_pass
```

The four sweeps are column gather/FFT/twiddle multiplication, transpose, row
FFTs, and a final transpose to natural output order. Small FFTs perform multiple
cache passes within each sweep. The prototype destroys its input and returns
output in caller-provided scratch, with no final copy back. It needs another
2 GiB for scratch and O(sqrt(N)) planning storage. DIT instead needs about
2 GiB for its full-size twiddle plan. `both` retains both plans and scratch,
so allow roughly 6 GiB of RAM. The optimized matrix split uses 2048 x 65536 for
f64 and 4096 x 65536 for f32 at 2 GiB, keeping the first-pass batches at 512 KiB.
Transposes use packed stack tiles of 32 x 128 elements (32 KiB per worker for
f64); the initial gather uses 32 x 16 tiles. No library API or default algorithm
is changed.

Original results are in [four_pass_results.txt](four_pass_results.txt); optimization
experiments and final raw samples are in
[four_pass_optimized_results.txt](four_pass_optimized_results.txt).

The optional SIMD transpose experiment preserves that matrix split and batching:

```bash
# Default: the existing packed scalar transpose.
FOUR_PASS_TRANSPOSE=scalar RAYON_NUM_THREADS=16 target/release/examples/benchmark_four_pass 64 27 10 four-pass
# Packed tiles, then an 8x8 f64 / 16x16 f32 register transpose.
FOUR_PASS_TRANSPOSE=simd RAYON_NUM_THREADS=16 target/release/examples/benchmark_four_pass 64 27 10 four-pass
# Same register transpose, loading directly from the source instead of packing.
FOUR_PASS_TRANSPOSE=direct RAYON_NUM_THREADS=16 target/release/examples/benchmark_four_pass 64 27 10 four-pass
```

For f32, substitute `32 28` for `64 27`. The benchmark reports the selected
transpose layout and detected SIMD backend. The experiment uses normal
`fearless_simd` runtime dispatch and safe Rust; it does not require native CPU
build flags. The scalar path remains the default because the SIMD variants did
not consistently improve whole-FFT timings on the Ryzen 9950X. Measurements are
in [four_pass_simd_results.txt](four_pass_simd_results.txt), with named-kernel
AVX-512 instruction evidence in
[four_pass_simd_disassembly.txt](four_pass_simd_disassembly.txt).

The same comparison at **64 MiB of input** uses `64 22` or `32 23` as the
precision/exponent arguments. It retains the current matrix split and tiling;
scratch adds another 64 MiB. See
[four_pass_64mib_results.txt](four_pass_64mib_results.txt) for 120 samples per
variant, pass timings and correctness checks. SIMD did not improve total FFT
time at this size either.

With **one Rayon worker pinned to one CPU core**, direct SIMD transposes do
improve total FFT time: 11.0%/10.8% less time at 64 MiB and 9.0%/4.3% less
at 2 GiB, for f64/f32 respectively. Packed SIMD helps only f32 at 64 MiB
(3.0%) in this comparison. See
[four_pass_single_thread_results.txt](four_pass_single_thread_results.txt)
for timings, run order, reproduction commands and sampled correctness checks.
These measurements use the same executable and layout as the 16-thread runs;
the scalar default is unchanged.

An experiment force-inlined the original scalar formulation into `dispatch!`,
with no explicit vector operations. LLVM emitted AVX-512 gathers from the packed
tile and 512-bit stores. It took 6.2–9.6% longer than scalar on one thread,
and showed no improvement with 16 threads. The wrapper, forced inlining and
`auto` mode have been removed; the measurements remain as historical records.
See [four_pass_auto_results.txt](four_pass_auto_results.txt) and
[four_pass_auto_disassembly.txt](four_pass_auto_disassembly.txt).

## Profiling

From the repo root:

```bash
# Linux: open access to performance monitoring
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid

./profile.sh
```
