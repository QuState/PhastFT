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
| `realfft`       | (none)             | PhastFT R2C/C2R (`r2c_fft_*`, `c2r_fft_*_with_planner_and_scratch`) vs. the `realfft` crate — forward + inverse, f32 + f64.                                                                               |
| `planner`       | (none)             | Planner construction cost (`PlannerDit{32,64}::new` vs. RustFFT's `FftPlanner::plan_fft_forward`).                                                                                                        |
| `planner_mode`  | (none)             | `PlannerMode::Heuristic` vs. `PlannerMode::Tune`. The two modes currently produce identical planners (the `_mode` argument is ignored by `with_mode`); the bench pins the API surface for the eventual `Tune` implementation. |
| `interleave`    | `complex-nums`     | Internal SIMD interleave / deinterleave kernels.                                                                                                                                                          |
| `bit_reversal`  | `bench-internals`  | Internal CO-BRAVO vs. BRAVO bit-reversal kernels.                                                                                                                                                         |

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
and emits `criterion_overlay_<group>_<log2_lo>_<log2_hi>.{png,svg}` —
two files per group (small-N and large-N halves split at the size
list's midpoint). The `GROUPS` registry in the script holds both the
humanized title and per-group default baseline (`RustFFT` for the
`c2c_*` groups, `realfft` for `r2c_*` / `c2r_*`, `Heuristic` for
`planner_mode_*`, `BRAVO` for `kernel_bit_reversal_*`). Groups on disk
that aren't in the registry get a loud warning and are skipped — better
to surface the gap than to silently normalize against a guessed
baseline. PEP 723 inline metadata means `uv run` fetches matplotlib /
numpy / pandas on demand — no venv required.

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

### Optional: pyFFTW comparison

The `requirements.txt` venv is **only** needed for the pyFFTW-based
comparisons in the project README — neither `benchmark_plots.py` nor
`plot_criterion_overlay.py` require it. If you need pyFFTW:

```bash
cd benches && python3 -m venv .env && source .env/bin/activate
pip install -r requirements.txt
cd ../pyphastft && pip install .
```

macOS on Apple Silicon: see [pyFFTW Issue #352][pyfftw-352] for the
install workaround.

[pyfftw-352]: https://github.com/pyFFTW/pyFFTW/issues/352#issuecomment-1945444558

## Reference benchmark configuration

Absolute timings will vary by hardware; the numbers in the project
README come from the reference machine below.

### Libraries

| Library / package | Version          | Language  | Compilation flags                                                                |
|-------------------|------------------|-----------|----------------------------------------------------------------------------------|
| `FFTW3`           | 3.3.10-1 amd64   | C         | `-O3`                                                                            |
| `RustFFT`         | 6.2.0            | Rust      | `-C opt-level=3 --edition=2021; codegen-units = 1; lto = true; panic = "abort"`  |
| `realfft`         | 3.5.0            | Rust      | release profile; same `Cargo.toml` settings as RustFFT                           |
| `PhastFT`         | 0.1.0            | Rust      | `-C opt-level=3 --edition=2021; codegen-units = 1; lto = true; panic = "abort"`  |
| `NumPy`           | 1.26.4           | Python, C | N/A                                                                              |
| `pyFFTW`          | 0.13.1           | Python, C | N/A                                                                              |

### Reference machine

|                           |                                                                                                 |
|---------------------------|-------------------------------------------------------------------------------------------------|
| **CPU**                   | AMD Ryzen 9 7950X (SMT off)                                                                     |
| L1d / L1i cache           | 512 KiB / 512 KiB (16 instances each)                                                           |
| L2 / L3 cache             | 16 MiB (16 instances) / 64 MiB (2 instances)                                                    |
| **Memory**                | 64 GiB (2 × 32 GiB DIMM, 6000 MHz)                                                              |
| **OS**                    | Linux 7950x 6.1.0-17-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.1.69-1 (2023-12-30) x86_64 GNU/Linux |
| CPU freq scaling governor | Performance                                                                                     |
| **Rust**                  | nightly (rustc 1.79.0-nightly, 7f2fc33da, 2024-04-22)                                           |

## Profiling

From the repo root:

```bash
# Linux: open access to performance monitoring
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid

./profile.sh
```
