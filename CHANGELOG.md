# Changelog

All notable changes to this project will be documented in this file.

## [0.4.1] - 2026-07-28

### Bug Fixes

- Make `fft_f32_dit_with_planner_and_opts_impl` inline.

### Changed

- Enable the `parallel` feature by default
  ([#123](https://github.com/smu160/PhastFT/pull/123)).
- Upgrade to `fearless_simd` v0.5.0
  ([#124](https://github.com/smu160/PhastFT/pull/124)).
- Trim `scalar_bit_reversal` trait bounds and derive the bit width from the
  input length.
- Relax `combine_re_im` from `Float` to `Copy`.

### Documentation

- Add bit-reversal benchmark results, plots, and benchmark machine details.

## [0.4.0] - 2026-06-04

The largest release since the initial publish: a SIMD backend with runtime CPU
dispatch, a real-valued FFT, optional multi-threading, and a unified public API.
Treat this as a fresh start — nearly every entry point was renamed.

### Breaking Changes

- The public API was renamed and unified. Every entry point now spells the
  precision as the Rust float type and names the FFT algorithm explicitly:
  - Complex FFT (split real/imag arrays): `fft_64` / `fft_32` are now
    `fft_f64_dit` / `fft_f32_dit`. The full-control `fft_64_with_opts_and_plan`
    / `fft_32_with_opts_and_plan` are now `fft_f64_dit_with_planner_and_opts` /
    `fft_f32_dit_with_planner_and_opts`, and a middle
    `fft_f{64,32}_dit_with_planner` tier was added.
  - Interleaved `Complex` FFT (feature `complex-nums`): `fft_64_interleaved` /
    `fft_32_interleaved` are now `fft_f64_dit_interleaved` /
    `fft_f32_dit_interleaved` (plus `_with_planner` / `_with_planner_and_opts`
    tiers).
  - Planner types `Planner64` / `Planner32` are now `PlannerDit64` /
    `PlannerDit32`, and are direction-agnostic and reusable — one instance
    drives both forward and inverse transforms.
- `Direction::Reverse` was renamed to `Direction::Inverse`.
- The minimum supported Rust version is now declared (`rust-version = "1.88"`).

### Added

- Real-valued FFT — `r2c_fft_f32` / `r2c_fft_f64` and the inverse `c2r_fft_f32`
  / `c2r_fft_f64`, producing/consuming the compact `N/2 + 1` spectrum; roughly
  2x faster than a zero-imaginary complex FFT
  ([#105](https://github.com/smu160/PhastFT/pull/105)).
- Optional multi-threading via the `parallel` feature (Rayon): threaded bit
  reversal and a cache-oblivious parallel recursive FFT.
- `Options::smallest_parallel_chunk_size` to tune the parallel split point.
- Fused multi-stage codelets — FFT-16 for `f64`, FFT-32 for `f32`
  ([#101](https://github.com/smu160/PhastFT/pull/101)).
- `Debug` / `PartialEq` / `Eq` / `Hash` implementations across the public types.

### Changed

- SIMD backend migrated to `fearless_simd` with runtime CPU-feature dispatch.
  `-C target-cpu=native` is no longer needed, and the crate now builds on
  **stable** Rust — the nightly requirement is gone.
- The FFT core is now a recursive, cache-blocked decimation-in-time algorithm;
  CO-BRAVO provides cache-optimal SIMD bit reversal
  ([#106](https://github.com/smu160/PhastFT/pull/106)).

### Removed

- The public `cobra` bit-reversal module, superseded by the internal CO-BRAVO
  implementation.
- The nightly-toolchain requirement.

### Performance

- Cache-blocked recursive DIT, fused first-stage codelets, FMA butterfly
  kernels, and SIMD-accelerated CO-BRAVO bit reversal.

### Fixed

- Inverse FFT output ordering and assorted correctness fixes.

## [0.3.0] - 2025-09-04

### Features

- Add DIT FFT algorithm & bit reversal control

### Bug Fixes

- Bump `bytemuck` to latest version

### Documentation

- Add example usage of interleaved fft

## [0.2.2] - 2025-09-03

### Features

- Add benchmark using criterion
- Add a more robust round-trip FFT test
- Add git hooks and contributing guide
- Add deinterleaving function
- Add criterion group for forward FFT f32
- Add new python benchmarking "framework"
- Add #![feature(doc_cfg)] to make rustdoc actually accept doc_cfg

### Bug Fixes

- Fix formatting
- Fix formatting
- Fix formatting
- Fix formatting
- Fix docsrs config according to Clippy

tbh not sure if Clippy is right in this case
- Avx512 is now stable, so placate clippy
- Update `criterion` benchmarks

### Other

- Vectorize deinterleaving of AoS --> SoA

Use bytemuck + SIMD::deinterleave to rearrange input data from a slice
of Complex values into 2 slices of f32 or f64 values
- Put macro definition begind feature flag
- Account for different signal sizes
- Forgot to add benchmark file
- Don't gate feature documentation on docs.rs, we're on nightly anyway
- Undo duplicate docs.rs all-features

### Refactor

- Make sure benchmark runs
- Make planner reusable

- Planner should be re-usable so it can be re-used for FFT's of the same
  size

- Add regression tests to make sure `fft_64`/`fft_32` gives the same
  results as `fft_64_with_opts_and_plan`/`fft_32_with_opts_and_plan`
- Move planner completely outside of bench function
- Avoid cloning twiddles
- Make examples output time elapsed in nanoseconds
- Use new de-interleaving function
- Make required cfg show up on docs.rs
- Make docs.rs build docs with all features enabled

### Documentation

- Update the normalization section

### Miscellaneous Tasks

- Remove index tracker from hot path
- Updates to benchmarks without planners
- Remove `array_chunks`
- Update pre-commit hook to run clippy everywhere
- Update dependencies and their usage

### Revert

- Revert "Make sure benchmark runs"

This reverts commit 7011dfc040b866cd773185667915a9920b9c5a80.
- Revert "Forgot to add benchmark file"

This reverts commit b70dd4b318f59af32740c4bd4c9e75a69bcd690b.

## [0.2.1] - 2024-05-03

### Features

- Add a function to separate AoS to SoA
- Add reverse separate fn and add test
- Add fft_*_interleaved impls and tests

### Bug Fixes

- Fix formatting
- Fix formatting
- Fix formatting
- Fixes inverse FFT ouput order issue

### Other

- Advertise runtime feature selection
- Transition `num-complex` dependency to optional

- Make num-complex optional and non-default, and all functions that take
  and test interleaved FFT are now under the same feature

- Bump num-complex version 0.4.5 --> 0.4.6

- Enable num-complex for docs/docs.rs

- Update github action workflow to run tests for all features

- Fix formatting in docs to fix links
- Placate clippy's complaints about assign op

### Refactor

- Simplify separate_re_im
- Move twiddles fwd * rev = identity test

- Testing that all the values in the forward twiddle factors pointwise
multiplied by all the values in the inverse twiddle factors is more of a
unit test for twiddles. It made no sense to keep it in the planner.

- The previous commit modifies the planner in a way that this test would
  no longer pass. We can get the same, good coverage by keeping this
  test under twiddles.

### Documentation

- Update docs for interleaved fft

## [0.2.0] - 2024-04-25

### Features

- Add badges for docs and latest published version
- Add more examples and fix typos in readme/plots

- Added rust bindings of FFTW as an example, which will be used for
  benchmarks

- Add fftw (rust bindings) crate as a dev-dependency

- Add an example of using pyphastft to reproduce an example use case of
  FFT from the FFT wikipedia page

- Fix typos in the README and distinguish pyphastft from phastft in the
  python benchmarks plots
- Add todo for improving twiddle generation perf
- Add normalization & output order info. closes #13
- Add tests for 32 bit planner
- Add audio visualization as example for pyphastft
- Add automatic CPU feature detection

### Bug Fixes

- Fix formatting with `cargo fmt`
- Fix formatting
- Fix lint issue brought up by CI

- Fixed `clippy::needless_doctest_main` issue in README

### Other

- Swapped out use of sincos() for more portability across platforms
- Implement SIMDized twiddle generation using macro
- Finish adding test macros for f32/f64

- Make float comparison generic for f32/f64

- Add f32/f64 tests for twiddles using macros

### Refactor

- Make cobra mod public for fast bit reversal
- Use 16 lanes for f32 in SIMD butterfly kernel
- Run `cargo fmt`

### Documentation

- Update README and docs

### Miscellaneous Tasks

- Updated benchmark instructions and fixed typos
- Update python benchmark plot title
- Update benchmark plots
- Update benchmark, plotting, and profiling scripts
- Remove duplicate public functions
- Bump black from 24.1.1 to 24.3.0 in /benches

Bumps [black](https://github.com/psf/black) from 24.1.1 to 24.3.0.
- [Release notes](https://github.com/psf/black/releases)
- [Changelog](https://github.com/psf/black/blob/main/CHANGES.md)
- [Commits](https://github.com/psf/black/compare/24.1.1...24.3.0)

---
updated-dependencies:
- dependency-name: black
  dependency-type: direct:production
...

Signed-off-by: dependabot[bot] <support@github.com>
- Update example and demo video
- Cleanup docs and examples
- Update codecov uploader to v4
- Update benchmarking readme
- Update benchmark plots
- Update pyphastft benchmark plots

## [0.1.1] - 2024-02-13

### Other

- Release v0.1.1 and update name for python wrapper

### Documentation

- Update README.md

Just skip to features, right away

## [0.1.0] - 2024-02-12

### Features

- Add cargo config
- Add rayon back
- Add chunk size 4 kernel
- Add opt flags
- Add check for dups
- Add COBRA initial impl pre-build twiddle factors
- Add basic tests for bit reversal
- Add bash script for profiling and update readme
- Add a sequential bit reversal permutation
- Add iterator for roots of unity
- Add cobra implementation and integrate into FFT
- Add separate mod for kernels
- Add separate kernel for size 8 butterflies
- Add utils
- Add benchmarks for pyfftw, numpy, and PHFT
- Add test for the SIMD impl
- Add github workflow for CI
- Add status badge
- Add tests for bit rev; rand signal; touch-up docs
- Add codecov job and codecov config for 90% target
- Add code coverage badge
- Add benchmarking scripts
- Add dir with instructions & code to run benchmarks
- Add FFTW3 benchmark
- Add Options struct
- Add fftw3 to benchmark script
- Add a public function that accepts caller-provided `Options`
- Add benchmark plots
- Add plots to readme
- Add `Planner` and ammend examples, tests, etc.
- Add planner mod
- Add bench system config info
- Add license files
- Add asert to check input length is 2^n

- Add a regression test to make sure non-power-of-two FFTs are not
  allowed for the time being
- Add instructions for reproducing benchmarks
- Add tests and make API breaking changes

- Add a test to make sure inverse twiddle factors multiplied by forward
  twiddle factors always gives 1.0

- `fft` now takes the real/imaginary input and the `Direction` of the `fft`

- Pre-built planners can be used with the `fft_with_opts_and_plan`
  function
- Add regression test for planner/fft mismatch
- Add regression docs to explain regression test
- Add requirements.txt for benchmarks

### Bug Fixes

- Fixes for profiling
- Fix for prebuilt twiddles and portable SIMD
- Fix assertion
- Fix overhead from bit reversal threading
- Fix typo in benchmark logging
- Fix build
- Fix typos in benchmarking code
- Fix bug in phast benchmark & add fft direction
- Fix FFTW benchmark to use wall time
- Fix perf regression for small input sizes

- Increased block size increases performance for larger input sizes, but
  the increase caused a regression on a AMD 7950x machine, for smaller
  input sizes. The `Planner` needs to be updated to computes an ideal
  block width. In the interim, we will use 2 * CACHE_LINE_SIZE. On
  x86-64, this is usually 2 * 64 = 128.
- Fix assertion in `fft_with_opts_and_plan`

- Twiddles should always be half the size of input
- Fix typo

### Other

- Don't precompute twiddle factors for large chunks
- Split into two binaries for profiling
- Parallelize bit reverse permutation with 2 threads
- Experiments
- Pre-compute twiddle factors for chunk n
- Bug fix -- clear twiddle factor cache after use
- Abstract out twiddle factor generation
- Prebuild only half the required twiddles
- Try out generating twiddles lazily
- Restructure into lib, add examples, scripts, etc.
- Proof of concept of caching 1/4 of twiddle factors
- Unroll for half of twiddle factors
- Reuse generated twiddle factors by filtering
- Go back to only generating half of twiddles
- Faster twiddles hopefully workable concept this time
- Plug SIMD twiddle generation in
- Generate random signal for py benchmarks
- Increase `BLOCK_SIZE` to 256
- Rewrite README
- Tweaks to "Features" in README
- Wire up guessing the options to the main FFT function
- Placate Clippy
- Exponentially decrease # of iters in benchmark
- Change plot ordering so colors are consistent
- Print human readable bytes without decimal
- Adjust pybindings to work with new API
- Expand on the Python installation instructions
- Mention the license in the README
- Set repository and unset documentation field; it will default to docs.rs
- Mention lower memory use in the "vs" section
- Automate findings of latest benchmark results
- Format benchmark scripts
- Expose `cobra_apply` for fast bit reverseal

### Refactor

- Use std `reverse_bits` for performance improvement
- Use faster impl of bit reversal implementation
- Replace vec with array since it's < 512_000 bytes
- Switch back to pre-built cache using vec
- Use in-place bit reversal in COBRA for small N
- Move kernels to separate mod
- Simplify size 2 butterfly
- Use numpy crate to avoid overhead in pybindings
- Use scipy to generate random signal
- Move rustfft benchmark to examples
- Use random signal for benchmarking rustfft
- Use random signal generator to benchmark phastft
- Move utility functions to separate lib
- Simplify options down to a single multi-threading knob

### Documentation

- Update README
- Update README with basic skeleton
- Update READMEs
- Update README
- Update README
- Update README text to match latest benchmarks
- Update documentation and readme
- Update README.md
- Update README.md

### Performance

- Optimize twiddles pre-compute
- Improve recursive bit-reversal perf for testing

- Tests run much faster now with a bit of loop unrolling, loop fusion,
  and pre-allocating vectors with the right sizes

- Add documentation and cite references
- Improve assertion functions

### Miscellaneous Tasks

- Remove rayon for profiling
- Cleanup for profiling
- Remove rayon for profiling
- Remove unused mods
- Update for benchmarks
- Update profiling section in readme
- Update profile script, add profiling to examples
- Remove useless files
- Remove benchmark mod from lib
- Cleanup all warnings and unused code
- Remove rayon dep, add bar plot, rename to PhastFT
- Update workflow
- Remove old BRAVO implementation written in python
- Updates to examples and plotting script
- Update benchmark plotting
- Update benchmark scripts
- Remove workspace
- Cleanup x-axis of plots
- Update plots for latest benchmarks
- Update the license in Cargo.toml
- Drop unused author fields
- Update MIT license for future contributors
- Cleanup plots and split bar plot for py benches
- Update benchmark plots
- Update saved figure names in benches
- Update benchmark readme

### Revert

- Revert to creating twiddle factors for large n
- Revert "Prebuild only half the required twiddles"

This reverts commit e552fb1ea2b5202c47e4cf1e8392adcefe1d476b.
- Revert "Proof of concept of caching 1/4 of twiddle factors"

This reverts commit de2cca02d99c93b76e3b42bf09d29deeb2eda62f.
- Revert "Add separate kernel for size 8 butterflies"

This reverts commit 249be77548a8b2f23aca4739a5d3aad536939cba.

<!-- generated by git-cliff -->
