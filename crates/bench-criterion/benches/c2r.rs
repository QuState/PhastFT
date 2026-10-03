//! Important: this benchmark only measures small-to-mid sizes; criterion is
//! not a good fit for measuring long-running tasks — see
//! `examples/benchmark.rs` for the harness for large sizes.
//!
//! This benchmark compares complex-to-real (C2R) FFT execution across
//! PhastFT, realfft, and FFTW for `f32` and `f64` where supported.
//!
//! FFTW is benchmarked with `ESTIMATE`, `MEASURE`, and `CONSERVE_MEMORY`
//! planning modes. FFTW wisdom is explicitly cleared between each mode so
//! that the planning modes remain isolated and do not reuse wisdom generated
//! by a previous benchmark.
//!
//! The PhastFT, realfft, and FFTW bench binaries all write into the same
//! `target/criterion/<group>/<id>/<size>/` tree; criterion does NOT
//! auto-aggregate across binaries, so use
//! `benches/plot_criterion_overlay.py` to produce a single overlay plot per
//! group after running them all.
#[cfg(feature = "fftw")]
use benchmark_criterion::backend::fftw_lib::fftw_c2r_all;
use benchmark_criterion::backend::phastft_lib::{phastft_c2r_f32, phastft_c2r_f64};
#[cfg(feature = "realfft")]
use benchmark_criterion::backend::realfft_lib::{realfft_c2r_f32, realfft_c2r_f64};
use criterion::criterion_main;

#[doc = "The function which runs the benchmarks."]
pub fn benches() {
    let mut criterion: criterion::Criterion<_> =
        (criterion::Criterion::default()).configure_from_args();
    phastft_c2r_f32(&mut criterion);
    phastft_c2r_f64(&mut criterion);
    #[cfg(feature = "realfft")]
    realfft_c2r_f32(&mut criterion);
    #[cfg(feature = "realfft")]
    realfft_c2r_f64(&mut criterion);
    #[cfg(feature = "fftw")]
    fftw_c2r_all(&mut criterion);
}
criterion_main!(benches);
