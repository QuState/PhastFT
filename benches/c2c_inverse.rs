//! Important: this benchmark only measures small-to-mid sizes, which are
//! not the focus of PhastFT. Criterion is not a good fit for measuring
//! long-running tasks — see `examples/benchmark.rs` for the harness for
//! large sizes.
//!
//! This benchmark compares inverse C2C FFT execution across PhastFT,
//! RustFFT, and FFTW for `f32` and `f64` where supported.
//!
//! FFTW is benchmarked with `ESTIMATE`, `MEASURE`, and `CONSERVE_MEMORY`
//! planning modes. FFTW wisdom is explicitly cleared between each mode so
//! that the planning modes remain isolated and do not reuse wisdom generated
//! by a previous benchmark.
//!
//! The PhastFT, RustFFT, and FFTW bench binaries all write into the same
//! `target/criterion/<group>/<id>/<size>/` tree; criterion does NOT
//! auto-aggregate across binaries, so use
//! `benches/plot_criterion_overlay.py` to produce a single overlay plot per
//! group after running them all.

use criterion::{criterion_group, criterion_main};

use crate::backend::fftw_lib::fftw_c2c_inv_all;
use crate::backend::phastft_lib::{phastft_c2c_inv_f32, phastft_c2c_inv_f64};
use crate::backend::rustfft_lib::{rustfft_inv_f32, rustfft_inv_f64};

mod backend;
mod common;

criterion_group!(
    benches,
    phastft_c2c_inv_f32,
    phastft_c2c_inv_f64,
    rustfft_inv_f32,
    rustfft_inv_f64,
    fftw_c2c_inv_all
);
criterion_main!(benches);
