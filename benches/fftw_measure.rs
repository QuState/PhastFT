//! Important: this benchmark only measures small-to-mid sizes; criterion is
//! not a good fit for measuring long-running tasks — see
//! `examples/benchmark.rs` for the harness for large sizes.
//!
//! Each FFTW planning mode (Estimate / Measure / Conserve) lives in its own
//! `[[bench]]` binary so FFTW's global per-process wisdom cache cannot leak
//! between modes; each run starts with a fresh process and empty wisdom.
//! Group names are shared with `bench.rs` / `rustfft.rs` / the other
//! `fftw_*.rs` binaries; criterion does NOT auto-aggregate across binaries
//! — use `benches/plot_criterion_overlay.py` for the cross-binary overlay.
//!
//! FFTW's MEASURE is the default rigor level (bit value 0 in the fftw
//! crate's `Flag` bitflags), so `| Flag::MEASURE` is a no-op at runtime; it
//! is kept here so the selected rigor level is visible alongside the other
//! flags.

use criterion::{criterion_group, criterion_main, Criterion};
use fftw::types::Flag;

mod common;
mod fftw_lib;

fn run(c: &mut Criterion) {
    fftw_lib::run_all(
        c,
        common::ids::FFTW_MEASURE,
        Flag::DESTROYINPUT | Flag::MEASURE,
    );
}

criterion_group!(benches, run);
criterion_main!(benches);
