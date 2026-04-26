use criterion::{criterion_group, criterion_main, Criterion};
use fftw::types::Flag;

mod common;
mod fftw_lib;

// IMPORTANT NOTE:
// This benchmark only measures small-to-mid sizes; criterion is not a good
// fit for measuring long-running tasks — see examples/benchmark.rs for the
// harness for large sizes.
//
// Each FFTW planning mode (Estimate / Measure / Conserve) lives in its own
// `[[bench]]` binary so FFTW's global per-process wisdom cache cannot leak
// between modes; each run starts with a fresh process and empty wisdom.
// Group names are shared with bench.rs / rustfft.rs / the other fftw_*.rs
// binaries; criterion does NOT auto-aggregate across binaries — use
// `benches/plot_criterion_overlay.py` for the cross-binary overlay.
//
// This series combines MEASURE with FFTW_CONSERVE_MEMORY so FFTW selects
// memory-efficient plan variants during the same search it would run for
// plain MEASURE. This is a more faithful comparison for PhastFT, which is
// also designed for low memory overhead — plain PATIENT / MEASURE let FFTW
// spend memory freely in pursuit of speed.

fn run(c: &mut Criterion) {
    fftw_lib::run_all(
        c,
        common::ids::FFTW_CONSERVE,
        Flag::DESTROYINPUT | Flag::MEASURE | Flag::CONSERVEMEMORY,
    );
}

criterion_group!(benches, run);
criterion_main!(benches);
