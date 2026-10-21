//! Important: this benchmark only measures small-to-mid sizes, which are
//! not the focus of PhastFT. Criterion is not a good fit for measuring
//! long-running tasks — see `examples/benchmark.rs` for the harness for
//! large sizes.
//!
//! The PhastFT, RustFFT, and FFTW bench binaries all write into the same
//! `target/criterion/<group>/<id>/<size>/` tree; criterion does NOT
//! auto-aggregate across binaries, so use
//! `benches/plot_criterion_overlay.py` to produce a single overlay plot per
//! group after running them all.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use phastft::options::Options;
use phastft::planner::{Direction, PlannerDit32, PlannerDit64};
use phastft::{fft_f32_dit_with_planner_and_opts, fft_f64_dit_with_planner_and_opts};

mod common;
use common::{
    bench_transforms_at_sizes, groups, ids, split_complex, throughput_complex,
    transform_batch_size, LENGTHS,
};

macro_rules! phastft_c2c {
    ($name:ident, $float:ty, $planner:ty, $fft:ident, $dir:expr, $group:expr) => {
        fn $name(c: &mut Criterion) {
            bench_transforms_at_sizes(
                c,
                $group,
                LENGTHS,
                throughput_complex::<$float>,
                |g, len| {
                    let opts = Options::guess_options(len);
                    let planner = <$planner>::new(len);
                    g.bench_function(BenchmarkId::new(ids::PHASTFT_DIT, len), |b| {
                        b.iter_batched_ref(
                            || split_complex::<$float>(len),
                            |(reals, imags)| {
                                $fft(reals, imags, $dir, &planner, &opts);
                                std::hint::black_box((reals, imags));
                            },
                            transform_batch_size(2 * len * size_of::<$float>()),
                        );
                    });
                },
            );
        }
    };
}

phastft_c2c!(
    fwd_f32,
    f32,
    PlannerDit32,
    fft_f32_dit_with_planner_and_opts,
    Direction::Forward,
    groups::C2C_FORWARD_F32
);
phastft_c2c!(
    inv_f32,
    f32,
    PlannerDit32,
    fft_f32_dit_with_planner_and_opts,
    Direction::Inverse,
    groups::C2C_INVERSE_F32
);
phastft_c2c!(
    fwd_f64,
    f64,
    PlannerDit64,
    fft_f64_dit_with_planner_and_opts,
    Direction::Forward,
    groups::C2C_FORWARD_F64
);
phastft_c2c!(
    inv_f64,
    f64,
    PlannerDit64,
    fft_f64_dit_with_planner_and_opts,
    Direction::Inverse,
    groups::C2C_INVERSE_F64
);

criterion_group!(benches, fwd_f32, inv_f32, fwd_f64, inv_f64);
criterion_main!(benches);
