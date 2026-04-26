use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use phastft::options::Options;
use phastft::planner::{Direction, PlannerDit32, PlannerDit64};
use phastft::{fft_32_dit_with_planner_and_opts, fft_64_dit_with_planner_and_opts};

mod common;
use common::{groups, ids, split_complex, sweep_complex, LENGTHS};

// IMPORTANT NOTE:
// This benchmark only measures small-to-mid sizes, which are not the focus of
// PhastFT. Criterion is not a good fit for measuring long-running tasks — see
// examples/benchmark.rs for the harness for large sizes.
//
// The PhastFT, RustFFT, and FFTW bench binaries all write into the same
// `target/criterion/<group>/<id>/<size>/` tree; criterion does NOT auto-
// aggregate across binaries, so use `benches/plot_criterion_overlay.py` to
// produce a single overlay plot per group after running them all.

macro_rules! phastft_c2c {
    ($name:ident, $float:ty, $planner:ty, $fft:ident, $dir:expr, $group:expr) => {
        fn $name(c: &mut Criterion) {
            sweep_complex::<$float, _>(c, $group, LENGTHS, |g, len| {
                let opts = Options::guess_options(len);
                let planner = <$planner>::new(len);
                g.bench_function(BenchmarkId::new(ids::PHASTFT_DIT, len), |b| {
                    b.iter_batched(
                        || split_complex::<$float>(len),
                        |(mut reals, mut imags)| {
                            $fft(&mut reals, &mut imags, $dir, &planner, &opts);
                            std::hint::black_box((&mut reals, &mut imags));
                        },
                        BatchSize::SmallInput,
                    );
                });
            });
        }
    };
}

phastft_c2c!(
    fwd_f32,
    f32,
    PlannerDit32,
    fft_32_dit_with_planner_and_opts,
    Direction::Forward,
    groups::C2C_FORWARD_F32
);
phastft_c2c!(
    inv_f32,
    f32,
    PlannerDit32,
    fft_32_dit_with_planner_and_opts,
    Direction::Reverse,
    groups::C2C_INVERSE_F32
);
phastft_c2c!(
    fwd_f64,
    f64,
    PlannerDit64,
    fft_64_dit_with_planner_and_opts,
    Direction::Forward,
    groups::C2C_FORWARD_F64
);
phastft_c2c!(
    inv_f64,
    f64,
    PlannerDit64,
    fft_64_dit_with_planner_and_opts,
    Direction::Reverse,
    groups::C2C_INVERSE_F64
);

criterion_group!(benches, fwd_f32, inv_f32, fwd_f64, inv_f64);
criterion_main!(benches);
