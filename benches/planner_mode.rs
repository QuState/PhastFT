//! Compares `PlannerMode::Heuristic` vs. `PlannerMode::Tune` on the same
//! PhastFT FFT call. The two modes currently produce identical planners
//! (the `_mode: PlannerMode` argument is ignored by
//! `PlannerDit*::with_mode`), so the bench primarily pins the API surface
//! — once `Tune` ships a real search, this becomes the apples-to-apples
//! comparison.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use phastft::options::Options;
use phastft::planner::{Direction, PlannerDit32, PlannerDit64, PlannerMode};
use phastft::{fft_32_dit_with_planner_and_opts, fft_64_dit_with_planner_and_opts};

mod common;
use common::{
    bench_at_sizes, groups, ids, split_complex, throughput_complex, PLANNER_MODE_LENGTHS,
};

macro_rules! planner_mode_bench {
    ($name:ident, $float:ty, $planner:ty, $fft:ident, $group:expr) => {
        fn $name(c: &mut Criterion) {
            bench_at_sizes(
                c,
                $group,
                PLANNER_MODE_LENGTHS,
                throughput_complex::<$float>,
                |g, len| {
                    let opts = Options::guess_options(len);
                    for (id, mode) in [
                        (ids::HEURISTIC, PlannerMode::Heuristic),
                        (ids::TUNE, PlannerMode::Tune),
                    ] {
                        let planner = <$planner>::with_mode(len, mode);
                        g.bench_function(BenchmarkId::new(id, len), |b| {
                            b.iter_batched(
                                || split_complex::<$float>(len),
                                |(mut reals, mut imags)| {
                                    $fft(
                                        &mut reals,
                                        &mut imags,
                                        Direction::Forward,
                                        &planner,
                                        &opts,
                                    );
                                },
                                BatchSize::SmallInput,
                            );
                        });
                    }
                },
            );
        }
    };
}

planner_mode_bench!(
    mode_f32,
    f32,
    PlannerDit32,
    fft_32_dit_with_planner_and_opts,
    groups::PLANNER_MODE_F32
);
planner_mode_bench!(
    mode_f64,
    f64,
    PlannerDit64,
    fft_64_dit_with_planner_and_opts,
    groups::PLANNER_MODE_F64
);

criterion_group!(benches, mode_f32, mode_f64);
criterion_main!(benches);
