//! Important: this benchmark only measures small-to-mid sizes; criterion is
//! not a good fit for measuring long-running tasks — see
//! `examples/benchmark.rs` for the harness for large sizes.
//!
//! Unlike the C2C cross-library comparison (split across `bench.rs` vs.
//! `rustfft.rs` vs. `fftw_*.rs`), both PhastFT R2C/C2R and the realfft
//! baseline live in this single bench binary. The split-per-library
//! convention exists primarily to isolate FFTW's per-process wisdom cache
//! between planning modes; realfft has no such cache, so a single binary
//! suffices and gives a self-contained PhastFT-vs-realfft comparison.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use phastft::planner::{PlannerR2c32, PlannerR2c64};
use phastft::{
    c2r_fft_f32_with_planner_and_scratch, c2r_fft_f64_with_planner_and_scratch,
    r2c_fft_f32_with_planner, r2c_fft_f64_with_planner,
};
use realfft::RealFftPlanner;

mod common;
use common::{groups, ids, real_signal, spectrum_interleaved, spectrum_split, sweep_real, LENGTHS};
//
// Group names (snake_case): r2c_f32 / r2c_f64 / c2r_f32 / c2r_f64 — distinct
// from the C2C groups, so no overlay aggregation across binaries needed.

macro_rules! r2c_bench {
    ($name:ident, $float:ty, $planner:ty, $fft_fn:ident, $group:expr) => {
        fn $name(c: &mut Criterion) {
            sweep_real::<$float, _>(c, $group, LENGTHS, |g, len| {
                // Plan + output buffers allocated outside iter_batched —
                // planning and allocation cost is excluded from per-sample timings.
                let phast_planner = <$planner>::new(len);
                let mut phast_re = vec![0 as $float; len / 2 + 1];
                let mut phast_im = vec![0 as $float; len / 2 + 1];
                g.bench_function(BenchmarkId::new(ids::PHASTFT_R2C, len), |b| {
                    b.iter_batched(
                        || real_signal::<$float>(len),
                        |input| {
                            $fft_fn(&input, &mut phast_re, &mut phast_im, &phast_planner);
                            std::hint::black_box((&mut phast_re, &mut phast_im));
                        },
                        BatchSize::SmallInput,
                    );
                });

                let mut rf_planner = RealFftPlanner::<$float>::new();
                let rf_r2c = rf_planner.plan_fft_forward(len);
                let mut rf_output = rf_r2c.make_output_vec();
                let mut rf_scratch = rf_r2c.make_scratch_vec();
                g.bench_function(BenchmarkId::new(ids::REALFFT, len), |b| {
                    b.iter_batched(
                        || real_signal::<$float>(len),
                        |mut input| {
                            rf_r2c
                                .process_with_scratch(&mut input, &mut rf_output, &mut rf_scratch)
                                .unwrap();
                            std::hint::black_box(&mut rf_output);
                        },
                        BatchSize::SmallInput,
                    );
                });
            });
        }
    };
}

macro_rules! c2r_bench {
    ($name:ident, $float:ty, $planner:ty, $fft_fn:ident, $group:expr) => {
        fn $name(c: &mut Criterion) {
            sweep_real::<$float, _>(c, $group, LENGTHS, |g, len| {
                let phast_planner = <$planner>::new(len);
                let mut phast_output = vec![0 as $float; len];
                let mut phast_scratch_re = vec![0 as $float; len / 2];
                let mut phast_scratch_im = vec![0 as $float; len / 2];
                g.bench_function(BenchmarkId::new(ids::PHASTFT_C2R, len), |b| {
                    b.iter_batched(
                        || spectrum_split::<$float>(len),
                        |(input_re, input_im)| {
                            $fft_fn(
                                &input_re,
                                &input_im,
                                &mut phast_output,
                                &phast_planner,
                                &mut phast_scratch_re,
                                &mut phast_scratch_im,
                            );
                            std::hint::black_box(&mut phast_output);
                        },
                        BatchSize::SmallInput,
                    );
                });

                let mut rf_planner = RealFftPlanner::<$float>::new();
                let rf_c2r = rf_planner.plan_fft_inverse(len);
                let mut rf_output = rf_c2r.make_output_vec();
                let mut rf_scratch = rf_c2r.make_scratch_vec();
                g.bench_function(BenchmarkId::new(ids::REALFFT, len), |b| {
                    b.iter_batched(
                        || spectrum_interleaved::<$float>(len),
                        |mut input| {
                            rf_c2r
                                .process_with_scratch(&mut input, &mut rf_output, &mut rf_scratch)
                                .unwrap();
                            std::hint::black_box(&mut rf_output);
                        },
                        BatchSize::SmallInput,
                    );
                });
            });
        }
    };
}

r2c_bench!(
    r2c_f32,
    f32,
    PlannerR2c32,
    r2c_fft_f32_with_planner,
    groups::R2C_F32
);
r2c_bench!(
    r2c_f64,
    f64,
    PlannerR2c64,
    r2c_fft_f64_with_planner,
    groups::R2C_F64
);
c2r_bench!(
    c2r_f32,
    f32,
    PlannerR2c32,
    c2r_fft_f32_with_planner_and_scratch,
    groups::C2R_F32
);
c2r_bench!(
    c2r_f64,
    f64,
    PlannerR2c64,
    c2r_fft_f64_with_planner_and_scratch,
    groups::C2R_F64
);

criterion_group!(benches, r2c_f32, c2r_f32, r2c_f64, c2r_f64);
criterion_main!(benches);
