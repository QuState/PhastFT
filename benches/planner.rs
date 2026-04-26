use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use phastft::planner::{PlannerDit32, PlannerDit64};
use utilities::rustfft::FftPlanner;

mod common;
use common::{groups, ids, sweep_complex, LENGTHS};

macro_rules! planner_bench {
    ($name:ident, $float:ty, $planner:ty, $group:expr) => {
        fn $name(c: &mut Criterion) {
            sweep_complex::<$float, _>(c, $group, LENGTHS, |g, len| {
                g.bench_function(BenchmarkId::new(ids::PHASTFT_DIT, len), |b| {
                    b.iter(|| <$planner>::new(len));
                });
                g.bench_function(BenchmarkId::new(ids::RUSTFFT, len), |b| {
                    b.iter(|| {
                        let mut planner = FftPlanner::<$float>::new();
                        planner.plan_fft_forward(len)
                    });
                });
            });
        }
    };
}

planner_bench!(planner_f32, f32, PlannerDit32, groups::PLANNER_F32);
planner_bench!(planner_f64, f64, PlannerDit64, groups::PLANNER_F64);

criterion_group!(benches, planner_f32, planner_f64);
criterion_main!(benches);
