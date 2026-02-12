use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use phastft::planner::{Direction, Planner32, Planner64, PlannerDit32, PlannerDit64};
use utilities::rustfft::FftPlanner;

const LENGTHS: &[usize] = &[
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
];

fn benchmark_planner_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Planner f32");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    for n in LENGTHS.iter() {
        let len = 1 << n;

        group.bench_function(BenchmarkId::new("PhastFT DIF", len), |b| {
            b.iter(|| Planner32::new(len, Direction::Forward));
        });

        group.bench_function(BenchmarkId::new("PhastFT DIT", len), |b| {
            b.iter(|| PlannerDit32::new(len, Direction::Forward));
        });

        group.bench_function(BenchmarkId::new("RustFFT", len), |b| {
            b.iter(|| {
                let mut planner = FftPlanner::<f32>::new();
                planner.plan_fft_forward(len)
            });
        });
    }
    group.finish();
}

fn benchmark_planner_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("Planner f64");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    for n in LENGTHS.iter() {
        let len = 1 << n;

        group.bench_function(BenchmarkId::new("PhastFT DIF", len), |b| {
            b.iter(|| Planner64::new(len, Direction::Forward));
        });

        group.bench_function(BenchmarkId::new("PhastFT DIT", len), |b| {
            b.iter(|| PlannerDit64::new(len, Direction::Forward));
        });

        group.bench_function(BenchmarkId::new("RustFFT", len), |b| {
            b.iter(|| {
                let mut planner = FftPlanner::<f64>::new();
                planner.plan_fft_forward(len)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_planner_f32, benchmark_planner_f64);
criterion_main!(benches);
