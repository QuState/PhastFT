use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use num_traits::Float;
use phastft::options::Options;
use phastft::planner::{Direction, PlannerDit32, PlannerDit64, PlannerMode};
use phastft::{fft_32_dit_with_planner_and_opts, fft_64_dit_with_planner_and_opts};
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const LENGTHS: &[usize] = &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];

fn generate_numbers<T: Float>(n: usize) -> (Vec<T>, Vec<T>)
where
    StandardUniform: Distribution<T>,
{
    let mut rng = SmallRng::from_os_rng();

    let samples: Vec<T> = (&mut rng)
        .sample_iter(StandardUniform)
        .take(2 * n)
        .collect();

    let mut reals = vec![T::zero(); n];
    let mut imags = vec![T::zero(); n];

    for ((z_re, z_im), rand_chunk) in reals
        .iter_mut()
        .zip(imags.iter_mut())
        .zip(samples.chunks_exact(2))
    {
        *z_re = rand_chunk[0];
        *z_im = rand_chunk[1];
    }

    (reals, imags)
}

fn benchmark_heuristic_vs_tune_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("Heuristic vs Tune f64");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );
    group.sample_size(20);

    for n in LENGTHS.iter() {
        let len = 1 << n;
        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f64>()) as u64,
        });

        let options = Options::guess_options(len);

        let planner_heuristic = PlannerDit64::new(len, Direction::Forward);
        group.bench_function(BenchmarkId::new("Heuristic", len), |b| {
            b.iter_batched(
                || generate_numbers::<f64>(len),
                |(mut reals, mut imags)| {
                    fft_64_dit_with_planner_and_opts(
                        &mut reals,
                        &mut imags,
                        &planner_heuristic,
                        &options,
                    );
                },
                BatchSize::SmallInput,
            );
        });

        let planner_tune = PlannerDit64::with_mode(len, Direction::Forward, PlannerMode::Tune);
        group.bench_function(BenchmarkId::new("Tune", len), |b| {
            b.iter_batched(
                || generate_numbers::<f64>(len),
                |(mut reals, mut imags)| {
                    fft_64_dit_with_planner_and_opts(
                        &mut reals,
                        &mut imags,
                        &planner_tune,
                        &options,
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_heuristic_vs_tune_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Heuristic vs Tune f32");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );
    group.sample_size(20);

    for n in LENGTHS.iter() {
        let len = 1 << n;
        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f32>()) as u64,
        });

        let options = Options::guess_options(len);

        let planner_heuristic = PlannerDit32::new(len, Direction::Forward);
        group.bench_function(BenchmarkId::new("Heuristic", len), |b| {
            b.iter_batched(
                || generate_numbers::<f32>(len),
                |(mut reals, mut imags)| {
                    fft_32_dit_with_planner_and_opts(
                        &mut reals,
                        &mut imags,
                        &planner_heuristic,
                        &options,
                    );
                },
                BatchSize::SmallInput,
            );
        });

        let planner_tune = PlannerDit32::with_mode(len, Direction::Forward, PlannerMode::Tune);
        group.bench_function(BenchmarkId::new("Tune", len), |b| {
            b.iter_batched(
                || generate_numbers::<f32>(len),
                |(mut reals, mut imags)| {
                    fft_32_dit_with_planner_and_opts(
                        &mut reals,
                        &mut imags,
                        &planner_tune,
                        &options,
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_heuristic_vs_tune_f64,
    benchmark_heuristic_vs_tune_f32,
);
criterion_main!(benches);
