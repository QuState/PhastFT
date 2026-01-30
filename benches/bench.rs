use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use num_traits::Float;
use phastft::options::Options;
use phastft::planner::{Direction, Planner32, Planner64, PlannerDit32, PlannerDit64};
use phastft::{
    fft_32_dit_with_planner_and_opts, fft_32_with_opts_and_plan, fft_64_dit_with_planner_and_opts,
    fft_64_with_opts_and_plan,
};
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use utilities::rustfft::num_complex::Complex;
use utilities::rustfft::FftPlanner;

const LENGTHS: &[usize] = &[
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
];

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

fn generate_complex_numbers<T: Float + Default>(n: usize) -> Vec<Complex<T>>
where
    StandardUniform: Distribution<T>,
{
    let mut rng = SmallRng::from_os_rng();

    let samples: Vec<T> = (&mut rng)
        .sample_iter(StandardUniform)
        .take(2 * n)
        .collect();

    let mut signal = vec![Complex::default(); n];

    for (z, rand_chunk) in signal.iter_mut().zip(samples.chunks_exact(2)) {
        z.re = rand_chunk[0];
        z.im = rand_chunk[1];
    }

    signal
}

fn benchmark_forward_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Forward f32");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );
    group.sample_size(20);

    for n in LENGTHS.iter() {
        let len = 1 << n; // 2.pow(n)
        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f32>()) as u64,
        });

        let id = "PhastFT DIF";
        let options = Options::guess_options(len);
        let planner = Planner32::new(len, Direction::Forward);

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_numbers::<f32>(len),
                |(mut reals, mut imags)| {
                    fft_32_with_opts_and_plan(&mut reals, &mut imags, &options, &planner);
                },
                BatchSize::SmallInput,
            );
        });

        let id = "PhastFT DIT";
        let options = Options::guess_options(len);
        let planner_dit = PlannerDit32::new(len, Direction::Forward);

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_numbers::<f32>(len),
                |(mut reals, mut imags)| {
                    fft_32_dit_with_planner_and_opts(
                        &mut reals,
                        &mut imags,
                        &planner_dit,
                        &options,
                    );
                },
                BatchSize::SmallInput,
            );
        });

        let id = "RustFFT";
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(len);

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_complex_numbers::<f32>(len),
                |mut signal| {
                    fft.process(&mut signal);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_inverse_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Inverse f32");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );
    group.sample_size(20);

    for n in LENGTHS.iter() {
        let len = 1 << n; // 2.pow(n)
        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f32>()) as u64,
        });

        let id = "PhastFT DIF";
        let options = Options::guess_options(len);
        let planner = Planner32::new(len, Direction::Reverse);

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_numbers::<f32>(len),
                |(mut reals, mut imags)| {
                    fft_32_with_opts_and_plan(&mut reals, &mut imags, &options, &planner);
                },
                BatchSize::SmallInput,
            );
        });

        let id = "PhastFT DIT";
        let options = Options::guess_options(len);
        let planner_dit = PlannerDit32::new(len, Direction::Reverse);

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_numbers::<f32>(len),
                |(mut reals, mut imags)| {
                    fft_32_dit_with_planner_and_opts(
                        &mut reals,
                        &mut imags,
                        &planner_dit,
                        &options,
                    );
                },
                BatchSize::SmallInput,
            );
        });

        let id = "RustFFT";
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_inverse(len);

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_complex_numbers::<f32>(len),
                |mut signal| {
                    fft.process(&mut signal);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_forward_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("Forward f64");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );
    group.sample_size(20);

    for n in LENGTHS.iter() {
        let len = 1 << n; // 2.pow(n)
        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f64>()) as u64,
        });

        let id = "PhastFT DIF";
        let options = Options::guess_options(len);
        let planner = Planner64::new(len, Direction::Forward);

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_numbers::<f64>(len),
                |(mut reals, mut imags)| {
                    fft_64_with_opts_and_plan(&mut reals, &mut imags, &options, &planner);
                },
                BatchSize::SmallInput,
            );
        });

        let id = "PhastFT DIT";
        let options = Options::guess_options(len);
        let planner_dit = PlannerDit64::new(len, Direction::Forward);

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_numbers::<f64>(len),
                |(mut reals, mut imags)| {
                    fft_64_dit_with_planner_and_opts(
                        &mut reals,
                        &mut imags,
                        &planner_dit,
                        &options,
                    );
                },
                BatchSize::SmallInput,
            );
        });

        let id = "RustFFT";
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(len);

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_complex_numbers::<f64>(len),
                |mut signal| {
                    fft.process(&mut signal);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_inverse_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("Inverse f64");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );
    group.sample_size(20);

    for n in LENGTHS.iter() {
        let len = 1 << n; // 2.pow(n)
        group.throughput(Throughput::ElementsAndBytes {
            elements: len as u64,
            bytes: (len * size_of::<f64>()) as u64,
        });

        let id = "PhastFT DIF";
        let options = Options::guess_options(len);
        let planner = Planner64::new(len, Direction::Reverse);

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_numbers::<f64>(len),
                |(mut reals, mut imags)| {
                    fft_64_with_opts_and_plan(&mut reals, &mut imags, &options, &planner);
                },
                BatchSize::SmallInput,
            );
        });

        let id = "PhastFT DIT";
        let options = Options::guess_options(len);
        let planner_dit = PlannerDit64::new(len, Direction::Reverse);

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_numbers::<f64>(len),
                |(mut reals, mut imags)| {
                    fft_64_dit_with_planner_and_opts(
                        &mut reals,
                        &mut imags,
                        &planner_dit,
                        &options,
                    );
                },
                BatchSize::SmallInput,
            );
        });

        let id = "RustFFT";
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_inverse(len);

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_complex_numbers::<f64>(len),
                |mut signal| {
                    fft.process(&mut signal);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_forward_f32,
    benchmark_inverse_f32,
    benchmark_forward_f64,
    benchmark_inverse_f64,
);
criterion_main!(benches);
