use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use num_traits::Float;
use phastft::{
    fft_32_with_opts_and_plan, fft_64_with_opts_and_plan,
    options::Options,
    planner::{Direction, Planner32, Planner64},
};
use rand::{distr::StandardUniform, prelude::Distribution, rng, Rng};
use utilities::rustfft::num_complex::Complex;
use utilities::rustfft::FftPlanner;

const LENGTHS: &[usize] = &[
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
];

fn generate_numbers<T: Float>(n: usize) -> (Vec<T>, Vec<T>)
where
    StandardUniform: Distribution<T>,
{
    let mut rng = rng();

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
    let mut rng = rng();

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

    for n in LENGTHS.iter() {
        let len = 1 << n;
        group.throughput(Throughput::Elements(len as u64));

        let id = "PhastFT FFT Forward";
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

        let id = "RustFFT FFT Forward";
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

    for n in LENGTHS.iter() {
        let len = 1 << n;
        group.throughput(Throughput::Elements(len as u64));

        let id = "PhastFT FFT Inverse";
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
    }
    group.finish();
}

fn benchmark_forward_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("Forward f64");

    for n in LENGTHS.iter() {
        let len = 1 << n;
        group.throughput(Throughput::Elements(len as u64));

        let id = "PhastFT FFT Forward";
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

        let id = "RustFFT FFT Forward";
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

    for n in LENGTHS.iter() {
        let len = 1 << n;
        group.throughput(Throughput::Elements(len as u64));

        let id = "PhastFT FFT Inverse";
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
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_forward_f32,
    benchmark_inverse_f32,
    benchmark_forward_f64,
    benchmark_inverse_f64
);
criterion_main!(benches);
