use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use phastft::complex_nums::{combine_re_im_into, deinterleave};
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// This benchmark requires a custom cfg to run to access private functions.
// Run it with:
// RUSTFLAGS="--cfg phastft_bench" cargo bench --bench interleave --features complex-nums

const LENGTHS: &[usize] = &[
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
];

fn generate_interleaved_f32(n: usize) -> Vec<f32> {
    let mut rng = SmallRng::from_os_rng();
    (&mut rng)
        .sample_iter(StandardUniform)
        .take(2 * n)
        .collect()
}

fn generate_interleaved_f64(n: usize) -> Vec<f64> {
    let mut rng = SmallRng::from_os_rng();
    (&mut rng)
        .sample_iter(StandardUniform)
        .take(2 * n)
        .collect()
}

fn generate_re_im_f32(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = SmallRng::from_os_rng();
    let reals: Vec<f32> = (&mut rng).sample_iter(StandardUniform).take(n).collect();
    let imags: Vec<f32> = (&mut rng).sample_iter(StandardUniform).take(n).collect();
    (reals, imags)
}

fn generate_re_im_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = SmallRng::from_os_rng();
    let reals: Vec<f64> = (&mut rng).sample_iter(StandardUniform).take(n).collect();
    let imags: Vec<f64> = (&mut rng).sample_iter(StandardUniform).take(n).collect();
    (reals, imags)
}

fn benchmark_deinterleave_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Deinterleave f32");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    for n in LENGTHS.iter() {
        let len = 1 << n;
        group.throughput(Throughput::Bytes((len * 2 * size_of::<f32>()) as u64));

        group.bench_function(BenchmarkId::new("deinterleave", len), |b| {
            let level = fearless_simd::Level::new();
            b.iter_batched(
                || generate_interleaved_f32(len),
                |input| deinterleave(&input),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_deinterleave_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("Deinterleave f64");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    for n in LENGTHS.iter() {
        let len = 1 << n;
        group.throughput(Throughput::Bytes((len * 2 * size_of::<f64>()) as u64));

        group.bench_function(BenchmarkId::new("deinterleave", len), |b| {
            let level = fearless_simd::Level::new();
            b.iter_batched(
                || generate_interleaved_f64(len),
                |input| deinterleave(&input),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_combine_re_im_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Combine re/im f32");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    for n in LENGTHS.iter() {
        let len = 1 << n;
        group.throughput(Throughput::Bytes((len * 2 * size_of::<f32>()) as u64));

        group.bench_function(BenchmarkId::new("combine_re_im", len), |b| {
            b.iter_batched(
                || {
                    let (reals, imags) = generate_re_im_f32(len);
                    let output = vec![0.0f32; len * 2];
                    (reals, imags, output)
                },
                |(reals, imags, mut output)| combine_re_im_into(&reals, &imags, &mut output),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_combine_re_im_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("Combine re/im f64");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    for n in LENGTHS.iter() {
        let len = 1 << n;
        group.throughput(Throughput::Bytes((len * 2 * size_of::<f64>()) as u64));

        group.bench_function(BenchmarkId::new("combine_re_im", len), |b| {
            b.iter_batched(
                || {
                    let (reals, imags) = generate_re_im_f64(len);
                    let output = vec![0.0f64; len * 2];
                    (reals, imags, output)
                },
                |(reals, imags, mut output)| combine_re_im_into(&reals, &imags, &mut output),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_deinterleave_f32,
    benchmark_deinterleave_f64,
    benchmark_combine_re_im_f32,
    benchmark_combine_re_im_f64,
);
criterion_main!(benches);
