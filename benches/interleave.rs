use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use phastft::complex_nums::{
    combine_re_im, deinterleave, deinterleave_inplace, interleave_inplace,
};
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::RngExt;

// This benchmark requires the bench-internals feature to access private functions.
// Run it with:
// cargo bench --bench interleave --features bench-internals

const LENGTHS: &[usize] = &[
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
];

fn generate_interleaved_f32(n: usize) -> Vec<f32> {
    let mut rng = rand::make_rng::<SmallRng>();
    (&mut rng)
        .sample_iter(StandardUniform)
        .take(2 * n)
        .collect()
}

fn generate_interleaved_f64(n: usize) -> Vec<f64> {
    let mut rng = rand::make_rng::<SmallRng>();
    (&mut rng)
        .sample_iter(StandardUniform)
        .take(2 * n)
        .collect()
}

fn generate_re_im_f32(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::make_rng::<SmallRng>();
    let reals: Vec<f32> = (&mut rng).sample_iter(StandardUniform).take(n).collect();
    let imags: Vec<f32> = (&mut rng).sample_iter(StandardUniform).take(n).collect();
    (reals, imags)
}

fn generate_re_im_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::make_rng::<SmallRng>();
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
            let _level = fearless_simd::Level::new();
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
            let _level = fearless_simd::Level::new();
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
                || generate_re_im_f32(len),
                |(reals, imags)| combine_re_im::<f32>(&reals, &imags),
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
                || generate_re_im_f64(len),
                |(reals, imags)| combine_re_im::<f64>(&reals, &imags),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

const BLOCK_SIZE_BYTES: usize = 8 * 1024;

fn benchmark_interleave_inplace_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Interleave inplace f32");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    let block_size = BLOCK_SIZE_BYTES / size_of::<f32>();

    for n in LENGTHS.iter() {
        let len = 1 << n;
        if len < block_size * 2 {
            continue;
        }
        group.throughput(Throughput::Bytes((len * 2 * size_of::<f32>()) as u64));

        group.bench_function(BenchmarkId::new("interleave_inplace", len), |b| {
            b.iter_batched(
                || generate_interleaved_f32(len),
                |mut data| interleave_inplace(&mut data, block_size),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_interleave_inplace_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("Interleave inplace f64");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    let block_size = BLOCK_SIZE_BYTES / size_of::<f64>();

    for n in LENGTHS.iter() {
        let len = 1 << n;
        if len < block_size * 2 {
            continue;
        }
        group.throughput(Throughput::Bytes((len * 2 * size_of::<f64>()) as u64));

        group.bench_function(BenchmarkId::new("interleave_inplace", len), |b| {
            b.iter_batched(
                || generate_interleaved_f64(len),
                |mut data| interleave_inplace(&mut data, block_size),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_deinterleave_inplace_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Deinterleave inplace f32");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    let block_size = BLOCK_SIZE_BYTES / size_of::<f32>();

    for n in LENGTHS.iter() {
        let len = 1 << n;
        if len < block_size * 2 {
            continue;
        }
        group.throughput(Throughput::Bytes((len * 2 * size_of::<f32>()) as u64));

        group.bench_function(BenchmarkId::new("deinterleave_inplace", len), |b| {
            b.iter_batched(
                || generate_interleaved_f32(len),
                |mut data| deinterleave_inplace(&mut data, block_size),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_deinterleave_inplace_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("Deinterleave inplace f64");
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    let block_size = BLOCK_SIZE_BYTES / size_of::<f64>();

    for n in LENGTHS.iter() {
        let len = 1 << n;
        if len < block_size * 2 {
            continue;
        }
        group.throughput(Throughput::Bytes((len * 2 * size_of::<f64>()) as u64));

        group.bench_function(BenchmarkId::new("deinterleave_inplace", len), |b| {
            b.iter_batched(
                || generate_interleaved_f64(len),
                |mut data| deinterleave_inplace(&mut data, block_size),
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
    benchmark_interleave_inplace_f32,
    benchmark_interleave_inplace_f64,
    benchmark_deinterleave_inplace_f32,
    benchmark_deinterleave_inplace_f64,
);
criterion_main!(benches);
