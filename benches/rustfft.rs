use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use num_traits::{Float, Zero};
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rand::RngExt;
use utilities::rustfft::num_complex::Complex;
use utilities::rustfft::FftPlanner;

// IMPORTANT NOTE:
// This benchmark only measures small-to-mid sizes, which are not the focus of
// PhastFT. Criterion is not a good fit for measuring long-running tasks — see
// examples/benchmark.rs for the benchmarking harness for large sizes.
//
// The RustFFT series lives in its own `[[bench]]` binary so it can be run
// independently of the PhastFT series. The benchmark group names are shared
// with benches/bench.rs ("Forward f32", "Inverse f32", "Forward f64",
// "Inverse f64") so that each bench binary writes into the same
// `target/criterion/<group>/<id>/<size>/` tree. Criterion itself does NOT
// auto-aggregate across bench binaries — the last bench to finish overwrites
// `target/criterion/<group>/report/{lines,violin}.svg` with a report that
// mentions only its own IDs. Use `benches/plot_criterion_overlay.py` to
// produce a single overlay plot per group after running all five benches.

const LENGTHS: &[usize] = &[
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
];

// Distribution parity with benches/bench.rs::generate_numbers (StandardUniform
// over [0, 1) with SmallRng). Each sample is an interleaved Complex<T>.
fn generate_complex_numbers<T: Float + Default>(n: usize) -> Vec<Complex<T>>
where
    StandardUniform: Distribution<T>,
{
    let mut rng = rand::make_rng::<SmallRng>();

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
            bytes: (2 * len * size_of::<f32>()) as u64,
        });

        let id = "RustFFT";
        // Plan and scratch allocation happen outside iter_batched — planning
        // cost is excluded from per-sample timings.
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(len);
        let mut scratch = vec![Complex::<f32>::zero(); fft.get_inplace_scratch_len()];

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_complex_numbers::<f32>(len),
                |mut signal| {
                    fft.process_with_scratch(&mut signal, &mut scratch);
                    std::hint::black_box(&mut signal);
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
            bytes: (2 * len * size_of::<f32>()) as u64,
        });

        let id = "RustFFT";
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_inverse(len);
        let mut scratch = vec![Complex::<f32>::zero(); fft.get_inplace_scratch_len()];

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_complex_numbers::<f32>(len),
                |mut signal| {
                    fft.process_with_scratch(&mut signal, &mut scratch);
                    std::hint::black_box(&mut signal);
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
            bytes: (2 * len * size_of::<f64>()) as u64,
        });

        let id = "RustFFT";
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(len);
        let mut scratch = vec![Complex::<f64>::zero(); fft.get_inplace_scratch_len()];

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_complex_numbers::<f64>(len),
                |mut signal| {
                    fft.process_with_scratch(&mut signal, &mut scratch);
                    std::hint::black_box(&mut signal);
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
            bytes: (2 * len * size_of::<f64>()) as u64,
        });

        let id = "RustFFT";
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_inverse(len);
        let mut scratch = vec![Complex::<f64>::zero(); fft.get_inplace_scratch_len()];

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || generate_complex_numbers::<f64>(len),
                |mut signal| {
                    fft.process_with_scratch(&mut signal, &mut scratch);
                    std::hint::black_box(&mut signal);
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
