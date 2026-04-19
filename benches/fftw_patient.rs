use std::ptr::slice_from_raw_parts_mut;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use fftw::array::AlignedVec;
use fftw::plan::{C2CPlan, C2CPlan32, C2CPlan64};
use fftw::types::{Flag, Sign};
use num_traits::Float;
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rand::RngExt;
use utilities::rustfft::num_complex::Complex;

// IMPORTANT NOTE:
// This benchmark only measures small-to-mid sizes, which are not the focus of
// PhastFT. Criterion is not a good fit for measuring long-running tasks — see
// examples/benchmark.rs for the benchmarking harness for large sizes.
//
// Each FFTW planning mode (Estimate / Measure / Patient) lives in its own
// `[[bench]]` binary so FFTW's global per-process wisdom cache cannot leak
// between modes; each run starts with a fresh process and empty wisdom. The
// benchmark group names are shared with benches/bench.rs ("Forward f32",
// "Forward f64") so criterion's HTML report auto-overlays the PhastFT, RustFFT,
// and three FFTW series at `target/criterion/Forward <precision>/report/`.
//
// PATIENT is FFTW's most thorough planner; at sizes beyond ~2^18 its planning
// time grows from seconds to minutes. LENGTHS is therefore capped at 2^18 to
// keep a full `cargo bench --bench fftw_patient` run tractable. ESTIMATE and
// MEASURE use the full bench.rs range.

const LENGTHS: &[usize] = &[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];

// Distribution parity with benches/bench.rs::generate_numbers (bench.rs:23-47).
// Do NOT substitute utilities::gen_random_signal_f* — those use Uniform(-1, 1)
// followed by L2 normalization, which would bias the FFTW series relative to
// the PhastFT/RustFFT series that share the same criterion group.
fn generate_numbers<T: Float>(n: usize) -> (Vec<T>, Vec<T>)
where
    StandardUniform: Distribution<T>,
{
    let mut rng = rand::make_rng::<SmallRng>();

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

        let id = "FFTW Patient";
        // Plan construction is outside iter_batched so planning cost is
        // excluded from per-sample timings — matches how bench.rs excludes
        // RustFFT's FftPlanner construction.
        let mut plan =
            C2CPlan32::aligned(&[len], Sign::Forward, Flag::DESTROYINPUT | Flag::PATIENT).unwrap();

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || {
                    let (reals, imags) = generate_numbers::<f32>(len);
                    let mut nums: AlignedVec<Complex<f32>> = AlignedVec::new(len);
                    for (z, (&re, &im)) in nums.iter_mut().zip(reals.iter().zip(imags.iter())) {
                        *z = Complex::new(re, im);
                    }
                    nums
                },
                |mut nums| {
                    plan.c2c(
                        // SAFETY: identical shape to examples/fftwrb.rs:42-48.
                        // DESTROYINPUT permits in-place c2c; the raw slice aliases
                        // `nums` for the duration of the call and `len` matches
                        // the AlignedVec allocation.
                        unsafe { &mut *slice_from_raw_parts_mut(nums.as_mut_ptr(), len) },
                        &mut nums,
                    )
                    .unwrap();
                    std::hint::black_box(&mut nums);
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

        let id = "FFTW Patient";
        let mut plan =
            C2CPlan64::aligned(&[len], Sign::Forward, Flag::DESTROYINPUT | Flag::PATIENT).unwrap();

        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter_batched(
                || {
                    let (reals, imags) = generate_numbers::<f64>(len);
                    let mut nums: AlignedVec<Complex<f64>> = AlignedVec::new(len);
                    for (z, (&re, &im)) in nums.iter_mut().zip(reals.iter().zip(imags.iter())) {
                        *z = Complex::new(re, im);
                    }
                    nums
                },
                |mut nums| {
                    plan.c2c(
                        // SAFETY: see benchmark_forward_f32 above.
                        unsafe { &mut *slice_from_raw_parts_mut(nums.as_mut_ptr(), len) },
                        &mut nums,
                    )
                    .unwrap();
                    std::hint::black_box(&mut nums);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_forward_f32, benchmark_forward_f64);
criterion_main!(benches);
