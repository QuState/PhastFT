//! Compare PhastFT and RustFFT's Bluestein implementations.
//!
//! RustFFT accepts any inner length `M >= 2N - 1`. This benchmark gives both
//! implementations the same power-of-two `M`, so they perform equivalent
//! convolution work. Plans and scratch buffers are reused in both cases.
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use num_traits::Zero;
use phastft::options::Options;
use phastft::planner::{Direction, PlannerBluestein32, PlannerBluestein64};
use phastft::{fft_f32_bluestein_with_planner_and_opts, fft_f64_bluestein_with_planner_and_opts};
use utilities::rustfft::algorithm::BluesteinsAlgorithm;
use utilities::rustfft::num_complex::Complex;
use utilities::rustfft::{Fft, FftPlanner};

mod common;
use common::{
    groups, ids, interleaved_complex, make_group, split_complex, throughput_complex,
    BLUESTEIN_LENGTHS,
};

macro_rules! bluestein_compare {
    ($name:ident, $T:ty, $planner:ty, $phastft_fn:ident, $dir:expr, $rustfft_plan:ident, $group:expr) => {
        fn $name(c: &mut Criterion) {
            let mut group = make_group(c, $group);
            for &len in BLUESTEIN_LENGTHS {
                group.throughput(throughput_complex::<$T>(len));

                let planner = <$planner>::new(len);
                let m = planner.scratch_len();
                let opts = Options::guess_options(m);
                let mut scratch_re = vec![0.0; m];
                let mut scratch_im = vec![0.0; m];
                group.bench_function(BenchmarkId::new(ids::PHASTFT_BLUESTEIN, len), |b| {
                    b.iter_batched(
                        || split_complex::<$T>(len),
                        |(mut re, mut im)| {
                            $phastft_fn(
                                &mut re,
                                &mut im,
                                $dir,
                                &planner,
                                &opts,
                                &mut scratch_re,
                                &mut scratch_im,
                            );
                            std::hint::black_box((&mut re, &mut im));
                        },
                        BatchSize::SmallInput,
                    );
                });

                let inner = FftPlanner::<$T>::new().$rustfft_plan(m);
                let fft = BluesteinsAlgorithm::new(len, inner);
                let mut scratch = vec![Complex::<$T>::zero(); fft.get_inplace_scratch_len()];
                group.bench_function(BenchmarkId::new(ids::RUSTFFT_BLUESTEIN, len), |b| {
                    b.iter_batched(
                        || interleaved_complex::<$T>(len),
                        |mut signal| {
                            fft.process_with_scratch(&mut signal, &mut scratch);
                            // RustFFT leaves inverse transforms unnormalized;
                            // PhastFT applies 1/N, so include the same work here.
                            if matches!($dir, Direction::Inverse) {
                                let scale = 1.0 as $T / len as $T;
                                for value in &mut signal {
                                    value.re *= scale;
                                    value.im *= scale;
                                }
                            }
                            std::hint::black_box(&mut signal);
                        },
                        BatchSize::SmallInput,
                    );
                });
            }
            group.finish();
        }
    };
}

bluestein_compare!(
    fwd_f64,
    f64,
    PlannerBluestein64,
    fft_f64_bluestein_with_planner_and_opts,
    Direction::Forward,
    plan_fft_forward,
    groups::C2C_BLUESTEIN_FORWARD_F64
);
bluestein_compare!(
    inv_f64,
    f64,
    PlannerBluestein64,
    fft_f64_bluestein_with_planner_and_opts,
    Direction::Inverse,
    plan_fft_inverse,
    groups::C2C_BLUESTEIN_INVERSE_F64
);
bluestein_compare!(
    fwd_f32,
    f32,
    PlannerBluestein32,
    fft_f32_bluestein_with_planner_and_opts,
    Direction::Forward,
    plan_fft_forward,
    groups::C2C_BLUESTEIN_FORWARD_F32
);
bluestein_compare!(
    inv_f32,
    f32,
    PlannerBluestein32,
    fft_f32_bluestein_with_planner_and_opts,
    Direction::Inverse,
    plan_fft_inverse,
    groups::C2C_BLUESTEIN_INVERSE_F32
);

criterion_group!(benches, fwd_f64, inv_f64, fwd_f32, inv_f32);
criterion_main!(benches);
