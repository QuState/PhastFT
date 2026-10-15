//! Important: this benchmark only measures small-to-mid sizes; criterion is
//! not a good fit for measuring long-running tasks — see
//! `examples/benchmark.rs` for the harness for large sizes.
//!
//! The RustFFT series lives in its own `[[bench]]` binary so it can be
//! re-run independently. Group names are shared with `bench.rs` /
//! `fftw_*.rs`; criterion does NOT auto-aggregate across binaries — use
//! `benches/plot_criterion_overlay.py` for the cross-binary overlay.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use num_traits::Zero;
use utilities::rustfft::num_complex::Complex;
use utilities::rustfft::FftPlanner;

mod common;
use common::{
    bench_transforms_at_sizes, groups, ids, interleaved_complex, throughput_complex,
    transform_batch_size, LENGTHS,
};

macro_rules! rustfft_c2c {
    ($name:ident, $float:ty, $plan_method:ident, $group:expr) => {
        fn $name(c: &mut Criterion) {
            bench_transforms_at_sizes(
                c,
                $group,
                LENGTHS,
                throughput_complex::<$float>,
                |g, len| {
                    // Plan and scratch are constructed outside iter_batched_ref so
                    // planning cost is excluded from per-sample timings.
                    let mut planner = FftPlanner::<$float>::new();
                    let fft = planner.$plan_method(len);
                    let mut scratch =
                        vec![Complex::<$float>::zero(); fft.get_inplace_scratch_len()];
                    g.bench_function(BenchmarkId::new(ids::RUSTFFT, len), |b| {
                        b.iter_batched_ref(
                            || interleaved_complex::<$float>(len),
                            |signal| {
                                fft.process_with_scratch(signal, &mut scratch);
                                std::hint::black_box(signal);
                            },
                            transform_batch_size(2 * len * size_of::<$float>()),
                        );
                    });
                },
            );
        }
    };
}

rustfft_c2c!(fwd_f32, f32, plan_fft_forward, groups::C2C_FORWARD_F32);
rustfft_c2c!(inv_f32, f32, plan_fft_inverse, groups::C2C_INVERSE_F32);
rustfft_c2c!(fwd_f64, f64, plan_fft_forward, groups::C2C_FORWARD_F64);
rustfft_c2c!(inv_f64, f64, plan_fft_inverse, groups::C2C_INVERSE_F64);

criterion_group!(benches, fwd_f32, inv_f32, fwd_f64, inv_f64);
criterion_main!(benches);
