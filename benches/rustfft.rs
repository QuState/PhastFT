//! Important: this benchmark only measures small-to-mid sizes; criterion is
//! not a good fit for measuring long-running tasks — see
//! `examples/benchmark.rs` for the harness for large sizes.
//!
//! The RustFFT series lives in its own `[[bench]]` binary so it can be
//! re-run independently. Group names are shared with `bench.rs` /
//! `fftw_*.rs`; criterion does NOT auto-aggregate across binaries — use
//! `benches/plot_criterion_overlay.py` for the cross-binary overlay.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use num_traits::Zero;
use utilities::rustfft::num_complex::Complex;
use utilities::rustfft::FftPlanner;

mod common;
use common::{bench_at_sizes, groups, ids, interleaved_complex, throughput_complex, LENGTHS};

macro_rules! rustfft_c2c {
    ($name:ident, $float:ty, $plan_method:ident, $group:expr) => {
        fn $name(c: &mut Criterion) {
            bench_at_sizes(
                c,
                $group,
                LENGTHS,
                throughput_complex::<$float>,
                |g, len| {
                    // Plan and scratch are constructed outside iter_batched so
                    // planning cost is excluded from per-sample timings.
                    let mut planner = FftPlanner::<$float>::new();
                    let fft = planner.$plan_method(len);
                    let mut scratch =
                        vec![Complex::<$float>::zero(); fft.get_inplace_scratch_len()];
                    g.bench_function(BenchmarkId::new(ids::RUSTFFT, len), |b| {
                        b.iter_batched(
                            || interleaved_complex::<$float>(len),
                            |mut signal| {
                                fft.process_with_scratch(&mut signal, &mut scratch);
                                std::hint::black_box(&mut signal);
                            },
                            BatchSize::SmallInput,
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
