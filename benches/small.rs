//! Small C2C transforms, including normalized inverse transforms for both libraries.
//! Plans and RustFFT scratch are reused; input setup is outside the timed region.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use phastft::options::Options;
use phastft::planner::{Direction, PlannerDit32, PlannerDit64};
use phastft::{fft_f32_dit_with_planner_and_opts, fft_f64_dit_with_planner_and_opts};
use utilities::rustfft::{num_complex::Complex, FftPlanner};

mod common;
use common::{interleaved_complex, split_complex, transform_batch_size};

macro_rules! benches {
    ($name:ident, $float:ty, $planner:ty, $fft:ident) => {
        fn $name(c: &mut Criterion) {
            for direction in [Direction::Forward, Direction::Inverse] {
                let mut group =
                    c.benchmark_group(format!("small_{direction:?}_{}", stringify!($float)));
                for log_n in 0..=10 {
                    let n = 1 << log_n;
                    let planner = <$planner>::new(n);
                    let opts = Options::guess_options(n);
                    let batch = transform_batch_size(2 * n * size_of::<$float>());
                    group.bench_function(BenchmarkId::new("PhastFT", n), |b| {
                        b.iter_batched_ref(
                            || split_complex::<$float>(n),
                            |(re, im)| {
                                $fft(re, im, direction, &planner, &opts);
                                std::hint::black_box((re, im));
                            },
                            batch,
                        );
                    });
                    let mut reference = FftPlanner::<$float>::new();
                    let fft = match direction {
                        Direction::Forward => reference.plan_fft_forward(n),
                        Direction::Inverse => reference.plan_fft_inverse(n),
                    };
                    let mut scratch = vec![Complex::default(); fft.get_inplace_scratch_len()];
                    group.bench_function(BenchmarkId::new("RustFFT", n), |b| {
                        b.iter_batched_ref(
                            || interleaved_complex::<$float>(n),
                            |signal| {
                                fft.process_with_scratch(signal, &mut scratch);
                                if direction == Direction::Inverse {
                                    let scale = 1.0 / n as $float;
                                    for z in signal.iter_mut() {
                                        *z *= scale;
                                    }
                                }
                                std::hint::black_box(signal);
                            },
                            batch,
                        );
                    });
                }
                group.finish();
            }
        }
    };
}

benches!(
    f32_benches,
    f32,
    PlannerDit32,
    fft_f32_dit_with_planner_and_opts
);
benches!(
    f64_benches,
    f64,
    PlannerDit64,
    fft_f64_dit_with_planner_and_opts
);
criterion_group!(benches, f32_benches, f64_benches);
criterion_main!(benches);
