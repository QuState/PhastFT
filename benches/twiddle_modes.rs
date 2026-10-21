//! Compare explicit twiddle modes without timing planning or input generation.
//! Separate serial/default-threading groups make the Auto crossover measurable.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use phastft::options::Options;
use phastft::planner::{
    Direction, PlannerDit32, PlannerDit64, PlannerR2c32, PlannerR2c64, TwiddleMode,
};
use phastft::{
    c2r_fft_f32_with_planner_and_opts, c2r_fft_f64_with_planner_and_opts,
    fft_f32_dit_with_planner_and_opts, fft_f64_dit_with_planner_and_opts,
    r2c_fft_f32_with_planner_and_opts, r2c_fft_f64_with_planner_and_opts,
};

mod common;
use common::{
    bench_transforms_at_sizes, real_signal, spectrum_split, split_complex, throughput_complex,
    throughput_real, transform_batch_size, LENGTHS,
};

macro_rules! bench_modes {
    ($name:ident, $float:ty, $planner:ident, $fft:ident) => {
        fn $name(c: &mut Criterion) {
            for parallel in [false, true] {
                if parallel && !cfg!(feature = "parallel") {
                    continue;
                }
                let threading = if parallel { "parallel" } else { "serial" };
                for direction in [Direction::Forward, Direction::Inverse] {
                    let group =
                        format!("twiddles_{}_{direction:?}_{threading}", stringify!($float))
                            .to_lowercase();
                    bench_transforms_at_sizes(
                        c,
                        &group,
                        LENGTHS,
                        throughput_complex::<$float>,
                        |g, n| {
                            let mut opts = Options::guess_options(n);
                            if !parallel {
                                opts.multithreaded_bit_reversal = false;
                                opts.smallest_parallel_chunk_size = usize::MAX;
                            }
                            for mode in [TwiddleMode::Precomputed, TwiddleMode::LowMemory] {
                                let planner = $planner::with_twiddle_mode(n, mode);
                                g.bench_function(BenchmarkId::new(format!("{mode:?}"), n), |b| {
                                    b.iter_batched_ref(
                                        || split_complex::<$float>(n),
                                        |(re, im)| {
                                            $fft(re, im, direction, &planner, &opts);
                                            std::hint::black_box((re, im));
                                        },
                                        transform_batch_size(2 * n * size_of::<$float>()),
                                    );
                                });
                            }
                        },
                    );
                }
            }
        }
    };
}

bench_modes!(
    f32_modes,
    f32,
    PlannerDit32,
    fft_f32_dit_with_planner_and_opts
);
bench_modes!(
    f64_modes,
    f64,
    PlannerDit64,
    fft_f64_dit_with_planner_and_opts
);

macro_rules! bench_real_modes {
    ($name:ident, $float:ty, $planner:ident, $r2c:ident, $c2r:ident) => {
        fn $name(c: &mut Criterion) {
            for parallel in [false, true] {
                if parallel && !cfg!(feature = "parallel") {
                    continue;
                }
                let threading = if parallel { "parallel" } else { "serial" };
                for direction in [Direction::Forward, Direction::Inverse] {
                    let group = format!(
                        "twiddles_real_{}_{direction:?}_{threading}",
                        stringify!($float)
                    )
                    .to_lowercase();
                    bench_transforms_at_sizes(
                        c,
                        &group,
                        LENGTHS,
                        throughput_real::<$float>,
                        |g, n| {
                            let mut opts = Options::guess_options(n / 2);
                            if !parallel {
                                opts.multithreaded_bit_reversal = false;
                                opts.smallest_parallel_chunk_size = usize::MAX;
                            }
                            for mode in [TwiddleMode::Precomputed, TwiddleMode::LowMemory] {
                                let planner = $planner::with_twiddle_mode(n, mode);
                                let mut scratch_re = vec![0.0; n / 2];
                                let mut scratch_im = vec![0.0; n / 2];
                                g.bench_function(BenchmarkId::new(format!("{mode:?}"), n), |b| {
                                    match direction {
                                        Direction::Forward => b.iter_batched_ref(
                                            || {
                                                (
                                                    real_signal::<$float>(n),
                                                    vec![0.0; n / 2 + 1],
                                                    vec![0.0; n / 2 + 1],
                                                )
                                            },
                                            |(input, re, im)| {
                                                $r2c(input, re, im, &planner, &opts);
                                                std::hint::black_box((re, im));
                                            },
                                            transform_batch_size((2 * n + 2) * size_of::<$float>()),
                                        ),
                                        Direction::Inverse => b.iter_batched_ref(
                                            || (spectrum_split::<$float>(n), vec![0.0; n]),
                                            |((re, im), output)| {
                                                $c2r(
                                                    re,
                                                    im,
                                                    output,
                                                    &planner,
                                                    &opts,
                                                    &mut scratch_re,
                                                    &mut scratch_im,
                                                );
                                                std::hint::black_box(output);
                                            },
                                            transform_batch_size((2 * n + 2) * size_of::<$float>()),
                                        ),
                                    }
                                });
                            }
                        },
                    );
                }
            }
        }
    };
}

bench_real_modes!(
    real_f32_modes,
    f32,
    PlannerR2c32,
    r2c_fft_f32_with_planner_and_opts,
    c2r_fft_f32_with_planner_and_opts
);
bench_real_modes!(
    real_f64_modes,
    f64,
    PlannerR2c64,
    r2c_fft_f64_with_planner_and_opts,
    c2r_fft_f64_with_planner_and_opts
);

criterion_group!(
    benches,
    f32_modes,
    f64_modes,
    real_f32_modes,
    real_f64_modes
);
criterion_main!(benches);
