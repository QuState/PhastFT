#![doc = include_str!("../README.md")]
#![warn(
    missing_docs,
    clippy::complexity,
    clippy::perf,
    clippy::style,
    clippy::correctness,
    clippy::suspicious
)]
#![forbid(unsafe_code)]

#[cfg(feature = "complex-nums")]
use num_complex::Complex;

#[cfg(feature = "complex-nums")]
use crate::complex_nums::{combine_re_im, deinterleave_complex32, deinterleave_complex64};
use crate::options::Options;
use crate::planner::{Direction, PlannerDit32, PlannerDit64};

#[cfg(not(feature = "bench-internals"))]
mod algorithms;
#[cfg(feature = "bench-internals")]
pub mod algorithms;
#[cfg(all(feature = "complex-nums", not(feature = "bench-internals")))]
mod complex_nums;
#[cfg(feature = "bench-internals")]
pub mod complex_nums;
mod kernels;
pub mod options;
mod parallel;
pub mod planner;
mod twiddles;

pub use algorithms::dit::{fft_f32_dit_with_planner_and_opts, fft_f64_dit_with_planner_and_opts};
pub use algorithms::r2c::{
    c2r_fft_f32, c2r_fft_f32_with_planner, c2r_fft_f32_with_planner_and_opts, c2r_fft_f64,
    c2r_fft_f64_with_planner, c2r_fft_f64_with_planner_and_opts, r2c_fft_f32,
    r2c_fft_f32_with_planner, r2c_fft_f32_with_planner_and_opts, r2c_fft_f64,
    r2c_fft_f64_with_planner, r2c_fft_f64_with_planner_and_opts,
};

#[cfg(feature = "complex-nums")]
macro_rules! impl_fft_interleaved_for {
    ($func_name:ident, $precision:ty, $fft_func:ident, $deinterleaving_func: ident, $planner:ty) => {
        /// FFT Interleaved — alternative entry point when the input data is a
        /// slice of [`Complex`]. Analogous to the
        /// `fft_*_dit_with_planner_and_opts` family except for the input format.
        ///
        /// **Note**: This function has to make a deinterleaved copy of the data.
        /// For maximum performance with minimal memory usage, use the split-array
        /// `fft_*_dit_with_planner_and_opts` API directly.
        pub fn $func_name(
            signal: &mut [Complex<$precision>],
            direction: Direction,
            planner: &$planner,
            opts: &Options,
        ) {
            let (mut reals, mut imags) = $deinterleaving_func(signal);
            $fft_func(&mut reals, &mut imags, direction, planner, opts);
            signal.copy_from_slice(&combine_re_im(&reals, &imags))
        }
    };
}

#[cfg(feature = "complex-nums")]
impl_fft_interleaved_for!(
    fft_f32_dit_interleaved_with_planner_and_opts,
    f32,
    fft_f32_dit_with_planner_and_opts,
    deinterleave_complex32,
    PlannerDit32
);
#[cfg(feature = "complex-nums")]
impl_fft_interleaved_for!(
    fft_f64_dit_interleaved_with_planner_and_opts,
    f64,
    fft_f64_dit_with_planner_and_opts,
    deinterleave_complex64,
    PlannerDit64
);

#[cfg(feature = "complex-nums")]
macro_rules! impl_fft_interleaved_with_planner {
    ($func_name:ident, $precision:ty, $fft_with_opts_func:ident, $planner:ty) => {
        /// FFT Interleaved with pre-computed planner -- convenience wrapper around
        /// the `_with_planner_and_opts` variant that automatically guesses options.
        ///
        /// For better control over options, use the `_with_planner_and_opts` variant.
        pub fn $func_name(
            signal: &mut [Complex<$precision>],
            direction: Direction,
            planner: &$planner,
        ) {
            let opts = Options::guess_options(signal.len());
            $fft_with_opts_func(signal, direction, planner, &opts);
        }
    };
}

#[cfg(feature = "complex-nums")]
impl_fft_interleaved_with_planner!(
    fft_f32_dit_interleaved_with_planner,
    f32,
    fft_f32_dit_interleaved_with_planner_and_opts,
    PlannerDit32
);
#[cfg(feature = "complex-nums")]
impl_fft_interleaved_with_planner!(
    fft_f64_dit_interleaved_with_planner,
    f64,
    fft_f64_dit_interleaved_with_planner_and_opts,
    PlannerDit64
);

#[cfg(feature = "complex-nums")]
macro_rules! impl_fft_interleaved {
    ($func_name:ident, $precision:ty, $fft_with_planner_func:ident, $planner:ty) => {
        /// FFT Interleaved -- convenience wrapper that creates a planner automatically.
        ///
        /// For better performance when running multiple FFTs of the same size,
        /// consider using the `_with_planner` variant.
        pub fn $func_name(signal: &mut [Complex<$precision>], direction: Direction) {
            let planner = <$planner>::new(signal.len());
            $fft_with_planner_func(signal, direction, &planner);
        }
    };
}

#[cfg(feature = "complex-nums")]
impl_fft_interleaved!(
    fft_f32_dit_interleaved,
    f32,
    fft_f32_dit_interleaved_with_planner,
    PlannerDit32
);
#[cfg(feature = "complex-nums")]
impl_fft_interleaved!(
    fft_f64_dit_interleaved,
    f64,
    fft_f64_dit_interleaved_with_planner,
    PlannerDit64
);

/// FFT using the Decimation-In-Time (DIT) algorithm for `f64`, reusing a
/// pre-computed planner. Options are guessed from the input size.
///
/// For full control over [`Options`], use [`fft_f64_dit_with_planner_and_opts`].
///
/// # Panics
///
/// Panics if the input length is not a power of two, or does not match the planner size.
pub fn fft_f64_dit_with_planner(
    reals: &mut [f64],
    imags: &mut [f64],
    direction: Direction,
    planner: &PlannerDit64,
) {
    let opts = Options::guess_options(reals.len());
    algorithms::dit::fft_f64_dit_with_planner_and_opts(reals, imags, direction, planner, &opts);
}

/// FFT using Decimation-In-Time (DIT) algorithm for f64.
///
/// This is a convenient wrapper that creates a planner automatically.
/// For better performance when running multiple FFTs of the same size,
/// consider using [`fft_f64_dit_with_planner`].
///
/// # Arguments
///
/// * `reals` - Real parts of the complex numbers (modified in-place)
/// * `imags` - Imaginary parts of the complex numbers (modified in-place)
/// * `direction` - Forward or inverse transform
///
/// # Panics
///
/// Panics if the input length is not a power of 2.
///
/// # Example
///
/// ```
/// use phastft::{fft_f64_dit, planner::Direction};
///
/// let mut reals = vec![1.0, 0.0, 0.0, 0.0];
/// let mut imags = vec![0.0; 4];
/// fft_f64_dit(&mut reals, &mut imags, Direction::Forward);
/// // Output is in normal order
/// ```
///
pub fn fft_f64_dit(reals: &mut [f64], imags: &mut [f64], direction: Direction) {
    let planner = PlannerDit64::new(reals.len());
    fft_f64_dit_with_planner(reals, imags, direction, &planner);
}

/// FFT using the Decimation-In-Time (DIT) algorithm for `f32`, reusing a
/// pre-computed planner. Options are guessed from the input size.
///
/// For full control over [`Options`], use [`fft_f32_dit_with_planner_and_opts`].
///
/// # Panics
///
/// Panics if the input length is not a power of two, or does not match the planner size.
pub fn fft_f32_dit_with_planner(
    reals: &mut [f32],
    imags: &mut [f32],
    direction: Direction,
    planner: &PlannerDit32,
) {
    let opts = Options::guess_options(reals.len());
    fft_f32_dit_with_planner_and_opts(reals, imags, direction, planner, &opts);
}

/// FFT using Decimation-In-Time (DIT) algorithm for f32.
///
/// This is a convenient wrapper that creates a planner automatically.
/// For better performance when running multiple FFTs of the same size,
/// consider using [`fft_f32_dit_with_planner`].
///
/// # Arguments
///
/// * `reals` - Real parts of the complex numbers (modified in-place)
/// * `imags` - Imaginary parts of the complex numbers (modified in-place)
/// * `direction` - Forward or inverse transform
///
/// # Panics
///
/// Panics if the input length is not a power of 2.
///
/// # Example
///
/// ```
/// use phastft::{fft_f32_dit, planner::Direction};
///
/// let mut reals = vec![1.0, 0.0, 0.0, 0.0];
/// let mut imags = vec![0.0; 4];
/// fft_f32_dit(&mut reals, &mut imags, Direction::Forward);
/// // Output is in normal order
/// ```
///
pub fn fft_f32_dit(reals: &mut [f32], imags: &mut [f32], direction: Direction) {
    let planner = PlannerDit32::new(reals.len());
    fft_f32_dit_with_planner(reals, imags, direction, &planner);
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use utilities::rustfft::num_complex::Complex;
    use utilities::rustfft::FftPlanner;
    use utilities::{assert_float_closeness, gen_random_signal_f32, gen_random_signal_f64};

    use super::*;

    macro_rules! non_power_of_2_planner {
        ($test_name:ident, $planner:ty) => {
            #[should_panic]
            #[test]
            fn $test_name() {
                let num_points = 5;

                // this test _should_ always fail at this stage
                let _ = <$planner>::new(num_points);
            }
        };
    }

    non_power_of_2_planner!(non_power_of_2_planner_32, PlannerDit32);
    non_power_of_2_planner!(non_power_of_2_planner_64, PlannerDit64);

    macro_rules! wrong_num_points_in_planner {
        ($test_name:ident, $planner:ty, $fft_with_opts_and_plan:ident) => {
            // A regression test to make sure the `Planner` is compatible with fft execution.
            #[should_panic]
            #[test]
            fn $test_name() {
                let n = 16;
                let num_points = 1 << n; // 2.pow(n)

                // We purposely set n = 16 and pass it to the planner.
                // n == 16 == 2^{4} is clearly a power of two, so the planner won't throw it out.
                // However, the call to `fft_with_opts_and_plan` should panic since it tests that the
                // planner size matches the input size.
                // In this case, we have an input of size 1024 (used for mp3), but we tell the planner the
                // input size is 16.
                let mut planner = <$planner>::new(n);

                let mut reals = vec![0.0; num_points];
                let mut imags = vec![0.0; num_points];
                let opts = Options::guess_options(reals.len());

                // this call should panic
                $fft_with_opts_and_plan(
                    &mut reals,
                    &mut imags,
                    Direction::Forward,
                    &mut planner,
                    &opts,
                );
            }
        };
    }

    wrong_num_points_in_planner!(
        wrong_num_points_in_planner_32,
        PlannerDit32,
        fft_f32_dit_with_planner_and_opts
    );
    wrong_num_points_in_planner!(
        wrong_num_points_in_planner_64,
        PlannerDit64,
        fft_f64_dit_with_planner_and_opts
    );

    macro_rules! test_fft_correctness {
        ($test_name:ident, $precision:ty, $fft_type:ident, $range_start:literal, $range_end:literal) => {
            #[test]
            fn $test_name() {
                let range = Range {
                    start: $range_start,
                    end: $range_end,
                };

                for k in range {
                    let n: usize = 1 << k; // 2.pow(k)

                    let mut reals: Vec<$precision> = (1..=n).map(|i| i as $precision).collect();
                    let mut imags: Vec<$precision> = (1..=n).map(|i| i as $precision).collect();
                    $fft_type(&mut reals, &mut imags, Direction::Forward);

                    let mut buffer: Vec<Complex<$precision>> = (1..=n)
                        .map(|i| Complex::new(i as $precision, i as $precision))
                        .collect();

                    let mut planner = FftPlanner::new();
                    let fft = planner.plan_fft_forward(buffer.len());
                    fft.process(&mut buffer);

                    reals
                        .iter()
                        .zip(imags.iter())
                        .enumerate()
                        .for_each(|(i, (z_re, z_im))| {
                            let expect_re = buffer[i].re;
                            let expect_im = buffer[i].im;
                            assert_float_closeness(*z_re, expect_re, 0.01);
                            assert_float_closeness(*z_im, expect_im, 0.01);
                        });
                }
            }
        };
    }

    test_fft_correctness!(fft_correctness_32, f32, fft_f32_dit, 4, 9);
    test_fft_correctness!(fft_correctness_64, f64, fft_f64_dit, 4, 17);

    macro_rules! test_fft_stage_scheduling {
        ($name:ident, $float:ty, $planner:ty, $fft:ident, $tolerance:expr) => {
            #[test]
            fn $name() {
                let mut reference_planner = FftPlanner::<$float>::new();
                // Both exponent parities, across cache leaves, recursive merges,
                // and the default parallel threshold. Test each direction against
                // an independent FFT rather than relying only on a round trip.
                // 2^18 is the default fused-kernel cutoff; 2^19 is above it.
                for log_n in 0..=19 {
                    let n = 1 << log_n;
                    let input: Vec<_> = (0..n)
                        .map(|i| {
                            Complex::new(
                                ((i * 17 % 257) as $float - 128.0) / 128.0,
                                ((i * 31 % 251) as $float - 125.0) / 125.0,
                            )
                        })
                        .collect();
                    for mode in [crate::planner::TwiddleMode::Precomputed, crate::planner::TwiddleMode::LowMemory, crate::planner::TwiddleMode::Auto] {
                    let planner = <$planner>::with_twiddle_mode(n, mode);
                    let opts = Options::guess_options(n);
                    for direction in [Direction::Forward, Direction::Inverse] {
                        let mut expected = input.clone();
                        let reference = match direction {
                            Direction::Forward => reference_planner.plan_fft_forward(n),
                            Direction::Inverse => reference_planner.plan_fft_inverse(n),
                        };
                        reference.process(&mut expected);
                        if direction == Direction::Inverse {
                            for z in &mut expected {
                                *z /= n as $float;
                            }
                        }
                        // Also force and disable parallel FFT compute work at selected odd
                        // sizes and on both sides of the kernel cutoff. MAX
                        // exercises saturating threshold multiplication.
                        let thresholds = [opts.smallest_parallel_chunk_size, 0, usize::MAX];
                        let configurations = if matches!(log_n, 9 | 11 | 18 | 19) {
                            3
                        } else {
                            1
                        };
                        for threshold in thresholds.into_iter().take(configurations) {
                            let mut opts = opts.clone();
                            opts.smallest_parallel_chunk_size = threshold;
                            let mut re: Vec<_> = input.iter().map(|z| z.re).collect();
                            let mut im: Vec<_> = input.iter().map(|z| z.im).collect();
                            $fft(&mut re, &mut im, direction, &planner, &opts);

                            let mut error_sq = 0.0f64;
                            let mut reference_sq = 0.0f64;
                            for ((re, im), expected) in re.iter().zip(&im).zip(&expected) {
                                assert!(re.is_finite() && im.is_finite());
                                error_sq += (*re as f64 - expected.re as f64).powi(2)
                                    + (*im as f64 - expected.im as f64).powi(2);
                                reference_sq +=
                                    (expected.re as f64).powi(2) + (expected.im as f64).powi(2);
                            }
                            let relative_error = (error_sq / reference_sq).sqrt();
                            assert!(
                                relative_error < $tolerance,
                                "n={n}, {mode:?}, {direction:?}, cutoff={threshold}: relative L2 error {relative_error}"
                            );
                        }
                    }
                    }
                }
            }
        };
    }

    test_fft_stage_scheduling!(
        fft_stage_scheduling_f32,
        f32,
        PlannerDit32,
        fft_f32_dit_with_planner_and_opts,
        2e-6
    );
    test_fft_stage_scheduling!(
        fft_stage_scheduling_f64,
        f64,
        PlannerDit64,
        fft_f64_dit_with_planner_and_opts,
        1e-12
    );

    macro_rules! test_large_impulse_accuracy {
        ($name:ident, $float:ty, $planner:ty, $fft:ident, $l2_limit:expr, $max_limit:expr) => {
            #[test]
            fn $name() {
                // An impulse at N-1 has an analytic spectrum: exp(+2*pi*i*k/N).
                // It exercises every twiddle without using another FFT as the
                // oracle. Sizes span fine/coarse/anchor boundaries, both stage
                // parities, and parallel fused passes. Reuse the plan both ways.
                for log_n in [9, 11, 16, 19, 20] {
                    let n = 1 << log_n;
                    for mode in [crate::planner::TwiddleMode::Precomputed, crate::planner::TwiddleMode::LowMemory, crate::planner::TwiddleMode::Auto] {
                    let mut planner = <$planner>::with_twiddle_mode(n, mode);
                    // Also exercise the baseline backend (without FMA on x86).
                    if matches!(log_n, 11 | 16) {
                        planner.simd_level = fearless_simd::Level::baseline();
                    }
                    for threshold in [usize::MAX, 0] {
                        let mut opts = Options::default();
                        opts.smallest_parallel_chunk_size = threshold;
                        for direction in [Direction::Forward, Direction::Inverse] {
                            let mut re = vec![0.0 as $float; n];
                            let mut im = vec![0.0 as $float; n];
                            re[n - 1] = 1.0;
                            $fft(&mut re, &mut im, direction, &planner, &opts);
                            let mut error_sq = 0.0;
                            let mut max_error = 0.0_f64;
                            for k in 0..n {
                                let angle = std::f64::consts::TAU * k as f64 / n as f64;
                                let (mut expected_im, expected_re) = angle.sin_cos();
                                let scale = if direction == Direction::Inverse {
                                    expected_im = -expected_im;
                                    n as f64
                                } else {
                                    1.0
                                };
                                let error = (re[k] as f64 * scale - expected_re)
                                    .hypot(im[k] as f64 * scale - expected_im);
                                assert!(error.is_finite());
                                error_sq += error * error;
                                max_error = max_error.max(error);
                            }
                            let relative_l2 = (error_sq / n as f64).sqrt();
                            assert!(
                                relative_l2 < $l2_limit && max_error < $max_limit,
                                "n={n}, {mode:?}, {direction:?}, cutoff={threshold}: L2={relative_l2}, max={max_error}"
                            );
                        }
                    }
                    }
                }
            }
        };
    }

    test_large_impulse_accuracy!(
        large_impulse_accuracy_f64,
        f64,
        PlannerDit64,
        fft_f64_dit_with_planner_and_opts,
        5e-15,
        2e-14
    );
    test_large_impulse_accuracy!(
        large_impulse_accuracy_f32,
        f32,
        PlannerDit32,
        fft_f32_dit_with_planner_and_opts,
        5e-7,
        2e-6
    );

    #[cfg(feature = "complex-nums")]
    #[test]
    fn fft_interleaved_correctness() {
        let n = 10;
        let big_n = 1 << n; // 2.pow(n)
        let mut actual_signal: Vec<_> = (1..=big_n).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut expected_reals: Vec<_> = (1..=big_n).map(|i| i as f64).collect();
        let mut expected_imags = vec![0.0; big_n];

        fft_f64_dit_interleaved(&mut actual_signal, Direction::Forward);
        fft_f64_dit(&mut expected_reals, &mut expected_imags, Direction::Forward);

        actual_signal
            .iter()
            .zip(expected_reals)
            .zip(expected_imags)
            .for_each(|((z, z_re), z_im)| {
                assert_float_closeness(z.re, z_re, 1e-10);
                assert_float_closeness(z.im, z_im, 1e-10);
            });

        let n = 10;
        let big_n = 1 << n; // 2.pow(n)
        let mut actual_signal: Vec<_> = (1..=big_n).map(|i| Complex::new(i as f32, 0.0)).collect();
        let mut expected_reals: Vec<_> = (1..=big_n).map(|i| i as f32).collect();
        let mut expected_imags = vec![0.0; big_n];

        fft_f32_dit_interleaved(&mut actual_signal, Direction::Forward);
        fft_f32_dit(&mut expected_reals, &mut expected_imags, Direction::Forward);

        actual_signal
            .iter()
            .zip(expected_reals)
            .zip(expected_imags)
            .for_each(|((z, z_re), z_im)| {
                assert_float_closeness(z.re, z_re, 1e-10);
                assert_float_closeness(z.im, z_im, 1e-10);
            });
    }

    #[test]
    fn test_dit_fft_64_followed_by_ifft_correctness() {
        for n in 4..12 {
            let size = 1 << n; // 2.pow(n)
            let mut reals_original = vec![0.0f64; size];
            let mut imags_original = vec![0.0f64; size];
            let mut reals = vec![0.0f64; size];
            let mut imags = vec![0.0f64; size];

            gen_random_signal_f64(&mut reals_original, &mut imags_original);
            reals.copy_from_slice(&reals_original);
            imags.copy_from_slice(&imags_original);

            fft_f64_dit(&mut reals, &mut imags, Direction::Forward);

            fft_f64_dit(&mut reals, &mut imags, Direction::Inverse);

            for i in 0..size {
                assert_float_closeness(reals[i], reals_original[i], 1e-10);
                assert_float_closeness(imags[i], imags_original[i], 1e-10);
            }
        }
    }

    #[test]
    fn test_dit_fft_32_followed_by_ifft_correctness() {
        // Include both exponent parities beyond the cache-sized leaves.
        for n in 4..15 {
            let size = 1 << n; // 2.pow(n)
            let mut reals_original = vec![0.0f32; size];
            let mut imags_original = vec![0.0f32; size];
            let mut reals = vec![0.0f32; size];
            let mut imags = vec![0.0f32; size];

            gen_random_signal_f32(&mut reals_original, &mut imags_original);
            reals.copy_from_slice(&reals_original);
            imags.copy_from_slice(&imags_original);

            fft_f32_dit(&mut reals, &mut imags, Direction::Forward);
            fft_f32_dit(&mut reals, &mut imags, Direction::Inverse);

            for i in 0..size {
                assert_float_closeness(reals[i], reals_original[i], 1e-7);
                assert_float_closeness(imags[i], imags_original[i], 1e-7);
            }
        }
    }

    #[test]
    fn public_types_impl_standard_traits() {
        assert_eq!(Direction::Forward, Direction::Forward);
        assert_ne!(Direction::Forward, Direction::Inverse);
        assert_eq!(format!("{:?}", Direction::Forward), "Forward");

        let a = Options::guess_options(1 << 10);
        assert_eq!(a, a.clone());

        let planner = PlannerDit64::new(1 << 10);
        let _ = format!("{planner:?}"); // terse Debug must not panic
        let _ = planner.clone();
    }
}
