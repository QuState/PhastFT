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
use crate::complex_nums::{combine_re_im_into, deinterleave_complex32, deinterleave_complex64};
use crate::options::Options;
use crate::planner::{Direction, PlannerDit32, PlannerDit64};

mod algorithms;
#[cfg(all(feature = "complex-nums", not(phastft_bench)))]
mod complex_nums;
#[cfg(all(feature = "complex-nums", phastft_bench))]
pub mod complex_nums;
mod kernels;
pub mod options;
mod parallel;
pub mod planner;
#[cfg(test)]
mod twiddles;

pub use algorithms::dit::{fft_32_dit_with_planner_and_opts, fft_64_dit_with_planner_and_opts};

#[cfg(feature = "complex-nums")]
macro_rules! impl_fft_interleaved_for {
    ($func_name:ident, $precision:ty, $fft_func:ident, $deinterleaving_func: ident, $planner:ty) => {
        /// FFT Interleaved -- this is an alternative to [`fft_64`]/[`fft_32`] in the case where
        /// the input data is a array of [`Complex`].
        ///
        /// Analogous to [fft_64_dit_with_planner_and_opts] except for the input format.
        ///
        /// **Note**: This function has to make a deinterleaved copy of the data.
        /// For maximum performance with minimal memory usage, use [fft_64_dit_with_planner_and_opts].
        pub fn $func_name(signal: &mut [Complex<$precision>], planner: &$planner, opts: &Options) {
            let (mut reals, mut imags) = $deinterleaving_func(signal);
            $fft_func(&mut reals, &mut imags, planner, opts);
            let signal_flat: &mut [$precision] = bytemuck::cast_slice_mut(signal);
            combine_re_im_into(&reals, &imags, signal_flat)
        }
    };
}

#[cfg(feature = "complex-nums")]
impl_fft_interleaved_for!(
    fft_32_interleaved_with_planner_and_opts,
    f32,
    fft_32_dit_with_planner_and_opts,
    deinterleave_complex32,
    PlannerDit32
);
#[cfg(feature = "complex-nums")]
impl_fft_interleaved_for!(
    fft_64_interleaved_with_planner_and_opts,
    f64,
    fft_64_dit_with_planner_and_opts,
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
        pub fn $func_name(signal: &mut [Complex<$precision>], planner: &$planner) {
            let opts = Options::guess_options(signal.len());
            $fft_with_opts_func(signal, planner, &opts);
        }
    };
}

#[cfg(feature = "complex-nums")]
impl_fft_interleaved_with_planner!(
    fft_32_interleaved_with_planner,
    f32,
    fft_32_interleaved_with_planner_and_opts,
    PlannerDit32
);
#[cfg(feature = "complex-nums")]
impl_fft_interleaved_with_planner!(
    fft_64_interleaved_with_planner,
    f64,
    fft_64_interleaved_with_planner_and_opts,
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
            let planner = <$planner>::new(signal.len(), direction);
            $fft_with_planner_func(signal, &planner);
        }
    };
}

#[cfg(feature = "complex-nums")]
impl_fft_interleaved!(
    fft_32_interleaved,
    f32,
    fft_32_interleaved_with_planner,
    PlannerDit32
);
#[cfg(feature = "complex-nums")]
impl_fft_interleaved!(
    fft_64_interleaved,
    f64,
    fft_64_interleaved_with_planner,
    PlannerDit64
);

/// FFT using Decimation-In-Time (DIT) algorithm for f64 with pre-computed planner
pub fn fft_64_dit_with_planner(reals: &mut [f64], imags: &mut [f64], planner: &PlannerDit64) {
    let opts = Options::guess_options(reals.len());
    algorithms::dit::fft_64_dit_with_planner_and_opts(reals, imags, planner, &opts);
}

/// FFT using Decimation-In-Time (DIT) algorithm for f64.
///
/// This is a convenient wrapper that creates a planner automatically.
/// For better performance when running multiple FFTs of the same size,
/// consider using [`fft_64_dit_with_planner`].
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
/// use phastft::{fft_64_dit, planner::Direction};
///
/// let mut reals = vec![1.0, 0.0, 0.0, 0.0];
/// let mut imags = vec![0.0; 4];
/// fft_64_dit(&mut reals, &mut imags, Direction::Forward);
/// // Output is in normal order
/// ```
///
pub fn fft_64_dit(reals: &mut [f64], imags: &mut [f64], direction: Direction) {
    let planner = PlannerDit64::new(reals.len(), direction);
    fft_64_dit_with_planner(reals, imags, &planner);
}

/// FFT using Decimation-In-Time (DIT) algorithm for f32 with pre-computed planner
pub fn fft_32_dit_with_planner(reals: &mut [f32], imags: &mut [f32], planner: &PlannerDit32) {
    let opts = Options::guess_options(reals.len());
    fft_32_dit_with_planner_and_opts(reals, imags, planner, &opts);
}

/// FFT using Decimation-In-Time (DIT) algorithm for f32.
///
/// This is a convenient wrapper that creates a planner automatically.
/// For better performance when running multiple FFTs of the same size,
/// consider using [`fft_32_dit_with_planner`].
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
/// use phastft::{fft_32_dit, planner::Direction};
///
/// let mut reals = vec![1.0, 0.0, 0.0, 0.0];
/// let mut imags = vec![0.0; 4];
/// fft_32_dit(&mut reals, &mut imags, Direction::Forward);
/// // Output is in normal order
/// ```
///
pub fn fft_32_dit(reals: &mut [f32], imags: &mut [f32], direction: Direction) {
    let planner = PlannerDit32::new(reals.len(), direction);
    fft_32_dit_with_planner(reals, imags, &planner);
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
                let _ = <$planner>::new(num_points, Direction::Forward);
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
                // size of the generated twiddle factors is half the size of the input.
                // In this case, we have an input of size 1024 (used for mp3), but we tell the planner the
                // input size is 16.
                let mut planner = <$planner>::new(n, Direction::Forward);

                let mut reals = vec![0.0; num_points];
                let mut imags = vec![0.0; num_points];
                let opts = Options::guess_options(reals.len());

                // this call should panic
                $fft_with_opts_and_plan(&mut reals, &mut imags, &mut planner, &opts);
            }
        };
    }

    wrong_num_points_in_planner!(
        wrong_num_points_in_planner_32,
        PlannerDit32,
        fft_32_dit_with_planner_and_opts
    );
    wrong_num_points_in_planner!(
        wrong_num_points_in_planner_64,
        PlannerDit64,
        fft_64_dit_with_planner_and_opts
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

    test_fft_correctness!(fft_correctness_32, f32, fft_32_dit, 4, 9);
    test_fft_correctness!(fft_correctness_64, f64, fft_64_dit, 4, 17);

    #[cfg(feature = "complex-nums")]
    #[test]
    fn fft_interleaved_correctness() {
        let n = 10;
        let big_n = 1 << n; // 2.pow(n)
        let mut actual_signal: Vec<_> = (1..=big_n).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut expected_reals: Vec<_> = (1..=big_n).map(|i| i as f64).collect();
        let mut expected_imags = vec![0.0; big_n];

        fft_64_interleaved(&mut actual_signal, Direction::Forward);
        fft_64_dit(&mut expected_reals, &mut expected_imags, Direction::Forward);

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

        fft_32_interleaved(&mut actual_signal, Direction::Forward);
        fft_32_dit(&mut expected_reals, &mut expected_imags, Direction::Forward);

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

            fft_64_dit(&mut reals, &mut imags, Direction::Forward);

            fft_64_dit(&mut reals, &mut imags, Direction::Reverse);

            for i in 0..size {
                assert_float_closeness(reals[i], reals_original[i], 1e-10);
                assert_float_closeness(imags[i], imags_original[i], 1e-10);
            }
        }
    }

    #[test]
    fn test_dit_fft_32_followed_by_ifft_correctness() {
        for n in 4..12 {
            let size = 1 << n; // 2.pow(n)
            let mut reals_original = vec![0.0f32; size];
            let mut imags_original = vec![0.0f32; size];
            let mut reals = vec![0.0f32; size];
            let mut imags = vec![0.0f32; size];

            gen_random_signal_f32(&mut reals_original, &mut imags_original);
            reals.copy_from_slice(&reals_original);
            imags.copy_from_slice(&imags_original);

            fft_32_dit(&mut reals, &mut imags, Direction::Forward);
            fft_32_dit(&mut reals, &mut imags, Direction::Reverse);

            for i in 0..size {
                assert_float_closeness(reals[i], reals_original[i], 1e-7);
                assert_float_closeness(imags[i], imags_original[i], 1e-7);
            }
        }
    }
}
