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

use crate::options::Options;
use crate::planner::{Direction, Planner32, Planner64, PlannerDit32, PlannerDit64};
#[cfg(feature = "complex-nums")]
use crate::utils::{combine_re_im, deinterleave_complex32, deinterleave_complex64};

mod algorithms;
mod kernels;
pub mod options;
pub mod planner;
mod twiddles;
mod utils;

pub use algorithms::dif::{fft_32_with_opts_and_plan, fft_64_with_opts_and_plan};
pub use algorithms::dit::{fft_32_dit_with_planner_and_opts, fft_64_dit_with_planner_and_opts};

macro_rules! impl_fft_for {
    ($func_name:ident, $precision:ty, $planner:ty, $opts_and_plan:ident) => {
        /// FFT using Decimation-in-Frequency (DIF) algorithm.
        ///
        /// This call to FFT is run in-place. Input should be provided in normal order.
        /// By default, output is bit-reversed (standard FFT output).
        ///
        /// To control bit reversal behavior, use [`fft_64_with_opts_and_plan`] or
        /// [`fft_32_with_opts_and_plan`] with `Options::dif_perform_bit_reversal`.
        ///
        /// # Panics
        ///
        /// Panics if `reals.len() != imags.len()`, or if the input length is _not_ a power of 2.
        ///
        /// # Bit Reversal
        ///
        /// - Input: Normal order
        /// - Output: Bit-reversed order
        ///
        /// ## References
        /// <https://www.cmlab.csie.ntu.edu.tw/cml/dsp/training/coding/transform/fft.html>
        ///
        pub fn $func_name(
            reals: &mut [$precision],
            imags: &mut [$precision],
            direction: Direction,
        ) {
            assert_eq!(
                reals.len(),
                imags.len(),
                "real and imaginary inputs must be of equal size, but got: {} {}",
                reals.len(),
                imags.len()
            );

            let planner = <$planner>::new(reals.len(), direction);
            assert!(
                planner.num_twiddles().is_power_of_two()
                    && planner.num_twiddles() == reals.len() / 2
            );

            let opts = Options::guess_options(reals.len());
            algorithms::dif::$opts_and_plan(reals, imags, &opts, &planner);
        }
    };
}

impl_fft_for!(fft_64, f64, Planner64, fft_64_with_opts_and_plan);
impl_fft_for!(fft_32, f32, Planner32, fft_32_with_opts_and_plan);

#[cfg(feature = "complex-nums")]
macro_rules! impl_fft_interleaved_for {
    ($func_name:ident, $precision:ty, $fft_func:ident, $deinterleaving_func: ident) => {
        /// FFT Interleaved -- this is an alternative to [`fft_64`]/[`fft_32`] in the case where
        /// the input data is a array of [`Complex`].
        ///
        /// The input should be provided in normal order, and then the modified input is
        /// bit-reversed.
        ///
        /// **Note**: This function currently allocates temporary buffers for deinterleaving.
        /// For maximum performance with minimal allocations, use the separate real/imaginary APIs.
        ///
        /// ## References
        /// <https://inst.eecs.berkeley.edu/~ee123/sp15/Notes/Lecture08_FFT_and_SpectAnalysis.key.pdf>
        pub fn $func_name(signal: &mut [Complex<$precision>], direction: Direction) {
            let (mut reals, mut imags) = $deinterleaving_func(signal);
            $fft_func(&mut reals, &mut imags, direction);
            signal.copy_from_slice(&combine_re_im(&reals, &imags))
        }
    };
}

#[cfg(feature = "complex-nums")]
impl_fft_interleaved_for!(fft_32_interleaved, f32, fft_32, deinterleave_complex32);
#[cfg(feature = "complex-nums")]
impl_fft_interleaved_for!(fft_64_interleaved, f64, fft_64, deinterleave_complex64);

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
/// # Algorithm Overview
///
/// The DIT algorithm:
/// 1. Applies bit-reversal to the input (reordering)
/// 2. Processes from small butterflies (size 2) to large
/// 3. Output is in natural order
///
/// This is the dual of the DIF algorithm, with opposite data flow.
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
/// # Algorithm Overview
///
/// The DIT algorithm:
/// 1. Applies bit-reversal to the input (reordering)
/// 2. Processes from small butterflies (size 2) to large
/// 3. Output is in natural order
///
/// This is the dual of the DIF algorithm, with opposite data flow.
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
    use utilities::{assert_float_closeness, gen_random_signal, gen_random_signal_f32};

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

    non_power_of_2_planner!(non_power_of_2_planner_32, Planner32);
    non_power_of_2_planner!(non_power_of_2_planner_64, Planner64);

    macro_rules! wrong_num_points_in_planner {
        ($test_name:ident, $planner:ty, $fft_with_opts_and_plan:ident) => {
            // A regression test to make sure the `Planner` is compatible with fft execution.
            #[should_panic]
            #[test]
            fn $test_name() {
                let n = 16;
                let num_points = 1 << n;

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
                $fft_with_opts_and_plan(&mut reals, &mut imags, &opts, &mut planner);
            }
        };
    }

    wrong_num_points_in_planner!(
        wrong_num_points_in_planner_32,
        Planner32,
        fft_32_with_opts_and_plan
    );
    wrong_num_points_in_planner!(
        wrong_num_points_in_planner_64,
        Planner64,
        fft_64_with_opts_and_plan
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
                    let n: usize = 1 << k;

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

    test_fft_correctness!(fft_correctness_32, f32, fft_32, 4, 9);
    test_fft_correctness!(fft_correctness_64, f64, fft_64, 4, 17);

    #[cfg(feature = "complex-nums")]
    #[test]
    fn fft_interleaved_correctness() {
        let n = 10;
        let big_n = 1 << n;
        let mut actual_signal: Vec<_> = (1..=big_n).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut expected_reals: Vec<_> = (1..=big_n).map(|i| i as f64).collect();
        let mut expected_imags = vec![0.0; big_n];

        fft_64_interleaved(&mut actual_signal, Direction::Forward);
        fft_64(&mut expected_reals, &mut expected_imags, Direction::Forward);

        actual_signal
            .iter()
            .zip(expected_reals)
            .zip(expected_imags)
            .for_each(|((z, z_re), z_im)| {
                assert_float_closeness(z.re, z_re, 1e-10);
                assert_float_closeness(z.im, z_im, 1e-10);
            });

        let n = 10;
        let big_n = 1 << n;
        let mut actual_signal: Vec<_> = (1..=big_n).map(|i| Complex::new(i as f32, 0.0)).collect();
        let mut expected_reals: Vec<_> = (1..=big_n).map(|i| i as f32).collect();
        let mut expected_imags = vec![0.0; big_n];

        fft_32_interleaved(&mut actual_signal, Direction::Forward);
        fft_32(&mut expected_reals, &mut expected_imags, Direction::Forward);

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
    fn fft_round_trip() {
        for i in 4..23 {
            let big_n = 1 << i;
            let mut reals = vec![0.0; big_n];
            let mut imags = vec![0.0; big_n];

            gen_random_signal(&mut reals, &mut imags);

            let original_reals = reals.clone();
            let original_imags = imags.clone();

            // Forward FFT
            fft_64(&mut reals, &mut imags, Direction::Forward);

            // Inverse FFT
            fft_64(&mut reals, &mut imags, Direction::Reverse);

            // Ensure we get back the original signal within some tolerance
            for ((orig_re, orig_im), (res_re, res_im)) in original_reals
                .into_iter()
                .zip(original_imags.into_iter())
                .zip(reals.into_iter().zip(imags.into_iter()))
            {
                assert_float_closeness(res_re, orig_re, 1e-6);
                assert_float_closeness(res_im, orig_im, 1e-6);
            }
        }
    }

    #[test]
    fn fft_64_with_opts_and_plan_vs_fft_64() {
        let num_points = 4096;

        let mut reals = vec![0.0; num_points];
        let mut imags = vec![0.0; num_points];
        gen_random_signal(&mut reals, &mut imags);

        let mut re = reals.clone();
        let mut im = imags.clone();

        let planner = Planner64::new(num_points, Direction::Forward);
        let opts = Options::guess_options(reals.len());
        fft_64_with_opts_and_plan(&mut reals, &mut imags, &opts, &planner);

        fft_64(&mut re, &mut im, Direction::Forward);

        reals
            .iter()
            .zip(imags.iter())
            .zip(re.iter())
            .zip(im.iter())
            .for_each(|(((r, i), z_re), z_im)| {
                assert_float_closeness(*r, *z_re, 1e-6);
                assert_float_closeness(*i, *z_im, 1e-6);
            });
    }

    #[test]
    fn test_dit_fft_correctness() {
        // Test DIT FFT against DIF FFT
        for n in 4..12 {
            let size = 1 << n;
            let mut reals_dit = vec![0.0f64; size];
            let mut imags_dit = vec![0.0f64; size];
            let mut reals_dif = vec![0.0f64; size];
            let mut imags_dif = vec![0.0f64; size];

            // Generate same random signal for both
            gen_random_signal(&mut reals_dit, &mut imags_dit);
            reals_dif.copy_from_slice(&reals_dit);
            imags_dif.copy_from_slice(&imags_dit);

            // Run DIT FFT
            fft_64_dit(&mut reals_dit, &mut imags_dit, Direction::Forward);

            // Run DIF FFT
            fft_64(&mut reals_dif, &mut imags_dif, Direction::Forward);

            // Compare results
            for i in 0..size {
                assert_float_closeness(reals_dit[i], reals_dif[i], 1e-10);
                assert_float_closeness(imags_dit[i], imags_dif[i], 1e-10);
            }
        }

        // Test f32 version
        for n in 4..10 {
            let size = 1 << n;
            let mut reals_dit = vec![0.0f32; size];
            let mut imags_dit = vec![0.0f32; size];
            let mut reals_dif = vec![0.0f32; size];
            let mut imags_dif = vec![0.0f32; size];

            // Generate same random signal for both
            gen_random_signal_f32(&mut reals_dit, &mut imags_dit);
            reals_dif.copy_from_slice(&reals_dit);
            imags_dif.copy_from_slice(&imags_dit);

            // Run DIT FFT
            fft_32_dit(&mut reals_dit, &mut imags_dit, Direction::Forward);

            // Run DIF FFT
            fft_32(&mut reals_dif, &mut imags_dif, Direction::Forward);

            for i in 0..size {
                assert_float_closeness(reals_dit[i], reals_dif[i], 1e-4);
                assert_float_closeness(imags_dit[i], imags_dif[i], 1e-4);
            }
        }
    }

    #[test]
    fn test_dit_fft_64_followed_by_ifft_correctness() {
        for n in 4..12 {
            let size = 1 << n;
            let mut reals_original = vec![0.0f64; size];
            let mut imags_original = vec![0.0f64; size];
            let mut reals = vec![0.0f64; size];
            let mut imags = vec![0.0f64; size];

            gen_random_signal(&mut reals_original, &mut imags_original);
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
            let size = 1 << n;
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

    #[test]
    fn fft_32_with_opts_and_plan_vs_fft_32() {
        let dirs = [Direction::Forward, Direction::Reverse];

        for direction in dirs {
            for n in 4..14 {
                let num_points = 1 << n;
                let mut reals = vec![0.0; num_points];
                let mut imags = vec![0.0; num_points];
                gen_random_signal_f32(&mut reals, &mut imags);

                let mut re = reals.clone();
                let mut im = imags.clone();

                let planner = Planner32::new(num_points, direction);
                let opts = Options::guess_options(reals.len());
                fft_32_with_opts_and_plan(&mut reals, &mut imags, &opts, &planner);

                fft_32(&mut re, &mut im, direction);

                reals
                    .iter()
                    .zip(imags.iter())
                    .zip(re.iter())
                    .zip(im.iter())
                    .for_each(|(((r, i), z_re), z_im)| {
                        assert_float_closeness(*r, *z_re, 1e-6);
                        assert_float_closeness(*i, *z_im, 1e-6);
                    });
            }
        }
    }
}
