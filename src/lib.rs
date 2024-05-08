#![doc = include_str!("../README.md")]
#![warn(clippy::complexity)]
#![warn(missing_docs)]
#![warn(clippy::style)]
#![warn(clippy::correctness)]
#![warn(clippy::suspicious)]
#![warn(clippy::perf)]
#![forbid(unsafe_code)]
#![feature(portable_simd, avx512_target_feature)]

#[cfg(feature = "complex-nums")]
use num_complex::Complex;
#[cfg(feature = "complex-nums")]
use num_traits::Float;

use crate::cobra::cobra_apply;
use crate::kernels::{
    fft_32_chunk_n_simd, fft_64_chunk_n_simd, fft_chunk_2, fft_chunk_4, fft_chunk_n,
};
use crate::options::Options;
use crate::planner::{Direction, Planner32, Planner64};
use crate::twiddles::filter_twiddles;

pub mod cobra;
mod kernels;
pub mod options;
pub mod planner;
mod twiddles;

macro_rules! impl_fft_for {
    ($func_name:ident, $precision:ty, $planner:ty, $opts_and_plan:ident) => {
        /// FFT -- Decimation in Frequency. This is just the decimation-in-time algorithm, reversed.
        /// This call to FFT is run, in-place.
        /// The input should be provided in normal order, and then the modified input is bit-reversed.
        ///
        /// # Panics
        ///
        /// Panics if `reals.len() != imags.len()`, or if the input length is _not_ a power of 2.
        ///
        /// ## References
        /// <https://inst.eecs.berkeley.edu/~ee123/sp15/Notes/Lecture08_FFT_and_SpectAnalysis.key.pdf>
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

            let mut planner = <$planner>::new(reals.len(), direction);
            assert!(
                planner.num_twiddles().is_power_of_two()
                    && planner.num_twiddles() == reals.len() / 2
            );

            let opts = Options::guess_options(reals.len());
            $opts_and_plan(reals, imags, &opts, &mut planner);
        }
    };
}

impl_fft_for!(fft_64, f64, Planner64, fft_64_with_opts_and_plan);
impl_fft_for!(fft_32, f32, Planner32, fft_32_with_opts_and_plan);

#[cfg(feature = "complex-nums")]
macro_rules! impl_fft_interleaved_for {
    ($func_name:ident, $precision:ty, $fft_func:ident) => {
        /// FFT Interleaved -- this is an alternative to [`fft_64`]/[`fft_32`] in the case where
        /// the input data is a array of [`num_complex::Complex`].
        ///
        /// The input should be provided in normal order, and then the modified input is
        /// bit-reversed.
        ///
        /// ## References
        /// <https://inst.eecs.berkeley.edu/~ee123/sp15/Notes/Lecture08_FFT_and_SpectAnalysis.key.pdf>
        pub fn $func_name(signal: &mut [Complex<$precision>], direction: Direction) {
            let (mut reals, mut imags) = separate_re_im(signal);
            $fft_func(&mut reals, &mut imags, direction);
            signal.copy_from_slice(&combine_re_im(&reals, &imags))
        }
    };
}

#[cfg(feature = "complex-nums")]
impl_fft_interleaved_for!(fft_32_interleaved, f32, fft_32);
#[cfg(feature = "complex-nums")]
impl_fft_interleaved_for!(fft_64_interleaved, f64, fft_64);

macro_rules! impl_fft_with_opts_and_plan_for {
    ($func_name:ident, $precision:ty, $planner:ty, $simd_butterfly_kernel:ident, $lanes:literal) => {
        /// Same as [fft], but also accepts [`Options`] that control optimization strategies, as well as
        /// a [`Planner`] in the case that this FFT will need to be run multiple times.
        ///
        /// `fft` automatically guesses the best strategy for a given input,
        /// so you only need to call this if you are tuning performance for a specific hardware platform.
        ///
        /// In addition, `fft` automatically creates a planner to be used. In the case that you plan
        /// on running an FFT many times on inputs of the same size, use this function with the pre-built
        /// [`Planner`].
        ///
        /// # Panics
        ///
        /// Panics if `reals.len() != imags.len()`, or if the input length is _not_ a power of 2.
        #[multiversion::multiversion(
                                    targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl", // x86_64-v4
                                            "x86_64+avx2+fma", // x86_64-v3
                                            "x86_64+sse4.2", // x86_64-v2
                                            "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
                                            "x86+avx2+fma",
                                            "x86+sse4.2",
                                            "x86+sse2",
        ))]
        pub fn $func_name(
            reals: &mut [$precision],
            imags: &mut [$precision],
            opts: &Options,
            planner: &mut $planner,
        ) {
            assert!(reals.len() == imags.len() && reals.len().is_power_of_two());
            let n: usize = reals.len().ilog2() as usize;

            let twiddles_re = &mut planner.twiddles_re;
            let twiddles_im = &mut planner.twiddles_im;

            // We shouldn't be able to execute FFT if the # of twiddles isn't equal to the distance
            // between pairs
            assert!(twiddles_re.len() == reals.len() / 2 && twiddles_im.len() == imags.len() / 2);

            for t in (0..n).rev() {
                let dist = 1 << t;
                let chunk_size = dist << 1;

                if chunk_size > 4 {
                    if t < n - 1 {
                        filter_twiddles(twiddles_re, twiddles_im);
                    }
                    if chunk_size >= $lanes * 2 {
                        $simd_butterfly_kernel(reals, imags, twiddles_re, twiddles_im, dist);
                    } else {
                        fft_chunk_n(reals, imags, twiddles_re, twiddles_im, dist);
                    }
                } else if chunk_size == 2 {
                    fft_chunk_2(reals, imags);
                } else if chunk_size == 4 {
                    fft_chunk_4(reals, imags);
                }
            }

            if opts.multithreaded_bit_reversal {
                std::thread::scope(|s| {
                    s.spawn(|| cobra_apply(reals, n));
                    s.spawn(|| cobra_apply(imags, n));
                });
            } else {
                cobra_apply(reals, n);
                cobra_apply(imags, n);
            }
        }
    };
}

impl_fft_with_opts_and_plan_for!(
    fft_64_with_opts_and_plan,
    f64,
    Planner64,
    fft_64_chunk_n_simd,
    8
);

impl_fft_with_opts_and_plan_for!(
    fft_32_with_opts_and_plan,
    f32,
    Planner32,
    fft_32_chunk_n_simd,
    16
);

/// Utility function to separate interleaved format signals (i.e., Vector of Complex Number Structs)
/// into separate vectors for the corresponding real and imaginary components.
#[cfg(feature = "complex-nums")]
pub fn separate_re_im<T: Float>(signal: &[Complex<T>]) -> (Vec<T>, Vec<T>) {
    signal.iter().map(|z| (z.re, z.im)).unzip()
}

/// Utility function to combine separate vectors of real and imaginary components
/// into a single vector of Complex Number Structs.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`.
#[cfg(feature = "complex-nums")]
pub fn combine_re_im<T: Float>(reals: &[T], imags: &[T]) -> Vec<Complex<T>> {
    assert_eq!(reals.len(), imags.len());

    reals
        .iter()
        .zip(imags.iter())
        .map(|(z_re, z_im)| Complex::new(*z_re, *z_im))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use utilities::assert_float_closeness;
    use utilities::rustfft::num_complex::Complex;
    use utilities::rustfft::FftPlanner;

    use super::*;

    #[cfg(feature = "complex-nums")]
    #[test]
    fn test_separate_and_combine_re_im() {
        let complex_vec = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];

        let (reals, imags) = separate_re_im(&complex_vec);

        let recombined_vec = combine_re_im(&reals, &imags);

        assert_eq!(complex_vec, recombined_vec);
    }

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
        let n = 4;
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

        let n = 4;
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
}
