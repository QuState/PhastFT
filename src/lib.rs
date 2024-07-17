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
#![feature(portable_simd, avx512_target_feature)]
#![feature(doc_cfg)]

#[cfg(feature = "complex-nums")]
use crate::utils::{combine_re_im, deinterleave_complex32, deinterleave_complex64};
#[cfg(feature = "complex-nums")]
use num_complex::Complex;

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
mod utils;

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

            let planner = <$planner>::new(reals.len(), direction);
            assert!(
                planner.num_twiddles().is_power_of_two()
                    && planner.num_twiddles() == reals.len() / 2
            );

            let opts = Options::guess_options(reals.len());

            $opts_and_plan(reals, imags, &opts, &planner);
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
        /// ## References
        /// <https://inst.eecs.berkeley.edu/~ee123/sp15/Notes/Lecture08_FFT_and_SpectAnalysis.key.pdf>
        pub fn $func_name(signal: &mut [Complex<$precision>], direction: Direction) {
            let (mut reals, mut imags) = $deinterleaving_func(signal);
            $fft_func(&mut reals, &mut imags, direction);
            signal.copy_from_slice(&combine_re_im(&reals, &imags))
        }
    };
}

#[doc(cfg(feature = "complex-nums"))]
#[cfg(feature = "complex-nums")]
impl_fft_interleaved_for!(fft_32_interleaved, f32, fft_32, deinterleave_complex32);
#[doc(cfg(feature = "complex-nums"))]
#[cfg(feature = "complex-nums")]
impl_fft_interleaved_for!(fft_64_interleaved, f64, fft_64, deinterleave_complex64);

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
            planner: &$planner,
        ) {
            assert!(reals.len() == imags.len() && reals.len().is_power_of_two());
            let n: usize = reals.len().ilog2() as usize;

            // Use references to avoid unnecessary clones
            let twiddles_re = &planner.twiddles_re;
            let twiddles_im = &planner.twiddles_im;

            // We shouldn't be able to execute FFT if the # of twiddles isn't equal to the distance
            // between pairs
            assert!(twiddles_re.len() == reals.len() / 2 && twiddles_im.len() == imags.len() / 2);

            match planner.direction {
                Direction::Reverse => {
                    for z_im in imags.iter_mut() {
                        *z_im = -*z_im;
                    }
                }
                _ => (),
            }

            // 0th stage is special due to no need to filter twiddle factor
            let dist = 1 << (n - 1);
            let chunk_size = dist << 1;

            if chunk_size > 4 {
                if chunk_size >= $lanes * 2 {
                    $simd_butterfly_kernel(reals, imags, twiddles_re, twiddles_im, dist);
                } else {
                    fft_chunk_n(reals, imags, twiddles_re, twiddles_im, dist);
                }
            }
            else if chunk_size == 4 {
                fft_chunk_4(reals, imags);
            }
            else if chunk_size == 2 {
                fft_chunk_2(reals, imags);
            }

            let (mut filtered_twiddles_re, mut filtered_twiddles_im) = filter_twiddles(twiddles_re, twiddles_im);

            for t in (0..n - 1).rev() {
                let dist = 1 << t;
                let chunk_size = dist << 1;

                if chunk_size > 4 {
                    if chunk_size >= $lanes * 2 {
                        $simd_butterfly_kernel(reals, imags, &filtered_twiddles_re, &filtered_twiddles_im, dist);
                    } else {
                        fft_chunk_n(reals, imags, &filtered_twiddles_re, &filtered_twiddles_im, dist);
                    }
                }
                else if chunk_size == 4 {
                    fft_chunk_4(reals, imags);
                }
                else if chunk_size == 2 {
                    fft_chunk_2(reals, imags);
                }
                (filtered_twiddles_re, filtered_twiddles_im) = filter_twiddles(&filtered_twiddles_re, &filtered_twiddles_im);
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

            match planner.direction {
                Direction::Reverse => {
                    let scaling_factor = (reals.len() as $precision).recip();
                    for (z_re, z_im) in reals.iter_mut().zip(imags.iter_mut()) {
                        *z_re *= scaling_factor;
                        *z_im *= -scaling_factor;
                    }
                }
                _ => (),
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
    fn fft_32_with_opts_and_plan_vs_fft_64() {
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
