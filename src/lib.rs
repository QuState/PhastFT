#![doc = include_str!("../README.md")]
#![warn(clippy::complexity)]
#![warn(missing_docs)]
#![warn(clippy::style)]
#![warn(clippy::correctness)]
#![warn(clippy::suspicious)]
#![warn(clippy::perf)]
#![forbid(unsafe_code)]
#![feature(portable_simd, avx512_target_feature)]

use std::f64::consts::PI;

use crate::cobra::cobra_apply;
use crate::kernels::{
    fft_32_chunk_n_simd, fft_64_chunk_n_simd, fft_chunk_2, fft_chunk_4, fft_chunk_n,
};
use crate::options::Options;
use crate::planner::{Direction, Planner32, Planner64};
use crate::twiddles::filter_twiddles;

pub mod cobra;
pub mod fft;
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

            let mut planner = <$planner>::new(reals.len(), Direction::Forward);
            assert!(
                planner.num_twiddles().is_power_of_two()
                    && planner.num_twiddles() == reals.len() / 2
            );

            let opts = Options::guess_options(reals.len());

            match direction {
                Direction::Reverse => {
                    for z_im in imags.iter_mut() {
                        *z_im = -*z_im;
                    }
                }
                _ => (),
            }

            $opts_and_plan(reals, imags, &opts, &mut planner);

            match direction {
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

impl_fft_for!(fft_64, f64, Planner64, fft_64_with_opts_and_plan);
impl_fft_for!(fft_32, f32, Planner32, fft_32_with_opts_and_plan);

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

fn compute_twiddle_factors(big_n: usize) -> (Vec<f64>, Vec<f64>) {
    let half_n = big_n / 2;
    let mut real_parts = Vec::with_capacity(half_n);
    let mut imag_parts = Vec::with_capacity(half_n);

    for k in 0..half_n {
        let angle = -2.0 * PI * (k as f64) / (big_n as f64);
        real_parts.push(angle.cos());
        imag_parts.push(angle.sin());
    }

    (real_parts, imag_parts)
}

// TODO: make this generic over f64/f32 using macro
/// Real-to-Complex FFT `f64`. Note the input is a real-valued signal.  
pub fn fft_64_r2c(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let big_n = signal.len();

    // z[n] = x_{e}[n] + j * x_{o}[n]
    let (mut reals, mut imags): (Vec<f64>, Vec<f64>) =
        signal.chunks_exact(2).map(|c| (c[0], c[1])).unzip();

    // Z[k] = DFT{z}
    fft_64(&mut reals, &mut imags, Direction::Forward);

    let mut x_evens_re = vec![0.0; big_n / 2];
    let mut x_evens_im = vec![0.0; big_n / 2];

    for k in 0..big_n / 2 {
        let re = 0.5 * (reals[k] + reals[big_n / 2 - 1 - k]);
        let im = 0.5 * (imags[k] - imags[big_n / 2 - 1 - k]);
        x_evens_re[k] = re;
        x_evens_im[k] = im;
    }

    let mut x_odds_re = vec![0.0; big_n / 2];
    let mut x_odds_im = vec![0.0; big_n / 2];

    // -i * ((a + ib) - (c - id)) = -(b + d) - i(a - c)
    for k in 0..big_n / 2 {
        let re = 0.5 * (-imags[k] - imags[big_n / 2 - 1 - k]);
        let im = 0.5 * (-reals[k] + reals[k]);
        x_odds_re[k] = re;
        x_odds_im[k] = im;
    }

    // 7. X[k] = X_{e}[k] + X_{o}[e] * e^{-2*j*pi*(k/N)}, for k \in {0, ..., N/2 - 1}
    let (twiddles_re, twiddles_im) = compute_twiddle_factors(big_n);
    for k in 0..big_n / 2 {
        let a = x_evens_re[k];
        let b = x_evens_im[k];
        let c = x_odds_re[k];
        let d = x_odds_im[k];
        let g = twiddles_re[k];
        let h = twiddles_im[k];

        // (a + ib) + (c + id) * (g + ih) = (a + cg - dh) + i(b + ch + dg)
        reals[k] = a + c * g - d * h;
        imags[k] = b + c * h + d * g;
    }

    // 8. X[k] = X_e[k] - X_{o}[k], for k = N/2
    let k = big_n / 2 - 1;
    reals[k] = x_evens_re[k] - x_odds_re[k];
    imags[k] = x_evens_im[k] - x_odds_im[k];

    // 9. X[k] = X*[N - k], for k \in {N/2 + 1, ..., N - 1}
    for k in (big_n / 2 + 1)..big_n {
        eprintln!("k: {k} and {}", big_n - k - 1);
        reals[k] = reals[big_n - k - 1];
        imags[k] = -imags[big_n - k - 1];
    }

    (reals, imags)
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use utilities::{assert_float_closeness, gen_random_signal};
    use utilities::rustfft::FftPlanner;
    use utilities::rustfft::num_complex::Complex;

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
    fn fft_r2c_vs_c2c() {
        let n = 4;
        let big_n = 1 << n;
        let mut reals: Vec<f64> = (1..=big_n).map(|i| i as f64).collect();

        let (signal_re, signal_im) = fft_64_r2c(&mut reals);
        println!("{:?}", signal_re);
        println!("{:?}\n", signal_im);

        let mut reals: Vec<f64> = (1..=big_n).map(|i| i as f64).collect();
        let mut imags = vec![0.0; big_n];
        fft_64(&mut reals, &mut imags, Direction::Forward);

        println!("{:?}", reals);
        println!("{:?}\n", imags);
    }
}
