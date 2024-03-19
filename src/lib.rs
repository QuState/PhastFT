#![doc = include_str!("../README.md")]
#![warn(clippy::complexity)]
#![warn(missing_docs)]
#![warn(clippy::style)]
#![warn(clippy::correctness)]
#![warn(clippy::suspicious)]
#![warn(clippy::perf)]
#![forbid(unsafe_code)]
#![feature(portable_simd)]

use crate::cobra::cobra_apply;
use crate::kernels::{
    fft_32_chunk_n_simd, fft_64_chunk_n_simd, fft_chunk_2, fft_chunk_4, fft_chunk_n,
};
use crate::options::Options;
use crate::planner::{Direction, Planner};
use crate::twiddles::filter_twiddles;

pub mod cobra;
mod kernels;
pub mod options;
pub mod planner;
mod twiddles;

/// FFT -- Decimation in Frequency. This is just the decimation-in-time algorithm, reversed.
/// This call to FFT is run, in-place.
/// The input should be provided in normal order, and then the modified input is bit-reversed.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`
///
/// ## References
/// <https://inst.eecs.berkeley.edu/~ee123/sp15/Notes/Lecture08_FFT_and_SpectAnalysis.key.pdf>
pub fn fft_64(reals: &mut [f64], imags: &mut [f64], direction: Direction) {
    assert_eq!(
        reals.len(),
        imags.len(),
        "real and imaginary inputs must be of equal size, but got: {} {}",
        reals.len(),
        imags.len()
    );

    let mut planner = Planner::new(reals.len(), direction);
    assert!(planner.num_twiddles().is_power_of_two() && planner.num_twiddles() == reals.len() / 2);

    let opts = Options::guess_options(reals.len());
    fft_64_with_opts_and_plan(reals, imags, &opts, &mut planner);
}

/// FFT -- Decimation in Frequency. This is just the decimation-in-time algorithm, reversed.
/// This call to FFT is run, in-place.
/// The input should be provided in normal order, and then the modified input is bit-reversed.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`
///
/// ## References
/// <https://inst.eecs.berkeley.edu/~ee123/sp15/Notes/Lecture08_FFT_and_SpectAnalysis.key.pdf>
pub fn fft_32(reals: &mut [f32], imags: &mut [f32], direction: Direction) {
    assert_eq!(
        reals.len(),
        imags.len(),
        "real and imaginary inputs must be of equal size, but got: {} {}",
        reals.len(),
        imags.len()
    );

    let mut planner = Planner::new(reals.len(), direction);
    assert!(planner.num_twiddles().is_power_of_two() && planner.num_twiddles() == reals.len() / 2);

    let opts = Options::guess_options(reals.len());
    fft_32_with_opts_and_plan(reals, imags, &opts, &mut planner);
}

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
/// Panics if `reals.len() != imags.len()`, or if the input length is *not* a power of two.
pub fn fft_32_with_opts_and_plan(
    reals: &mut [f32],
    imags: &mut [f32],
    opts: &Options,
    planner: &mut Planner<f32>,
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
            if chunk_size >= 16 {
                fft_32_chunk_n_simd(reals, imags, twiddles_re, twiddles_im, dist);
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
/// Panics if `reals.len() != imags.len()`, or if the input length is *not* a power of two.
pub fn fft_64_with_opts_and_plan(
    reals: &mut [f64],
    imags: &mut [f64],
    opts: &Options,
    planner: &mut Planner<f64>,
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
            if chunk_size >= 16 {
                fft_64_chunk_n_simd(reals, imags, twiddles_re, twiddles_im, dist);
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

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use utilities::{
        assert_f64_closeness,
        rustfft::{FftPlanner, num_complex::Complex64},
    };

    use super::*;

    #[should_panic]
    #[test]
    fn non_power_of_two_fft() {
        let num_points = 5;

        // this test will actually always fail at this stage
        let mut planner = Planner::new(num_points, Direction::Forward);

        let mut reals = vec![0.0; num_points];
        let mut imags = vec![0.0; num_points];
        let opts = Options::guess_options(reals.len());

        // but this call should, in principle, panic as well
        fft_64_with_opts_and_plan(&mut reals, &mut imags, &opts, &mut planner);
    }

    // A regression test to make sure the `Planner` is compatible with fft execution.
    #[should_panic]
    #[test]
    fn wrong_num_points_in_planner() {
        let n = 16;
        let num_points = 1 << n;

        // We purposely set n = 16 and pass it to the planner.
        // n = 16 == 2^{4} is clearly a power of two, so the planner won't throw it out.
        // However, the call to `fft_with_opts_and_plan` should panic since it tests that the
        // size of the generated twiddle factors is half the size of the input.
        // In this case, we have an input of size 1024 (used for mp3), but we tell the planner the
        // input size is 16.
        let mut planner = Planner::new(n, Direction::Forward);

        let mut reals = vec![0.0; num_points];
        let mut imags = vec![0.0; num_points];
        let opts = Options::guess_options(reals.len());

        // but this call should panic as well
        fft_64_with_opts_and_plan(&mut reals, &mut imags, &opts, &mut planner);
    }

    #[test]
    fn fft_correctness() {
        let range = Range { start: 4, end: 17 };

        for k in range {
            let n: usize = 1 << k;

            let mut reals: Vec<_> = (1..=n).map(|i| i as f64).collect();
            let mut imags: Vec<_> = (1..=n).map(|i| i as f64).collect();
            fft_64(&mut reals, &mut imags, Direction::Forward);

            let mut buffer: Vec<Complex64> = (1..=n)
                .map(|i| Complex64::new(i as f64, i as f64))
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
                    assert_f64_closeness(*z_re, expect_re, 0.01);
                    assert_f64_closeness(*z_im, expect_im, 0.01);
                });
        }
    }
}
