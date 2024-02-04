#![feature(portable_simd)]

use crate::cobra::cobra_apply;
use crate::kernels::{fft_chunk_2, fft_chunk_4, fft_chunk_n, fft_chunk_n_simd, Float};
use crate::options::Options;
use crate::planner::Planner;
use crate::twiddles::filter_twiddles;

mod cobra;
mod kernels;
pub mod options;
pub mod planner;
mod twiddles;

/// FFT -- Decimation in Frequency
///
/// This is just the decimation-in-time algorithm, reversed.
/// The inputs are in normal order, and the outputs are then bit reversed.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`
///
/// [1] https://inst.eecs.berkeley.edu/~ee123/sp15/Notes/Lecture08_FFT_and_SpectAnalysis.key.pdf
pub fn fft(reals: &mut [Float], imags: &mut [Float], planner: &mut Planner) {
    let opts = Options::guess_options(reals.len());
    fft_with_opts(reals, imags, &opts, planner);
}

/// Same as [fft], but also accepts [`Options`] that control optimization strategies.
///
/// `fft_dif` automatically guesses the best strategy for a given input,
/// so you only need to call this if you are tuning performance for a specific hardware platform.
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`
pub fn fft_with_opts(
    reals: &mut [Float],
    imags: &mut [Float],
    opts: &Options,
    planner: &mut Planner,
) {
    assert_eq!(reals.len(), imags.len());
    let n: usize = reals.len().ilog2() as usize;

    let twiddles_re = &mut planner.twiddles_re;
    let twiddles_im = &mut planner.twiddles_im;

    for t in (0..n).rev() {
        let dist = 1 << t;
        let chunk_size = dist << 1;

        if chunk_size > 4 {
            if t < n - 1 {
                filter_twiddles(twiddles_re, twiddles_im);
            }
            if chunk_size >= 16 {
                fft_chunk_n_simd(reals, imags, twiddles_re, twiddles_im, dist);
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
        rustfft::{num_complex::Complex64, FftPlanner},
    };

    use super::*;

    #[test]
    fn fft_correctness() {
        let range = Range { start: 4, end: 17 };

        for k in range {
            let n: usize = 1 << k;

            let mut reals: Vec<Float> = (1..=n).map(|i| i as f64).collect();
            let mut imags: Vec<Float> = (1..=n).map(|i| i as f64).collect();
            let mut planner = Planner::new(n);
            fft(&mut reals, &mut imags, &mut planner);

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
