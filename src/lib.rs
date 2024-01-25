#![feature(portable_simd)]

use std::simd::prelude::*;

use spinoza::{core::State, math::Float};

use crate::kernels::{fft_chunk_2, fft_chunk_4, fft_chunk_n};
use crate::{cobra::cobra_apply, twiddles::generate_twiddles};

mod bravo;
mod cobra;
mod kernels;
mod twiddles;

/// FFT -- Decimation in Frequency
///
/// This is just the decimation-in-time algorithm, reversed.
/// The inputs are in normal order, and the outputs are bit reversed.
///
/// [1] https://inst.eecs.berkeley.edu/~ee123/sp15/Notes/Lecture08_FFT_and_SpectAnalysis.key.pdf
pub fn fft_dif(state: &mut State) {
    let n: usize = state.n.into();
    // eprintln!("n: {n}\n");

    for t in (0..n).rev() {
        let dist = 1 << t;
        let chunk_size = dist << 1;
        // eprintln!("target: {t}\nchunk size: {chunk_size}\ndistance: {dist}");

        if chunk_size == 2 {
            fft_chunk_2(state);
        } else if chunk_size == 4 {
            fft_chunk_4(state);
        } else {
            let (mut twiddles_re, mut twiddles_im) = generate_twiddles(dist);
            // eprintln!("twiddles_re: {twiddles_re:?}\ntwiddles:im: {twiddles_im:?}\nnow call fft_chunk_n\n------------------------------------");
            fft_chunk_n(state, &mut twiddles_re, &mut twiddles_im, dist);
        }
    }

    if n < 22 {
        cobra_apply(&mut state.reals, n);
        cobra_apply(&mut state.imags, n);
    } else {
        std::thread::scope(|s| {
            s.spawn(|| cobra_apply(&mut state.reals, n));
            s.spawn(|| cobra_apply(&mut state.imags, n));
        });
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rustfft::{num_complex::Complex64, FftPlanner};
    use spinoza::utils::assert_float_closeness;

    use super::*;

    #[test]
    fn fft() {
        let start = 2;
        let end = 16;
        let range = Range {
            start,
            end: end + 1,
        };

        for k in range {
            let n = 1 << k;

            let x_re: Vec<Float> = (1..=n).map(|i| i as Float).collect();
            let x_im = (1..=n).map(|i| i as Float).collect();
            let mut state = State {
                reals: x_re,
                imags: x_im,
                n: k as u8,
            };
            fft_dif(&mut state);

            let mut buffer: Vec<Complex64> = (1..=n)
                .map(|i| Complex64::new(f64::from(i), f64::from(i)))
                .collect();

            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(buffer.len());
            fft.process(&mut buffer);

            state
                .reals
                .iter()
                .zip(state.imags.iter())
                .enumerate()
                .for_each(|(i, (z_re, z_im))| {
                    let expect_re = buffer[i].re;
                    let expect_im = buffer[i].im;
                    assert_float_closeness(*z_re, expect_re, 0.001);
                    assert_float_closeness(*z_im, expect_im, 0.001);
                });
            println!("============================================================\n");
        }
    }
}
