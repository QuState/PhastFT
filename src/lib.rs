#![feature(portable_simd)]
use crate::kernels::{fft_chunk_2, fft_chunk_4, fft_chunk_n, fft_chunk_n_simd};
use crate::{cobra::cobra_apply, twiddles::generate_twiddles};
use spinoza::{core::State, math::Float};
use std::simd::prelude::*;

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

    for t in (0..n).rev() {
        let dist = 1 << t;
        let chunk_size = dist << 1;

        if chunk_size > 4 {
            let (twiddles_re, twiddles_im) = generate_twiddles(dist);
            if chunk_size >= 16 {
                fft_chunk_n_simd(state, &twiddles_re, &twiddles_im, dist);
            } else {
                fft_chunk_n(state, &twiddles_re, &twiddles_im, dist);
            }
        } else if chunk_size == 2 {
            fft_chunk_2(state);
        } else if chunk_size == 4 {
            fft_chunk_4(state);
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
        let range = Range { start: 2, end: 17 };

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
        }
    }
}
