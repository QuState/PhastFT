#![feature(portable_simd)]

use std::simd::prelude::*;

use spinoza::{core::State, math::Float};

use crate::benchmark::bm_fft;
use crate::cobra::cobra_apply;
use crate::twiddles::generate_twiddles;

mod benchmark;
mod bravo;
mod cobra;
mod twiddles;

fn fft_chunk_n_simd(state: &mut State, twiddles_re: &[Float], twiddles_im: &[Float], dist: usize) {
    let chunk_size = dist << 1;
    assert!(chunk_size >= 16);

    state
        .reals
        .chunks_exact_mut(chunk_size)
        .zip(state.imags.chunks_exact_mut(chunk_size))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

            reals_s0
                .chunks_exact_mut(8)
                .zip(reals_s1.chunks_exact_mut(8))
                .zip(imags_s0.chunks_exact_mut(8))
                .zip(imags_s1.chunks_exact_mut(8))
                .zip(twiddles_re.chunks_exact(8))
                .zip(twiddles_im.chunks_exact(8))
                .for_each(|(((((re_s0, re_s1), im_s0), im_s1), w_re), w_im)| {
                    let real_c0 = f64x8::from_slice(re_s0);
                    let real_c1 = f64x8::from_slice(re_s1);
                    let imag_c0 = f64x8::from_slice(im_s0);
                    let imag_c1 = f64x8::from_slice(im_s1);

                    let tw_re = f64x8::from_slice(w_re);
                    let tw_im = f64x8::from_slice(w_im);

                    re_s0.copy_from_slice((real_c0 + real_c1).as_array());
                    im_s0.copy_from_slice((imag_c0 + imag_c1).as_array());
                    let v_re = real_c0 - real_c1;
                    let v_im = imag_c0 - imag_c1;
                    re_s1.copy_from_slice((v_re * tw_re - v_im * tw_im).as_array());
                    im_s1.copy_from_slice((v_re * tw_im + v_im * tw_re).as_array());
                });
        });
}

// TODO(saveliy): parallelize
fn fft_chunk_n(state: &mut State, twiddles_re: &[Float], twiddles_im: &[Float], dist: usize) {
    let chunk_size = dist << 1;

    state
        .reals
        .chunks_exact_mut(chunk_size)
        .zip(state.imags.chunks_exact_mut(chunk_size))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

            reals_s0
                .iter_mut()
                .zip(reals_s1.iter_mut())
                .zip(imags_s0.iter_mut())
                .zip(imags_s1.iter_mut())
                .zip(twiddles_re.iter())
                .zip(twiddles_im.iter())
                .for_each(|(((((re_s0, re_s1), im_s0), im_s1), w_re), w_im)| {
                    let real_c0 = *re_s0;
                    let real_c1 = *re_s1;
                    let imag_c0 = *im_s0;
                    let imag_c1 = *im_s1;

                    *re_s0 = real_c0 + real_c1;
                    *im_s0 = imag_c0 + imag_c1;
                    let v_re = real_c0 - real_c1;
                    let v_im = imag_c0 - imag_c1;
                    *re_s1 = v_re * w_re - v_im * w_im;
                    *im_s1 = v_re * w_im + v_im * w_re;
                });
        });
}

/// chunk_size == 4, so hard code twiddle factors
fn fft_chunk_4(state: &mut State) {
    let dist = 2;
    let chunk_size = dist << 1;

    state
        .reals
        .chunks_exact_mut(chunk_size)
        .zip(state.imags.chunks_exact_mut(chunk_size))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

            let real_c0 = reals_s0[0];
            let real_c1 = reals_s1[0];
            let imag_c0 = imags_s0[0];
            let imag_c1 = imags_s1[0];

            reals_s0[0] = real_c0 + real_c1;
            imags_s0[0] = imag_c0 + imag_c1;
            reals_s1[0] = real_c0 - real_c1;
            imags_s1[0] = imag_c0 - imag_c1;

            let real_c0 = reals_s0[1];
            let real_c1 = reals_s1[1];
            let imag_c0 = imags_s0[1];
            let imag_c1 = imags_s1[1];

            reals_s0[1] = real_c0 + real_c1;
            imags_s0[1] = imag_c0 + imag_c1;
            reals_s1[1] = imag_c0 - imag_c1;
            imags_s1[1] = -(real_c0 - real_c1);
        });
}

/// chunk_size == 2, so skip phase
fn fft_chunk_2(state: &mut State) {
    let dist = 1;
    state
        .reals
        .chunks_exact_mut(2)
        .zip(state.imags.chunks_exact_mut(2))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

            reals_s0
                .iter_mut()
                .zip(reals_s1.iter_mut())
                .zip(imags_s0.iter_mut())
                .zip(imags_s1.iter_mut())
                .for_each(|(((re_s0, re_s1), im_s0), im_s1)| {
                    let real_c0 = *re_s0;
                    let real_c1 = *re_s1;
                    let imag_c0 = *im_s0;
                    let imag_c1 = *im_s1;

                    *re_s0 = real_c0 + real_c1;
                    *im_s0 = imag_c0 + imag_c1;
                    *re_s1 = real_c0 - real_c1;
                    *im_s1 = imag_c0 - imag_c1;
                });
        });
}

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

fn main() {
    let n = 31;
    bm_fft(n);
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rustfft::{num_complex::Complex64, FftPlanner};
    use spinoza::utils::assert_float_closeness;

    use super::*;

    #[test]
    fn fft() {
        let range = Range { start: 2, end: 19 };

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
                    assert_float_closeness(*z_re, expect_re, 0.1);
                    assert_float_closeness(*z_im, expect_im, 0.1);
                });
        }
    }
}
