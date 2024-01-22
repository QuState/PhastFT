#![feature(portable_simd)]

use std::simd::prelude::*;

use spinoza::{core::State, math::Float};

use crate::benchmark::bm_fft;
use crate::cobra::{cobra_apply, LOG_BLOCK_WIDTH};
use crate::twiddles::generate_twiddles;

mod benchmark;
mod bravo;
mod cobra;
mod twiddles;

// Source: https://www.katjaas.nl/bitreversal/bitreversal.html
fn bit_rev(buf: &mut [Float], logN: usize) {
    let mut nodd: usize;
    let mut noddrev; // to hold bitwise negated or odd values

    let N = 1 << logN;
    let halfn = N >> 1; // frequently used 'constants'
    let quartn = N >> 2;
    let nmin1 = N - 1;

    let mut forward = halfn; // variable initialisations
    let mut rev = 1;

    let mut i = quartn;
    while i > 0 {
        // start of bitreversed permutation loop, N/4 iterations

        // Gray code generator for even values:

        nodd = !i; // counting ones is easier

        let mut zeros = 0;
        while (nodd & 1) == 1 {
            nodd >>= 1;
            zeros += 1;
        }

        forward ^= 2 << zeros; // toggle one bit of forward
        rev ^= quartn >> zeros; // toggle one bit of rev

        // swap even and ~even conditionally
        if forward < rev {
            buf.swap(forward, rev);
            nodd = nmin1 ^ forward; // compute the bitwise negations
            noddrev = nmin1 ^ rev;
            buf.swap(nodd, noddrev); // swap bitwise-negated pairs
        }

        nodd = forward ^ 1; // compute the odd values from the even
        noddrev = rev ^ halfn;
        // swap(nodd, noddrev, real, im); // swap odd unconditionally
        buf.swap(nodd, noddrev);
        i -= 1;
    }
}

fn complex_bit_rev(state: &mut State, logN: usize) {
    let mut nodd: usize;
    let mut noddrev; // to hold bitwise negated or odd values

    let N = 1 << logN;
    let halfn = N >> 1; // frequently used 'constants'
    let quartn = N >> 2;
    let nmin1 = N - 1;

    let mut forward = halfn; // variable initialisations
    let mut rev = 1;

    let mut i = quartn;
    while i > 0 {
        // start of bitreversed permutation loop, N/4 iterations

        // Gray code generator for even values:

        nodd = !i; // counting ones is easier

        let mut zeros = 0;
        while (nodd & 1) == 1 {
            nodd >>= 1;
            zeros += 1;
        }

        forward ^= 2 << zeros; // toggle one bit of forward
        rev ^= quartn >> zeros; // toggle one bit of rev

        // swap even and ~even conditionally
        if forward < rev {
            state.reals.swap(forward, rev);
            state.imags.swap(forward, rev);
            nodd = nmin1 ^ forward; // compute the bitwise negations
            noddrev = nmin1 ^ rev;

            // swap bitwise-negated pairs
            state.reals.swap(nodd, noddrev);
            state.imags.swap(nodd, noddrev);
        }

        nodd = forward ^ 1; // compute the odd values from the even
        noddrev = rev ^ halfn;

        // swap odd unconditionally
        state.reals.swap(nodd, noddrev);
        state.imags.swap(nodd, noddrev);
        i -= 1;
    }
}

fn bit_reverse_permute_state_par(state: &mut State) {
    std::thread::scope(|s| {
        s.spawn(|| bit_rev(&mut state.reals, state.n as usize));
        s.spawn(|| bit_rev(&mut state.imags, state.n as usize));
    });
}

fn bit_reverse_permute_state_seq(state: &mut State) {
    complex_bit_rev(state, state.n as usize);
}

fn bit_reverse_permutation<T>(buf: &mut [T]) {
    let n = buf.len();
    let mut j = 0;

    for i in 1..n {
        let mut bit = n >> 1;

        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;

        if i < j {
            buf.swap(i, j);
            // println!("{i}, {j}");
        }
    }
}

fn fft_chunk_n_simd(state: &mut State, twiddles_re: &[Float], twiddles_im: &[Float], dist: usize) {
    let chunk_size = dist << 1;
    assert!(chunk_size >= 16);
    let mut scratch = [0.0; 16];

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

    if n <= 2 * LOG_BLOCK_WIDTH {
        bit_reverse_permute_state_seq(state);
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
    // benchmark_bit_reversal_permutation();
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rustfft::{num_complex::Complex64, FftPlanner};
    use spinoza::utils::assert_float_closeness;

    use super::*;

    #[test]
    fn bit_reversal() {
        let n = 3;
        let N = 1 << n;
        let mut buf: Vec<Float> = (0..N).map(|i| i as Float).collect();
        bit_rev(&mut buf, n);
        println!("{buf:?}");
        assert_eq!(buf, vec![0.0, 4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0]);

        let n = 4;
        let N = 1 << n;
        let mut buf: Vec<Float> = (0..N).map(|i| i as Float).collect();
        bit_rev(&mut buf, n);
        println!("{buf:?}");
        assert_eq!(
            buf,
            vec![
                0.0, 8.0, 4.0, 12.0, 2.0, 10.0, 6.0, 14.0, 1.0, 9.0, 5.0, 13.0, 3.0, 11.0, 7.0,
                15.0,
            ]
        );
    }

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
