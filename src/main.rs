#![feature(portable_simd)]

use std::simd::prelude::*;

use spinoza::core::State;
use spinoza::math::{Float, PI};
use spinoza::utils::pretty_print_int;

use crate::bravo::bravo;

mod bravo;

fn br_perm<T>(buf: &mut [T]) {
    let n = buf.len();
    let shift = (n - 1).leading_zeros();

    for i in 1..n {
        let j = i.reverse_bits() >> shift;

        if i < j {
            buf.swap(i, j);
        }
    }
}

fn bit_reverse_permute_state_par(state: &mut State) {
    std::thread::scope(|s| {
        s.spawn(|| br_perm(&mut state.reals));
        s.spawn(|| br_perm(&mut state.imags));
    });
}

// fn bit_reverse_permute_state_par(state: &mut State) {
//     std::thread::scope(|s| {
//         bit_reverse_permutation(&mut state.reals);
//         bit_reverse_permutation(&mut state.imags);
//     });
// }

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

                    let twiddles_re = f64x8::from_slice(w_re);
                    let twiddles_im = f64x8::from_slice(w_im);

                    re_s0.copy_from_slice((real_c0 + real_c1).as_array());
                    im_s0.copy_from_slice((imag_c0 + imag_c1).as_array());
                    let v_re = real_c0 - real_c1;
                    let v_im = imag_c0 - imag_c1;
                    re_s1.copy_from_slice((v_re * twiddles_re - v_im * twiddles_im).as_array());
                    im_s1.copy_from_slice((v_re * twiddles_im + v_im * twiddles_re).as_array());
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
    let mut twiddles_re = Vec::with_capacity(1 << (n - 1));
    let mut twiddles_im = Vec::with_capacity(1 << (n - 1));
    twiddles_re.push(1.0);
    twiddles_im.push(0.0);

    for t in (0..n).rev() {
        let dist = 1 << t;
        let chunk_size = dist << 1;

        if chunk_size > 16 {
            fft_chunk_n_simd(state, &twiddles_re, &twiddles_im, dist);
        }
        if chunk_size == 2 {
            fft_chunk_2(state);
        } else if chunk_size == 4 {
            fft_chunk_4(state);
        } else {
            let angle = -PI / (dist as Float);
            let (st, ct) = angle.sin_cos();
            let wlen_re = ct;
            let wlen_im = st;

            (1..dist).for_each(|i| {
                let mut w_re = twiddles_re[i - 1];
                let mut w_im = twiddles_im[i - 1];
                let temp = w_re;
                w_re = w_re * wlen_re - w_im * wlen_im;
                w_im = temp * wlen_im + w_im * wlen_re;
                twiddles_re.push(w_re);
                twiddles_im.push(w_im);
            });
            fft_chunk_n(state, &twiddles_re, &twiddles_im, dist);
            twiddles_re.clear();
            twiddles_im.clear();
            twiddles_re.push(1.0);
            twiddles_im.push(0.0);
        }
    }
    bit_reverse_permute_state_par(state);
}

fn bm_fft(num_qubits: usize) {
    for i in 4..num_qubits {
        println!("run PhastFT with {i} qubits");
        let n = 1 << i;
        let x_re: Vec<Float> = (1..=n).map(|i| i as Float).collect();
        let x_im: Vec<Float> = (1..=n).map(|i| i as Float).collect();
        let mut state = State {
            reals: x_re,
            imags: x_im,
            n: num_qubits as u8,
        };

        let now = std::time::Instant::now();
        fft_dif(&mut state);
        let elapsed = pretty_print_int(now.elapsed().as_micros());
        println!("time elapsed: {elapsed} us");
    }
}

fn bm_brp(num_qubits: usize) {
    for i in (2..num_qubits).into_iter().map(|i| 1 << i) {
        let mut buf0: Vec<Float> = (0..i).map(|i| i as Float).collect();
        let now = std::time::Instant::now();
        bravo(&mut buf0);
        let elapsed = pretty_print_int(now.elapsed().as_micros());
        println!("BRAVO: {buf0:?}");
        println!("time elapsed: {elapsed}");

        let mut buf1: Vec<Float> = (0..i).map(|i| i as Float).collect();
        let now = std::time::Instant::now();
        bit_reverse_permutation(&mut buf1);
        let elapsed = pretty_print_int(now.elapsed().as_micros());
        println!("naive BR: {buf1:?}");
        println!("time elapsed: {elapsed}\n-----------------------");
        assert_eq!(buf0, buf1);
    }
}

fn main() {
    bm_fft(26);
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rustfft::{num_complex::Complex64, FftPlanner};
    use spinoza::utils::assert_float_closeness;

    use super::*;

    #[test]
    fn bit_reversal() {
        let N = 16;
        let mut buf = (0..N).collect::<Vec<usize>>();

        br_perm(&mut buf);
        println!("{:?}", buf);
        assert_eq!(
            buf,
            vec![0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
        );

        let N = 8;
        let mut buf = (0..N).collect::<Vec<usize>>();
        br_perm(&mut buf);
        println!("{:?}", buf);
        assert_eq!(buf, vec![0, 4, 2, 6, 1, 5, 3, 7]);
    }

    #[test]
    fn bit_reversal_32_points() {
        let N = 32;
        let mut buf = (0..N).collect::<Vec<usize>>();
        br_perm(&mut buf);
        //bit_reverse_permutation(&mut buf);
        println!("{:?}", buf);
    }

    #[test]
    fn fft() {
        let range = Range { start: 2, end: 18 };

        for k in range {
            let n = 1 << k;

            let x_re: Vec<Float> = (1..n + 1).map(|i| i as Float).collect();
            let x_im = (1..n + 1).map(|i| i as Float).collect();
            let mut state = State {
                reals: x_re,
                imags: x_im,
                n: k as u8,
            };
            fft_dif(&mut state);

            let mut buffer: Vec<Complex64> = (1..n + 1)
                .map(|i| Complex64::new(i as f64, i as f64))
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
