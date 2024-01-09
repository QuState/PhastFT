use std::f64::consts::PI;
use std::ops::Range;

use rayon::prelude::*;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;
use spinoza::core::State;
use spinoza::math::{Float, SQRT_ONE_HALF};
use spinoza::utils::{assert_float_closeness, pretty_print_int};

fn padded_bin(n: usize, k: usize) -> usize {
    let reversed_str = format!("{:0width$b}", k, width = n)
        .chars()
        .rev()
        .collect::<String>();
    usize::from_str_radix(&reversed_str, 2).unwrap()
}

fn bit_reverse_permutation(state: &mut State) {
    let n = state.len();
    let mut j = 0;

    for i in 1..n {
        let mut bit = n >> 1;

        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;

        if i < j {
            state.reals.swap(i, j);
            state.imags.swap(i, j);
        }
    }
}

// fn fft(x_re: &mut [f64], x_im: &mut [f64]) {
//     assert_eq!(x_re.len(), x_im.len());
//     let n = x_re.len();
//     assert!(n.is_power_of_two(), "n must be a power of 2");
//
//     // DFT
//     let N = x_re.len();
//     let mut k = N;
//     let mut n;
//
//     let theta_t = PI / N as f64;
//     let mut c = theta_t.cos();
//     let mut d = -(theta_t.sin());
//
//     // (a + ib) * (c + id) = ac + iad + ibc - bd = ac - bd + i(ad + bc)
//     // Complex phiT = Complex(cos(thetaT), -sin(thetaT)), T;
//     while k > 1 {
//         n = k;
//         k >>= 1;
//         let phiT_re = c * c - d * d;
//         let phiT_im = c * d + d * c;
//         c = phiT_re;
//         d = phiT_im;
//         let (mut T_re, mut T_im) = (1.0, 0.0);
//
//         for l in 0..k {
//             let mut a = l;
//             while a < N {
//                 let b = a + k;
//                 let t_re = x_re[a] - x_re[b];
//                 let t_im = x_im[a] - x_im[b];
//                 x_re[a] += x_re[b];
//                 x_im[a] += x_im[b];
//                 x_re[b] = t_re.mul_add(T_re, -t_im * T_im);
//                 x_im[b] = t_re.mul_add(T_im, -t_im * T_re);
//                 a += n;
//             }
//
//             // (a + ib) * (c + id) = ac + iad + ibc - bd = ac - bd + i(ad + bc)
//             let (w, x, y, z) = (T_re, T_im, phiT_re, phiT_im);
//             T_re = w.mul_add(y, -x * z);
//             T_im = w.mul_add(z, x * y);
//         }
//     }
// }

fn fft_chunk_n(
    state: &mut State,
    twiddles_re: &[Float],
    twiddles_im: &[Float],
    dist: usize,
    chunk_size: usize,
) {
    state
        .reals
        .par_chunks_exact_mut(chunk_size)
        .zip(state.imags.par_chunks_exact_mut(chunk_size))
        .with_max_len(1 << 11)
        .enumerate()
        //.with_max_len(1 << 11)
        .for_each(|(_c, (reals_chunk, imags_chunk))| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);
            // eprintln!("chunk #: {c}");

            reals_s0
                .par_iter_mut()
                .zip_eq(reals_s1.par_iter_mut())
                .zip_eq(imags_s0.par_iter_mut())
                .zip_eq(imags_s1.par_iter_mut())
                .zip_eq(twiddles_re.par_iter())
                .zip_eq(twiddles_im.par_iter())
                .with_min_len(1 << 15)
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
    let twiddles_re = [1.0, SQRT_ONE_HALF, 0.0, -SQRT_ONE_HALF];
    let twiddles_re = [0.0, -SQRT_ONE_HALF, -1.0, SQRT_ONE_HALF];

    state
        .reals
        .par_chunks_exact_mut(4)
        .zip(state.imags.par_chunks_exact_mut(4))
        .with_max_len(1 << 15)
        //.with_max_len(1 << 11)
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);
            //eprintln!("chunk #: {c}");

            let real_c0 = reals_s0[0];
            let real_c1 = reals_s1[0];
            let imag_c0 = imags_s0[0];
            let imag_c1 = imags_s1[0];

            reals_s0[0] = real_c0 + real_c1;
            reals_s1[0] = imag_c0 + imag_c1;
            imags_s0[0] = real_c0 - real_c1;
            imags_s1[0] = imag_c0 - imag_c1;

            let real_c0 = reals_s0[1];
            let real_c1 = reals_s1[1];
            let imag_c0 = imags_s0[1];
            let imag_c1 = imags_s1[1];

            reals_s0[1] = real_c0 + real_c1;
            reals_s1[1] = imag_c0 + imag_c1;
            imags_s0[1] = real_c0 - real_c1;
            imags_s1[1] = imag_c0 - imag_c1;
        });
}

/// chunk_size == 2, so skip phase
fn fft_chunk_2(state: &mut State) {
    let dist = 1;
    state
        .reals
        .par_chunks_exact_mut(2)
        .zip(state.imags.par_chunks_exact_mut(2))
        .with_max_len(1 << 16)
        //.with_max_len(1 << 11)
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);
            //eprintln!("chunk #: {c}");

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

/// FFT -- Decimation in Time (DIT)
pub fn fft_dit(state: &mut State) {
    let n: usize = state.n.into();
    let mut twiddles_re: Vec<_> = Vec::with_capacity(1 << (n - 1));
    let mut twiddles_im: Vec<_> = Vec::with_capacity(1 << (n - 1));

    for t in (0..n).rev() {
        let chunk_size = 1 << (t + 1);
        let dist = 1 << t;

        let angle = -PI / (dist as Float);
        let (st, ct) = angle.sin_cos();
        let wlen_re = ct;
        let wlen_im = st;
        //eprintln!("stage: {t} --> chunk size: {chunk_size}");

        if chunk_size != 2 {
            twiddles_re.push(1.0);
            twiddles_im.push(0.0);

            (1..dist).for_each(|i| {
                let mut w_re = twiddles_re[i - 1];
                let mut w_im = twiddles_im[i - 1];
                let temp = w_re;
                w_re = w_re * wlen_re - w_im * wlen_im;
                w_im = temp * wlen_im + w_im * wlen_re;
                twiddles_re.push(w_re);
                twiddles_im.push(w_im);
                // eprintln!("{w_re} + i{w_im}");
            });
            fft_chunk_n(state, &twiddles_re, &twiddles_im, dist, chunk_size);
            twiddles_re.clear();
            twiddles_im.clear();
        } else {
            fft_chunk_2(state);
        }
    }
    bit_reverse_permutation(state);
}

fn main() {
    let range = std::ops::Range { start: 5, end: 6 };

    for k in range {
        let n = 1 << k;

        let now = std::time::Instant::now();
        let x_re: Vec<Float> = (1..n + 1).map(|i| i as Float).collect();
        let x_im = (1..n + 1).map(|i| i as Float).collect();
        let mut state = State {
            reals: x_re,
            imags: x_im,
            n: k as u8,
        };
        fft_dit(&mut state);
        // fft(&mut state.reals, &mut state.imags);
        let elapsed = pretty_print_int(now.elapsed().as_micros());
        eprintln!("# qubits: {k}\nqifft time elapsed {elapsed} us");
        // state
        //     .reals
        //     .iter()
        //     .zip(state.imags.iter())
        //     .for_each(|(z_re, z_im)| {
        //         eprintln!("{z_re} + i{z_im}");
        //     });

        let mut buffer: Vec<Complex64> = (1..n + 1)
            .map(|i| Complex64::new(i as f64, i as f64))
            .collect();

        let now = std::time::Instant::now();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(buffer.len());
        fft.process(&mut buffer);
        let elapsed = pretty_print_int(now.elapsed().as_micros());
        eprintln!("RustFFT time elapsed {elapsed} us");
        // for z in buffer.iter() {
        //     let z_re = z.re;
        //     let z_im = z.im;
        //     eprintln!("{z_re} + i{z_im}");
        // }
        eprintln!("--------------");

        // state
        //     .reals
        //     .iter()
        //     .zip(state.imags.iter())
        //     .enumerate()
        //     .for_each(|(i, (z_re, z_im))| {
        //         // let i_r = padded_bin(state.len().ilog2() as usize, i);
        //         let expect_re = buffer[i].re;
        //         let expect_im = buffer[i].im;
        //         assert_float_closeness(*z_re, expect_re, 0.1);
        //         assert_float_closeness(*z_im, expect_im, 0.1);
        //     });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reversal() {
        // let N: usize = 1 << 3;
        // let n = N.ilog2() as usize;
        //
        // for k in N..N + 1 {
        //     bit_reverse_permutation(k);
        //     println!("---------------------");
        // }
    }

    #[test]
    fn fft() {
        let range = Range { start: 2, end: 26 };

        for k in range {
            let n = 1 << k;

            let x_re: Vec<Float> = (1..n + 1).map(|i| i as Float).collect();
            let x_im = (1..n + 1).map(|i| i as Float).collect();
            let mut state = State {
                reals: x_re,
                imags: x_im,
                n: k as u8,
            };
            fft_dit(&mut state);

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
