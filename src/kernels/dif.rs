//! DIF-specific FFT Kernels
//!
//! This module contains SIMD-optimized butterfly kernels specifically for
//! the Decimation-in-Frequency algorithm.

use num_traits::Float;
use wide::{f32x16, f64x8};

/// SIMD-optimized DIF butterfly for f64
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    "aarch64+neon",
))]
#[inline]
pub fn fft_64_chunk_n_simd(
    reals: &mut [f64],
    imags: &mut [f64],
    twiddles_re: &[f64],
    twiddles_im: &[f64],
    dist: usize,
) {
    const LANES: usize = 8;
    let chunk_size = dist << 1;
    assert!(chunk_size >= LANES * 2);

    reals
        .chunks_exact_mut(chunk_size)
        .zip(imags.chunks_exact_mut(chunk_size))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

            reals_s0
                .chunks_exact_mut(LANES)
                .zip(reals_s1.chunks_exact_mut(LANES))
                .zip(imags_s0.chunks_exact_mut(LANES))
                .zip(imags_s1.chunks_exact_mut(LANES))
                .zip(twiddles_re.chunks_exact(LANES))
                .zip(twiddles_im.chunks_exact(LANES))
                .for_each(|(((((re_s0, re_s1), im_s0), im_s1), w_re), w_im)| {
                    let real_c0 = f64x8::new(re_s0[0..8].try_into().unwrap());
                    let real_c1 = f64x8::new(re_s1[0..8].try_into().unwrap());
                    let imag_c0 = f64x8::new(im_s0[0..8].try_into().unwrap());
                    let imag_c1 = f64x8::new(im_s1[0..8].try_into().unwrap());

                    let tw_re = f64x8::new(w_re[0..8].try_into().unwrap());
                    let tw_im = f64x8::new(w_im[0..8].try_into().unwrap());

                    re_s0.copy_from_slice((real_c0 + real_c1).as_array_ref());
                    im_s0.copy_from_slice((imag_c0 + imag_c1).as_array_ref());
                    let v_re = real_c0 - real_c1;
                    let v_im = imag_c0 - imag_c1;
                    re_s1.copy_from_slice((v_re * tw_re - v_im * tw_im).as_array_ref());
                    im_s1.copy_from_slice((v_re * tw_im + v_im * tw_re).as_array_ref());
                });
        });
}

/// SIMD-optimized DIF butterfly for f32
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    "aarch64+neon",
))]
#[inline]
pub fn fft_32_chunk_n_simd(
    reals: &mut [f32],
    imags: &mut [f32],
    twiddles_re: &[f32],
    twiddles_im: &[f32],
    dist: usize,
) {
    const LANES: usize = 16;
    let chunk_size = dist << 1;
    assert!(chunk_size >= LANES * 2);

    reals
        .chunks_exact_mut(chunk_size)
        .zip(imags.chunks_exact_mut(chunk_size))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

            reals_s0
                .chunks_exact_mut(LANES)
                .zip(reals_s1.chunks_exact_mut(LANES))
                .zip(imags_s0.chunks_exact_mut(LANES))
                .zip(imags_s1.chunks_exact_mut(LANES))
                .zip(twiddles_re.chunks_exact(LANES))
                .zip(twiddles_im.chunks_exact(LANES))
                .for_each(|(((((re_s0, re_s1), im_s0), im_s1), w_re), w_im)| {
                    let real_c0 = f32x16::new(re_s0[0..16].try_into().unwrap());
                    let real_c1 = f32x16::new(re_s1[0..16].try_into().unwrap());
                    let imag_c0 = f32x16::new(im_s0[0..16].try_into().unwrap());
                    let imag_c1 = f32x16::new(im_s1[0..16].try_into().unwrap());

                    let tw_re = f32x16::new(w_re[0..16].try_into().unwrap());
                    let tw_im = f32x16::new(w_im[0..16].try_into().unwrap());

                    re_s0.copy_from_slice((real_c0 + real_c1).as_array_ref());
                    im_s0.copy_from_slice((imag_c0 + imag_c1).as_array_ref());
                    let v_re = real_c0 - real_c1;
                    let v_im = imag_c0 - imag_c1;
                    re_s1.copy_from_slice((v_re * tw_re - v_im * tw_im).as_array_ref());
                    im_s1.copy_from_slice((v_re * tw_im + v_im * tw_re).as_array_ref());
                });
        });
}

/// General-purpose DIF butterfly for arbitrary chunk sizes
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86_64+avx2+fma",
    "x86_64+sse4.2",
    "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    "x86+avx2+fma",
    "x86+sse4.2",
    "x86+sse2",
    "aarch64+neon",
))]
#[inline]
pub fn fft_chunk_n<T: Float>(
    reals: &mut [T],
    imags: &mut [T],
    twiddles_re: &[T],
    twiddles_im: &[T],
    dist: usize,
) {
    let chunk_size = dist << 1;

    reals
        .chunks_exact_mut(chunk_size)
        .zip(imags.chunks_exact_mut(chunk_size))
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
                    *re_s1 = v_re * *w_re - v_im * *w_im;
                    *im_s1 = v_re * *w_im + v_im * *w_re;
                });
        });
}
